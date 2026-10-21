#!/usr/bin/env python3
"""Finite, unbiased ds41_generate protocol benchmark (no server or downloads).

Example:
  python tools/v41/bench/benchmark.py --engine /path/ds41_generate \
    --engine-revision COMMIT --model /path/first.gguf --config /path/config.json \
    --ids-file prompt.json \
    --output build-bench/baseline --slots 512 --arena-gib 16

The IDs file is a JSON array of token IDs. Use the same file across comparisons.
--repeats 2 measures process-cold then process-warm full prefill; prefix reuse is
disabled. OS page-cache warmth is uncontrolled: this is NOT a cold-disk test.
Each invocation loads exactly one model, on physical GPU0. Logs, token IDs,
hardware, actual engine diagnostics and the exact command accompany result.json.
No retries or smaller-configuration fallbacks occur on failure.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time


GIB = 1024 ** 3
DIAGNOSTIC_FAILURE = re.compile(r"out of memory|CUDA error|failed|\bOOM\b", re.I)


def available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / GIB
    raise RuntimeError("MemAvailable unavailable")


def capture(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError(f"{command!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def process_io(pid):
    """Linux task I/O includes mmap page faults; read_bytes is storage, not PCIe."""
    try:
        return {key: int(value) for key, value in
                (line.split(":") for line in Path(f"/proc/{pid}/io").read_text().splitlines())}
    except (OSError, ValueError):
        return None


def io_delta(before, after):
    return {key: after[key] - value for key, value in before.items()
            if key in after} if before is not None and after is not None else None


def memory_kib(pid):
    try:
        return {key: int(value.split()[0]) for key, value in
                (line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                 if line.startswith(("VmRSS:", "VmHWM:", "VmLck:")))}
    except (OSError, ValueError):
        return {}


def parse_done(line, observed, expected, prompt_tokens):
    fields = line.split()
    if len(fields) != 15 or fields[0] != "DONE":
        raise RuntimeError(f"malformed DONE: {line}")
    generated, total = int(fields[1]), int(fields[2])
    prefill_ms, decode_ms = float(fields[3]), float(fields[4])
    reused, hits, lookups = int(fields[8]), int(fields[9]), int(fields[10])
    if generated != observed or generated != expected or total != prompt_tokens:
        raise RuntimeError(f"incomplete generation: {line}, observed={observed}")
    if fields[5] != "length" or reused or int(fields[14]) != total:
        raise RuntimeError(f"unexpected finish or prefix reuse: {line}")
    if (not all(math.isfinite(v) and v > 0 for v in (prefill_ms, decode_ms))
            or not 0 <= hits <= lookups):
        raise RuntimeError(f"invalid timing/counters: {line}")
    return {
        "input_tokens": total, "output_tokens": generated, "reused_tokens": reused,
        "prefill_ms": prefill_ms, "decode_ms": decode_ms,
        "prefill_tokens_per_second": total * 1000 / prefill_ms,
        # First sampled token needs no decode step, matching the engine's metric.
        "decode_tokens_per_second": (generated - 1) * 1000 / decode_ms,
        "expert_hits": hits, "expert_lookups": lookups,
        "expert_hit_fraction": hits / lookups if lookups else None,
        "file_routes": int(fields[12]),
    }


def command(args):
    argv = [str(args.engine), "-m", str(args.model), "--serve", "--backend", "cuda",
            "--experts", "gpu", "--slots", args.slots,
            "--vram-margin", str(args.vram_margin), "--arena-gib", str(args.arena_gib),
            "--pcie", str(args.pcie), "--pf-b", str(args.pf_b),
            "--prompt-batch", str(args.prompt_batch), "--prefill-chunk", str(args.chunk),
            "--threads", str(args.threads), "--ctx", str(args.ctx),
            "--temp", "0", "--seed", str(args.seed), "--route-bias", "0",
            "--route-bias-file", "0", "--skip-miss", "0", "--skip-file", "0",
            "--no-enc-skip", "--no-prefix-cache", "--vram-lru"]
    if args.chunk_batch:
        argv.append("--chunk-batch")
    return argv


def run_protocol(args, ids, result):
    events = queue.Queue()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("STRATA_", "DS41_", "GGML_"))}
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["STRATA_V41_CONFIG"] = str(args.config)
    if args.step_max is not None:
        env["STRATA_SERVE_STEP_MAX"] = str(args.step_max)
    switches = {
        "STRATA_DEVICE_ACTIVATIONS": args.device_activations,
        "STRATA_TIER_GRAPH": args.tier_graph,
        "STRATA_DMA_OVERLAP": args.dma_overlap,
        "GGML_CUDA_DISABLE_GRAPHS": args.disable_cuda_graphs,
    }
    for name, enabled in switches.items():
        if enabled:
            env[name] = "1"
    result["environment"] = {"CUDA_VISIBLE_DEVICES": "0",
                             "STRATA_V41_CONFIG": str(args.config),
                             "STRATA_SERVE_STEP_MAX": env.get("STRATA_SERVE_STEP_MAX"),
                             **{name: env.get(name) for name in switches},
                             "engine_overrides": "all other STRATA_*, DS41_*, GGML_* removed"}
    start = time.monotonic()
    process = subprocess.Popen(result["command"], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, bufsize=1, env=env, start_new_session=True)
    result["pid"] = process.pid
    readers = []

    def pump(stream, name):
        with (args.output / f"{name}.log").open("w") as log:
            for line in stream:
                log.write(line)
                log.flush()
                events.put((name, line.strip(), time.monotonic()))
        events.put((name, None, time.monotonic()))

    for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
        thread = threading.Thread(target=pump, args=(stream, name), daemon=True)
        thread.start()
        readers.append(thread)

    deadline = start + args.timeout
    result["minimum_mem_available_gib"] = available_gib()
    result["peak_process_memory_kib"] = {}

    def event():
        while True:
            for key, value in memory_kib(process.pid).items():
                result["peak_process_memory_kib"][key] = max(
                    result["peak_process_memory_kib"].get(key, 0), value)
            free = available_gib()
            result["minimum_mem_available_gib"] = min(result["minimum_mem_available_gib"], free)
            if free <= args.min_available_gib:
                raise RuntimeError(f"memory pressure: MemAvailable={free:.2f} GiB")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"run exceeded {args.timeout}s")
            try:
                name, line, timestamp = events.get(timeout=0.25)
            except queue.Empty:
                if process.poll() is not None:
                    raise RuntimeError(f"engine exited {process.returncode} before DONE")
                continue
            if line is None:
                if name == "stdout":
                    raise RuntimeError(f"engine stdout closed (exit={process.poll()})")
                continue
            if name == "stderr":
                if DIAGNOSTIC_FAILURE.search(line):
                    raise RuntimeError(f"engine diagnostic failure: {line}")
                continue
            if line.startswith("ERR"):
                raise RuntimeError(line)
            return line, timestamp

    try:
        while True:
            line, timestamp = event()
            if line.startswith("READY "):
                result["startup_ms"] = (timestamp - start) * 1000
                result["ready"] = line
                break
        for repeat in range(args.repeats):
            io_before = process_io(process.pid)
            request_start = time.monotonic()
            process.stdin.write(f"GEN {args.output_tokens} temperature=0 seed={args.seed} "
                                + ",".join(map(str, ids)) + "\n")
            process.stdin.flush()
            tokens, first = [], None
            while True:
                line, timestamp = event()
                if line.startswith("T "):
                    tokens.append(int(line.split()[1]))
                    first = timestamp if first is None else first
                elif line.startswith("DONE "):
                    measurement = parse_done(line, len(tokens), args.output_tokens, len(ids))
                    if first is None:
                        raise RuntimeError("DONE without tokens")
                    measurement.update({
                        "repeat": repeat,
                        "cache_state": "process-cold" if repeat == 0 else "process-warm",
                        "ttft_ms": (first - request_start) * 1000,
                        "request_wall_ms": (timestamp - request_start) * 1000,
                        "output_ids": tokens,
                        "process_io_delta": io_delta(io_before, process_io(process.pid)),
                    })
                    result["measurements"].append(measurement)
                    break
        process.stdin.write("QUIT\n")
        process.stdin.flush()
        process.stdin.close()
        process.wait(timeout=max(1, min(30, deadline - time.monotonic())))
        if process.returncode:
            raise RuntimeError(f"engine exit={process.returncode}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=10)
        if not process.stdin.closed:
            process.stdin.close()
        process.stdout.close()
        process.stderr.close()
        result["exit_code"] = process.returncode
        result["total_wall_seconds"] = time.monotonic() - start
        result["host_memory_after"] = Path("/proc/meminfo").read_text()
        result["diagnostic_tail"] = (args.output / "stderr.log").read_text()[-12000:]
        result["routing_diagnostics"] = [
            line for line in (args.output / "stderr.log").read_text().splitlines()
            if line.startswith("serve:") and ("experts:" in line or "pcie /" in line)]
    # The independent stderr reader can deliver a failure after stdout's DONE.
    for line in (args.output / "stderr.log").read_text().splitlines():
        if DIAGNOSTIC_FAILURE.search(line):
            raise RuntimeError(f"engine diagnostic failure: {line}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-revision", required=True, help="revision used to build the executable")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="exact local HF config.json")
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new artifact directory")
    parser.add_argument("--slots", default="512")
    parser.add_argument("--arena-gib", type=float, default=16)
    parser.add_argument("--vram-margin", type=float, default=2)
    parser.add_argument("--pcie", type=float, default=1)
    parser.add_argument("--pf-b", type=float, default=0)
    parser.add_argument("--prompt-batch", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--step-max", type=int,
                        help="explicit short-turn pass cutoff; 0 forces layer-major prefill")
    parser.add_argument("--chunk-batch", action="store_true", help="opt in to four-column expert kernels")
    parser.add_argument("--device-activations", action="store_true", help="opt in to device decode inputs")
    parser.add_argument("--tier-graph", action="store_true", help="opt in to expert-tier graph capture")
    parser.add_argument("--dma-overlap", action="store_true", help="opt in to expert demand-copy overlap")
    parser.add_argument("--disable-cuda-graphs", action="store_true",
                        help="force eager ggml execution in a graph-enabled build")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600, help="whole process deadline, seconds")
    parser.add_argument("--min-available-gib", type=float, default=40)
    args = parser.parse_args()
    if (not 0 < args.arena_gib <= 96
            or not math.isfinite(args.min_available_gib) or args.min_available_gib < 40):
        parser.error("arena must be (0,96] GiB and MemAvailable floor >=40 GiB")
    if args.slots != "auto" and (not args.slots.isdigit() or int(args.slots) <= 0):
        parser.error("slots must be auto or a positive integer")
    if (not all(math.isfinite(v) for v in
                (args.pcie, args.pf_b, args.vram_margin, args.timeout))
            or not 0 <= args.pcie <= 1 or args.pf_b < 0 or args.vram_margin < 1
            or min(args.prompt_batch, args.chunk, args.threads, args.repeats) < 1
            or (args.step_max is not None and args.step_max < 0)
            or args.output_tokens < 2 or args.timeout <= 0):
        parser.error("invalid configuration bounds")
    for attr in ("engine", "model", "config", "ids_file", "output"):
        setattr(args, attr, getattr(args, attr).resolve())
    ids = json.loads(args.ids_file.read_text())
    if (not isinstance(ids, list) or not ids
            or any(type(token) is not int or token < 0 for token in ids)
            or len(ids) + args.output_tokens > args.ctx):
        parser.error("ids-file must contain nonnegative integer IDs fitting ctx")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"schema": 1, "status": "error", "measurements": [],
              "engine_revision": args.engine_revision,
              "configuration": {k: str(v) if isinstance(v, Path) else v
                                for k, v in vars(args).items()},
              "command": command(args), "input_ids": ids,
              "page_cache": "uncontrolled; never claim disk-cold",
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        result["harness_revision"] = capture(["git", "rev-parse", "HEAD"])
        result["harness_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
        with args.engine.open("rb") as executable:
            result["engine_sha256"] = hashlib.file_digest(executable, "sha256").hexdigest()
        result["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
        result["model_shards"] = [
            {"path": str(p), "size_bytes": p.stat().st_size,
             "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(args.model.parent.glob("*.gguf"))]
        result["host_memory_before"] = Path("/proc/meminfo").read_text()
        result["gpu_hardware"] = capture(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,pci.bus_id", "--format=csv"])
        result["cpu_hardware"] = capture(["lscpu"])
        result["gpu_processes_before"] = capture(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"])
        gpu0 = result["gpu_hardware"].splitlines()[1].split(",")[1].strip()
        if any(gpu0 in line for line in result["gpu_processes_before"].splitlines()):
            raise RuntimeError("GPU0 already occupied; refusing checkpoint load")
        if available_gib() <= args.min_available_gib:
            raise RuntimeError("insufficient MemAvailable before launch")
        run_protocol(args, ids, result)
        result["status"] = "ok"
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
