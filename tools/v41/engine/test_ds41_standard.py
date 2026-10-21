"""Exercise the standard 1017 GGUF path without model downloads or Python packages."""

import argparse
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import struct
import subprocess
import threading


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def fixture(directory, gpu_experts=False, nonzero=False, batch_experts=False):
    dim = 5120 if batch_experts else 256 if gpu_experts else 128
    ff = 2304 if batch_experts else dim
    rank = 128 if batch_experts else dim
    prefix = "deepseek41."
    metadata = {"general.architecture": "deepseek41"}
    integers = {
        "embedding_length": dim, "block_count": 2, "attention.head_count": 1,
        "attention.key_length": 64, "rope.dimension_count": 16,
        "attention.q_lora_rank": rank, "attention.output_lora_rank": 64,
        "attention.output_group_count": 1, "attention.sliding_window": 4,
        "expert_count": 2, "expert_used_count": 1, "expert_feed_forward_length": ff,
        "expert_shared_count": 1, "hyper_connection.count": 4,
        "hyper_connection.sinkhorn_iterations": 4, "attention.indexer.head_count": 32,
        "attention.indexer.key_length": 128, "attention.indexer.top_k": 1,
        "context_length": 4096, "engram.head_count": 1, "engram.key_length": 32,
        "engram.max_ngram_size": 3, "engram.pad_id": 0, "expert_gating_func": 4,
        "expert_weights_norm": 1,
    }
    floats = {
        "expert_weights_scale": 1., "hyper_connection.epsilon": 1.e-6,
        "attention.layer_norm_rms_epsilon": 1.e-6, "rope.freq_base": 10000.,
        "attention.compress_rope_freq_base": 10000., "rope.scaling.factor": 1.,
        "rope.scaling.yarn_beta_fast": 32., "rope.scaling.yarn_beta_slow": 1.,
        "rope.scaling.original_context_length": 4096.,
    }
    metadata.update({prefix + key: value for key, value in {**integers, **floats}.items()})
    metadata.update({
        "tokenizer.ggml.bos_token_id": 0, "tokenizer.ggml.eos_token_id": 1,
        prefix + "attention.compress_ratios": [2, 2],
        prefix + "swiglu_clamp_exp": [10., 10.],
        prefix + "swiglu_clamp_shexp": [10., 10.],
        prefix + "engram.layer_ids": [0],
        prefix + "engram.token_map": [0, 1, 2, 3],
        prefix + "engram.primes": [5, 7],
        prefix + "engram.multipliers": [3, 5, 7],
    })
    config = {
        "candidate_source_layer_id": 0, "candidate_topk_blocks": 1,
        "candidate_block_size": 2, "hidden_size": dim, "num_hidden_layers": 2,
        "n_routed_experts": 2, "vocab_size": 4, "engram_head_dim": 32,
        "engram_n_heads": 1, "engram_max_ngram_size": 3, "swiglu_limit": 10.,
    }
    tensors = []

    def add(name, shape, type_id=0):
        count = 1
        for dim in shape:
            count *= dim
        if type_id == 8:
            payload = (b"".join(struct.pack("<e", 0.01) + bytes([1 + row % 31]) * 32
                                for row in range(count // 32)) if nonzero else bytes(count // 32 * 34))
        elif type_id in (10, 11):
            block = bytes(84 if type_id == 10 else 110)
            if nonzero:
                block = (bytes([0x11]) * 16 + bytes([0xe4]) * 64 + struct.pack("<ee", .02, .005)
                         if type_id == 10 else
                         bytes([0xaa]) * 32 + bytes([0xe4]) * 64 + bytes([0x21]) * 12 + struct.pack("<e", .02))
            payload = block * (count // 256)
        else:
            value = 1. if "norm" in name or "engram_q." in name or "engram_k." in name else 0.
            if nonzero and value == 0:
                payload = b"".join(struct.pack("<f", .015 * math.sin(i * .013 + len(tensors))) for i in range(count))
            else:
                payload = struct.pack("<f", value) * count
        tensors.append((name, shape, type_id, payload))

    add("token_embd.weight", [dim, 4])
    add("output_norm.weight", [dim])
    add("output.weight", [dim, 4])
    for layer in range(2):
        base = f"blk.{layer}."
        for site in ("attn", "ffn"):
            add(base + f"hc_{site}_fn.weight", [4 * dim, 24])
            add(base + f"hc_{site}_base.weight", [24])
            add(base + f"hc_{site}_scale.weight", [3])
            add(base + f"{site}_norm.weight", [dim])
        for name, shape in (
            ("attn_sinks", [1]), ("attn_q_a", [dim, rank]), ("attn_q_b", [rank, 64]),
            ("attn_q_a_norm", [rank]), ("attn_kv", [dim, 64]), ("attn_kv_a_norm", [64]),
            ("attn_output_a", [64, 64]), ("attn_output_b", [64, dim]),
            ("ffn_gate_inp", [dim, 2]),
            ("ffn_gate_shexp", [dim, ff]), ("ffn_up_shexp", [dim, ff]),
            ("ffn_down_shexp", [ff, dim]),
            ("ffn_gate_exps", [dim, ff, 2]), ("ffn_up_exps", [dim, ff, 2]),
            ("ffn_down_exps", [ff, dim, 2]),
        ):
            type_id = (11 if "down" in name else 10) if gpu_experts and "_exps" in name else 0
            add(base + name + ".weight", shape, type_id)
        add(base + "exp_probs_b.bias", [2])
        add(base + "exp_probs_b_vl.bias", [2])
    for name, shape in (
        ("attn_compressor_kv", [dim, 64]), ("attn_compressor_norm", [64]),
        ("attn_compressor_gate", [dim, 64]), ("indexer.attn_k", [64, 128]),
        ("indexer.k_norm", [128]), ("indexer.attn_q_b", [rank, 4096]),
        ("indexer.proj", [dim, 32]), ("engram_q", [dim, 4]),
        ("engram_k", [dim, 4]), ("engram_wkv", [64, 5 * dim]),
    ):
        add("blk.0." + name + ".weight", shape)
    add("blk.0.engram_embd.weight", [32, 12], 8)

    def encode(value):
        if isinstance(value, str):
            return 8, string(value)
        if isinstance(value, list):
            kind, _ = encode(value[0])
            return 9, struct.pack("<IQ", kind, len(value)) + b"".join(encode(v)[1] for v in value)
        if isinstance(value, float):
            return 12, struct.pack("<d", value)
        return 10, struct.pack("<Q", value)

    header = struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        kind, data = encode(value)
        header += string(key) + struct.pack("<I", kind) + data
    payload = bytearray()
    for name, shape, type_id, data in tensors:
        payload += bytes((-len(payload)) % 32)
        header += string(name) + struct.pack("<I", len(shape))
        header += struct.pack("<" + "Q" * len(shape), *shape)
        header += struct.pack("<IQ", type_id, len(payload))
        payload += data
    header += bytes((-len(header)) % 32)
    model = directory / "standard.gguf"
    model.write_bytes(header + payload)
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config))
    return model, config_path, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--dspark-built", action="store_true")
    parser.add_argument("--rollback-test", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    directory = args.work.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    try:
        model, config_path, config = fixture(directory, nonzero=args.rollback_test)
        if args.rollback_test:
            result = subprocess.run([str(args.engine.resolve()), "--rollback-test", str(model)],
                                    env=dict(os.environ, STRATA_V41_CONFIG=str(config_path)),
                                    text=True, capture_output=True, timeout=180)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            print(result.stdout, end="")
            return
        env = dict(os.environ, STRATA_V41_CONFIG=str(config_path))
        for key in ("STRATA_DSPARK", "STRATA_SERVE_DSPARK"):
            env.pop(key, None)
        command = [
            str(args.engine.resolve()), "-m", str(model), "--ids", "0,2,3",
            "-n", "2", "--backend", args.backend, "--experts", "cpu",
            "--threads", "2", "--prompt-batch", "0", "--prefill-chunk", "1",
            "--no-enc-skip", "--route-bias", "0", "--route-bias-file", "0", "--quiet",
        ]
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
        if args.backend == "cuda" and result.returncode and "no cuda backend" in result.stderr:
            print("No CUDA device/backend available: skipped")
            raise SystemExit(77)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        if len(result.stdout.split()) != 2:
            raise AssertionError("Expected two generated token IDs: " + result.stdout + result.stderr)
        profile_directory = directory / "route-calibration"
        profile_directory.mkdir()
        profile_model, profile_config, _ = fixture(profile_directory, nonzero=True)
        profile_command = command.copy()
        profile_command[profile_command.index("-m") + 1] = str(profile_model)
        profile_env = dict(env, STRATA_V41_CONFIG=str(profile_config))
        baseline_logits = directory / "profile-baseline.f32"
        observed_logits = directory / "profile-observed.f32"
        baseline = subprocess.run(profile_command + ["--dump-last", str(baseline_logits)],
                                  env=profile_env, text=True, capture_output=True, timeout=60)
        profile_path = directory / "routes.profile"
        profiled = subprocess.run(profile_command + ["--route-profile", str(profile_path),
                                                    "--dump-last", str(observed_logits)], env=profile_env,
                                  text=True, capture_output=True, timeout=60)
        if (baseline.returncode or profiled.returncode or profiled.stdout != baseline.stdout or
                observed_logits.read_bytes() != baseline_logits.read_bytes()):
            raise AssertionError("Route profiling changed tokens: " + profiled.stdout + profiled.stderr)
        values = struct.unpack("<4f", baseline_logits.read_bytes())
        if not all(math.isfinite(v) for v in values) or not any(values):
            raise AssertionError("Route profiling fixture must produce finite nonzero logits")
        rows = [line.split() for line in profile_path.read_text().splitlines() if line.startswith("count ")]
        expected_decode = 2 * (len(profiled.stdout.split()) - 1)
        if (len(rows) != 4 or sum(int(row[3]) for row in rows) != 6 or
                sum(int(row[4]) for row in rows) != expected_decode):
            raise AssertionError("Route profiling must distinguish three prompt tokens and one decode step per layer")
        served = subprocess.run(
            profile_command + ["--serve", "--no-prefix-cache", "--route-profile", str(profile_path)],
            input="GEN 2 0,2,3\nGEN 2 0,2,3\nQUIT\n", env=profile_env,
            text=True, capture_output=True, timeout=60)
        if served.returncode or served.stdout.count("DONE ") != 2:
            raise AssertionError("Profile serve requests failed: " + served.stdout + served.stderr)
        rows = [line.split() for line in profile_path.read_text().splitlines() if line.startswith("count ")]
        generated = [int(line.split()[1]) for line in served.stdout.splitlines() if line.startswith("DONE ")]
        if (sum(int(row[3]) for row in rows) != 12 or
                sum(int(row[4]) for row in rows) != 2 * sum(n - 1 for n in generated)):
            raise AssertionError("Profile must accumulate prompt/decode counts across serve requests")
        with (directory / "profile-live.stderr").open("w") as stderr:
            live = subprocess.Popen(
                profile_command + ["--serve", "--no-prefix-cache", "--route-profile", str(profile_path)],
                env=profile_env, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
            output = queue.Queue()
            def read_live():
                for line in live.stdout:
                    output.put(line)
                output.put(None)
            reader = threading.Thread(target=read_live, daemon=True)
            reader.start()
            try:
                for request in (1, 2):
                    live.stdin.write("GEN 2 0,2,3\n")
                    live.stdin.flush()
                    while True:
                        line = output.get(timeout=60)
                        if line is None:
                            raise AssertionError("Live profile server exited before DONE")
                        if line.startswith("DONE "):
                            break
                    live_rows = [line.split() for line in profile_path.read_text().splitlines()
                                 if line.startswith("count ")]
                    if sum(int(row[3]) for row in live_rows) != request * 6:
                        raise AssertionError("Cumulative profile must be available before DONE, without QUIT")
                live.stdin.write("QUIT\n")
                live.stdin.flush()
                if live.wait(timeout=60):
                    raise AssertionError("Live profile server failed on QUIT")
            finally:
                if live.poll() is None:
                    live.kill()
                    live.wait(timeout=60)
                live.stdin.close()
                reader.join(timeout=60)
                live.stdout.close()
        seed = directory / "selected.seed"
        metadata = "".join(line + "\n" for line in profile_path.read_text().splitlines()[1:]
                           if not line.startswith("count "))
        seed.write_text("DS41_SEED_V1\n" + metadata + "select 0 0\n")
        rejected = subprocess.run(profile_command + ["--cache-seed", str(seed)], env=profile_env,
                                  text=True, capture_output=True, timeout=60)
        if rejected.returncode == 0 or "requires persistent multi-GPU" not in rejected.stderr:
            raise AssertionError("Exact seed must reject nonpersistent single-device execution")
        print("Route profiling: exact generated tokens, separate prefill/decode counts, seed mode guard passed")
        fallback = subprocess.run(
            command, env=dict(env, STRATA_DEVICE_ACTIVATIONS="1"), text=True, capture_output=True, timeout=60,
        )
        if fallback.returncode or fallback.stdout != result.stdout or "device activations:" in fallback.stderr:
            raise AssertionError("CPU experts must keep the host activation path: " + fallback.stdout + fallback.stderr)
        if args.backend == "cuda":
            expert_directory = directory / "gpu-experts"
            expert_directory.mkdir()
            expert_model, expert_config, _ = fixture(expert_directory, gpu_experts=True)
            expert_env = dict(env, STRATA_V41_CONFIG=str(expert_config))
            expert_command = command.copy()
            expert_command[expert_command.index("-m") + 1] = str(expert_model)
            expert_command[expert_command.index("--experts") + 1] = "gpu"
            expert_command += ["--arena-gib", "0", "--pf-b", "0", "--pcie", "0"]
            for deferred in (False, True):
                for slots, margin, capacity in (("auto", "10000", 0), ("auto", "0", 4), ("1", "0", 1)):
                    extra = ["--slots", slots, "--vram-margin", margin]
                    if deferred:
                        extra += ["--prompt-batch", "2", "--prefill-chunk", "2"]
                    result = subprocess.run(
                        expert_command + extra, env=expert_env, text=True, capture_output=True, timeout=60,
                    )
                    initial_capacity = 0 if deferred else capacity
                    if (result.returncode or len(result.stdout.split()) != 2 or
                            f"{initial_capacity} cache slots allocated" not in result.stderr):
                        raise AssertionError("GPU expert cache allocation failed: " + result.stdout + result.stderr)
                    if deferred and f"{capacity} experts in VRAM (" not in result.stderr:
                        raise AssertionError("Deferred GPU expert cache seed failed: " + result.stderr)
            print("GPU expert launcher: auto, explicit, zero budget and deferred prompt allocation passed")
            activation_directory = directory / "activations"
            activation_directory.mkdir()
            activation_model, activation_config, _ = fixture(activation_directory, gpu_experts=True, nonzero=True)
            activation_env = dict(expert_env, STRATA_V41_CONFIG=str(activation_config), STRATA_FILE_GPU="0")
            for key in ("STRATA_DEVICE_ACTIVATIONS", "STRATA_X_HOST", "STRATA_DEC_HOSTSUM"):
                activation_env.pop(key, None)
            activation_command = expert_command.copy()
            activation_command[activation_command.index("-m") + 1] = str(activation_model)
            activation_command[activation_command.index("-n") + 1] = "4"
            forced_path = activation_directory / "forced.txt"
            forced_path.write_text("0\n2\n3\n0\n")
            activation_env["STRATA_FORCE_IDS"] = str(forced_path)
            for slots in ("4", "1", "0"):
                baseline = None
                for setting in (None, "0", "1"):
                    logit_path = activation_directory / f"logits-{slots}-{setting}.f32"
                    decode_path = activation_directory / f"decode-{slots}-{setting}.f32"
                    run_env = dict(activation_env)
                    run_env["STRATA_DUMP_DECODE_LOGITS"] = str(decode_path)
                    if setting is not None:
                        run_env["STRATA_DEVICE_ACTIVATIONS"] = setting
                    result = subprocess.run(
                        activation_command + ["--slots", slots, "--vram-margin", "10000" if slots == "0" else "0",
                                              "--dump-logits", str(logit_path)],
                        env=run_env, text=True, capture_output=True, timeout=60,
                    )
                    if result.returncode:
                        raise AssertionError("Activation generation failed: " + result.stdout + result.stderr)
                    logits = logit_path.read_bytes()
                    decode = decode_path.read_bytes()
                    if len(decode) != 4 * 4 * 4:
                        raise AssertionError("Expected four decode vocabulary rows")
                    values = struct.unpack("<" + "f" * (len(decode) // 4), decode)
                    if not values or not all(math.isfinite(v) for v in values) or not any(values):
                        raise AssertionError("Activation fixture must have finite, nonzero logits")
                    # The only change is where identical F32 bytes are quantized: require exact logits and tokens.
                    observed = (result.stdout, logits, decode)
                    if baseline is None:
                        baseline = observed
                    elif observed != baseline:
                        raise AssertionError(f"Activation setting={setting}, slots={slots} changed logits/tokens")
                    if setting == "1":
                        transfers = re.search(
                            r"device activations: dense D2H skipped (\d+) bytes .*decode tier "
                            r"D2H (\d+) / H2D (\d+) bytes, direct (\d+) calls", result.stderr,
                        )
                        if not transfers or int(transfers[1]) == 0:
                            raise AssertionError("Opt-in did not skip the dense activation readback: " + result.stderr)
                        _, d2h, h2d, direct = map(int, transfers.groups())
                        if slots == "4" and (d2h or h2d or direct == 0):
                            raise AssertionError("All-resident decode still staged activations: " + result.stderr)
                        if slots == "0" and (d2h == 0 or h2d != d2h or direct):
                            raise AssertionError("No-cache CPU misses lost their activation fallback: " + result.stderr)
                        print(f"Device activation slots={slots}: dense D2H skipped={transfers[1]} bytes; "
                              f"decode tier D2H={d2h} H2D={h2d} bytes, direct={direct}")
                    elif "device activations:" in result.stderr:
                        raise AssertionError("Unset/zero activation opt-in changed the default path")
            print("Device activations: unset/zero/on exact nonzero logits and tokens; all-GPU, mixed and no-cache passed")
            for flag in ("--chunk-batch", "--no-chunk-batch"):
                result = subprocess.run(
                    expert_command + ["--slots", "1", "--prompt-batch", "2", "--prefill-chunk", "2", flag],
                    env=expert_env, text=True, capture_output=True, timeout=60,
                )
                if result.returncode or len(result.stdout.split()) != 2:
                    raise AssertionError("Native batch flag failed on fallback layout: " +
                                         result.stdout + result.stderr)
            print("Native batch CLI: enabled/disabled options accepted on unsupported-width fallback fixture")
            batch_directory = directory / "batch-experts"
            batch_directory.mkdir()
            batch_model, batch_config, _ = fixture(batch_directory, gpu_experts=True, batch_experts=True)
            batch_env = dict(expert_env, STRATA_V41_CONFIG=str(batch_config), MIMO_CHUNK_PROF="1")
            batch_command = expert_command.copy()
            batch_command[batch_command.index("-m") + 1] = str(batch_model)
            outputs = []
            for flag in ("--no-chunk-batch", "--chunk-batch"):
                result = subprocess.run(
                    batch_command + ["--slots", "1", "--prompt-batch", "2", "--prefill-chunk", "2", flag],
                    env=batch_env, text=True, capture_output=True, timeout=120,
                )
                if result.returncode or len(result.stdout.split()) != 2:
                    raise AssertionError("Full-width native batch CLI failed: " + result.stdout + result.stderr)
                if ("chunk batch:" in result.stderr) != (flag == "--chunk-batch"):
                    raise AssertionError("Full-width batch dispatch was not selected as requested: " + result.stderr)
                outputs.append(result.stdout.split())
            if outputs[0] != outputs[1]:
                raise AssertionError("Full-width batch CLI changed generated token IDs")
            print("Native batch CLI: full-width 5120/2304 Q2_K/Q3_K dispatch exercised; enabled/disabled IDs match")
        for extra, message in (
            (["--chunk-mmq"], "--chunk-mmq is unavailable"),
            (["--chunk-batch"], "--chunk-batch requires --experts gpu"),
            (["--expert-gpus", "0,1"], "--expert-gpus requires CUDA GPU experts"),
            (["--experts", "invalid"], "--experts must be cpu|gpu"),
        ):
            result = subprocess.run(command + extra, env=env, text=True, capture_output=True, timeout=60)
            if result.returncode != 2 or message not in result.stderr:
                raise AssertionError("Unsupported option was not rejected: " + result.stderr)
        if args.dspark_built:
            for settings, message in (
                ({"STRATA_SERVE_DSPARK": "unused.gguf"}, "use STRATA_DSPARK"),
                ({"STRATA_DSPARK": ""}, "sidecar path must not be empty"),
                *(({"STRATA_DSPARK": "unused.gguf", "STRATA_DSPARK_DEPTH": depth}, "integer in 1..5")
                  for depth in ("0", "6", "-1", "2junk", "")),
            ):
                result = subprocess.run(command, env=dict(env, **settings), text=True,
                                        capture_output=True, timeout=60)
                if result.returncode != 2 or message not in result.stderr:
                    raise AssertionError("Invalid DSpark option was not rejected: " + result.stderr)
        else:
            for key in ("STRATA_DSPARK", "STRATA_SERVE_DSPARK"):
                result = subprocess.run(
                    command, env=dict(env, **{key: "not-built.gguf"}),
                    text=True, capture_output=True, timeout=60,
                )
                if result.returncode != 2 or "built without DSpark" not in result.stderr:
                    raise AssertionError("Disabled DSpark was not rejected: " + result.stderr)
        config["hidden_size"] = 256
        config_path.write_text(json.dumps(config))
        result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60)
        if result.returncode == 0 or "HF config disagrees with GGUF: hidden_size" not in result.stderr:
            raise AssertionError("Mismatched model/config was not rejected: " + result.stderr)
        print("standard GGUF: embedded quantized Engram, shared KV/index, generation, config and CLI rejection passed")
    finally:
        shutil.rmtree(directory)


if __name__ == "__main__":
    main()
