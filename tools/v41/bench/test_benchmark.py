"""CPU-only protocol tests; artifacts stay inside this worktree."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid

spec = importlib.util.spec_from_file_location("benchmark", Path(__file__).with_name("benchmark.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


class BenchmarkTests(unittest.TestCase):
    def test_done_metrics_and_incomplete_generation(self):
        line = "DONE 32 64 2000 1000 length 0 0 0 20 100 0 80 0 64"
        metrics = B.parse_done(line, 32, 32, 64)
        self.assertEqual(metrics["prefill_tokens_per_second"], 32)
        self.assertEqual(metrics["decode_tokens_per_second"], 31)
        self.assertEqual(metrics["expert_hit_fraction"], .2)
        for bad in (line.replace("32 64", "31 64"), line.replace("length", "error"),
                    line.replace("0 20 100", "1 20 100"),
                    line.replace("2000", "nan"), line.replace("1000", "inf"),
                    line.replace("20 100", "101 100")):
            with self.assertRaises(RuntimeError):
                B.parse_done(bad, 32, 32, 64)

    def test_io_deltas_and_unavailable_counters(self):
        self.assertEqual(B.io_delta({"read_bytes": 10}, {"read_bytes": 35}), {"read_bytes": 25})
        self.assertIsNone(B.io_delta(None, {}))
        self.assertIsNone(B.process_io(999999999))

    def test_chunk_batch_is_explicit(self):
        args = SimpleNamespace(engine="engine", model="model", slots="auto",
                               vram_margin=2, arena_gib=64, pcie=1, pf_b=0,
                               prompt_batch=64, chunk=32, threads=8, ctx=4096,
                               seed=1, chunk_batch=False, expert_gpus="auto")
        self.assertNotIn("--chunk-batch", B.command(args))
        args.chunk_batch = True
        self.assertIn("--chunk-batch", B.command(args))

    def test_mock_process_success_error_and_timeout_cleanup(self):
        root = Path.cwd() / ("build-bench-test-" + uuid.uuid4().hex)
        root.mkdir()
        try:
            ids = root / "ids.json"
            ids.write_text("[1, 2, 3]")
            for mode in ("success", "error", "timeout", "diagnostic", "late-diagnostic",
                         "memory", "occupied", "occupied-multi"):
                fake = root / ("engine-" + mode)
                fake.write_text(
                    "#!/usr/bin/env python3\nimport sys,time\n"
                    "print('READY 4096 stop',flush=True)\n"
                    "for line in sys.stdin:\n"
                    + (" if line.startswith('QUIT'):\n"
                       "  print('serve: decode failed: late',file=sys.stderr,flush=True)\n"
                       "  break\n" if mode == "late-diagnostic" else
                       " if line.startswith('QUIT'): break\n")
                    + (" print('ERR test failure',flush=True)\n" if mode == "error" else
                       " time.sleep(30)\n" if mode == "timeout" else
                       " print('serve: decode failed: test',file=sys.stderr,flush=True)\n time.sleep(30)\n"
                       if mode == "diagnostic" else
                       " print('T 7\\nT 8\\nDONE 2 3 1000 500 length 0 0 0 1 4 0 3 0 3',flush=True)\n"))
                fake.chmod(0o755)
                out = root / mode
                argv = ["benchmark", "--engine", str(fake), "--engine-revision", "test",
                        "--model", str(ids), "--config", str(ids), "--ids-file", str(ids), "--output", str(out),
                        "--output-tokens", "2", "--repeats", "2", "--timeout", "1.5",
                        "--step-max", "0"]
                if mode == "success":
                    argv += ["--device-activations", "--tier-graph", "--dma-overlap",
                             "--disable-cuda-graphs", "--chunk-batch", "--fused-gu",
                             "--down-mode", "4"]
                if mode == "occupied-multi":
                    argv += ["--gpus", "0,3"]
                def capture(cmd):
                    if cmd[0] == "nvidia-smi":
                        return ("index, uuid\n0, fake-gpu\n3, other-gpu" if "--query-gpu=" in " ".join(cmd)
                                else "fake-gpu, 42, other, 100" if mode == "occupied" else
                                "other-gpu, 42, other, 100" if mode == "occupied-multi" else "")
                    return "test"
                memory = iter([100, 100, 39]) if mode == "memory" else None
                with mock.patch.object(sys, "argv", argv), mock.patch.object(B, "capture", capture), \
                        mock.patch.dict(B.os.environ, {"STRATA_DEVICE_ACTIVATIONS": "stale",
                                                       "STRATA_EXP_MODE_T11": "stale",
                                                       "STRATA_DS41_BATCH_FUSED_GU": "stale",
                                                       "STRATA_TIER_GRAPH": "stale",
                                                       "STRATA_DMA_OVERLAP": "stale",
                                                       "GGML_CUDA_DISABLE_GRAPHS": "stale"}), \
                        mock.patch.object(B, "available_gib",
                                          side_effect=(lambda: next(memory, 39)) if memory else lambda: 100), \
                        contextlib.redirect_stdout(io.StringIO()):
                    code = B.main()
                result = json.loads((out / "result.json").read_text())
                self.assertEqual(code, 0 if mode == "success" else 1)
                if mode == "success":
                    self.assertEqual(len(result["measurements"]), 2)
                    self.assertEqual(result["measurements"][1]["cache_state"], "process-warm")
                    self.assertEqual(result["environment"]["STRATA_SERVE_STEP_MAX"], "0")
                    self.assertEqual(result["environment"]["CUDA_VISIBLE_DEVICES"], "0")
                    self.assertIn("--chunk-batch", result["command"])
                    self.assertEqual(result["environment"]["STRATA_EXP_MODE_T11"], "4")
                else:
                    self.assertIn("error", result)
                    if mode.startswith("occupied"):
                        self.assertNotIn("pid", result)
                    else:
                        self.assertIsNotNone(result["exit_code"])
                        with self.assertRaises(ProcessLookupError):
                            B.os.kill(result["pid"], 0)
                if not mode.startswith("occupied"):
                    if mode != "success":
                        self.assertIsNone(result["environment"]["STRATA_EXP_MODE_T11"])
                    for name in ("STRATA_DS41_BATCH_FUSED_GU", "STRATA_DEVICE_ACTIVATIONS", "STRATA_TIER_GRAPH",
                                 "STRATA_DMA_OVERLAP", "GGML_CUDA_DISABLE_GRAPHS"):
                        self.assertEqual(result["environment"][name],
                                         "1" if mode == "success" else None)
        finally:
            shutil.rmtree(root)

    def test_fused_gate_up_requires_batch_dispatch(self):
        argv = ["benchmark", "--engine", "engine", "--engine-revision", "test",
                "--model", "model", "--config", "config", "--ids-file", "ids",
                "--output", "unused", "--fused-gu"]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                B.main()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
