# Bounded local measurements, 2026-10-18

These are five serial process runs, seven complete requests, not a tuning sweep
or a quality evaluation. No defaults changed. No weights were downloaded.

## Hardware and inputs

- Physical GPU0: Quadro RTX 8000, 49,152 MiB, compute capability 7.5,
  NVIDIA driver 580.178.04. Other GPUs were not used by this worker.
- Xeon W-2295, 18 cores / 36 logical CPUs, 251 GiB system RAM. Each engine used
  eight CPU threads. Other roadmap workers shared the host; CPU/PCIe traffic
  was not isolated system-wide.
- Read-only engine: `build-ds41/cuda/ds41_generate` in the main checkout,
  built at `6d55d2034c1154af14dc3b36c550751ba5de2f55`.
  SHA256: `1164d1f4bb7182f9a056bc746123e12416882fbd363e61ef3f3c4b3e7c6869b2`.
  All five executable hashes match.
- Local ten-shard `DeepSeek-V4.1-Flash-1017-Q2_K` GGUF, original local config
  and tokenizer. Standard quantized Engram remained enabled.
  Tokenizer SHA256:
  `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`.
  Local `tokenizers` 0.22.1 encoded the repeated virtual-memory explanation
  prompt without added special tokens, truncated to the first 64 or 128 IDs.
- All requests: context 4096, 32 output tokens, temperature 0, seed 1,
  route/file biases 0, no pruning/skipping, no prefix reuse, no DSpark,
  VRAM LRU on, prediction prefetch 0, VRAM margin 2 GiB.
- Every process had a 600-second deadline and MemAvailable floor 40 GiB.
  Arena sizes never exceeded 64 GiB. No retries or fallbacks were used.

## Configurations

| Artifact name | Input | Slots | Arena GiB | Prompt batch / chunk | Step cutoff | PCIe fraction |
| --- | ---: | --- | ---: | --- | --- | ---: |
| baseline64 | 64 | 512 | 16 | 32 / 16 | engine default 768 | 1 |
| auto64 | 64 | auto | 64 | 32 / 16 | engine default 768 | 1 |
| auto128 | 128 | auto | 64 | 32 / 16 | explicitly 0 | 1 |
| geometry128 | 128 | auto | 64 | 64 / 32 | explicitly 0 | 1 |
| cpu64 | 64 | auto | 64 | 32 / 16 | engine default 768 | 0 |

The 64-ID runs actually used short-turn passes. Both 128-ID runs explicitly
forced layer-major batch prefill; changing batch/chunk at 64 IDs alone would
not measure that path. The two residency runs jointly change slots and arena,
not an isolated slot-only or arena-only comparison. `cpu64` changes only the
PCIe fraction relative to the cold `auto64` request.

## Observations

TTFT includes cache opening/seeding where applicable; prefill timing does not.
Decode throughput uses 31 decode steps, not all 32 emitted tokens.

| Run / process state | TTFT s | Prefill s | Prefill tok/s | Decode s | Decode tok/s | Decode hit % | Decode file routes | Request storage read GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline64 / cold | 98.699 | 98.031 | 0.653 | 48.813 | 0.635 | 4.247 | 6794 | 248.612 |
| baseline64 / warm | 97.528 | 97.520 | 0.656 | 48.662 | 0.637 | 4.368 | 6779 | 247.952 |
| auto64 / cold | 77.857 | 72.012 | 0.889 | 36.009 | 0.861 | 24.691 | 4843 | 177.272 |
| auto64 / warm | 71.797 | 71.790 | 0.891 | 35.937 | 0.863 | 24.785 | 4837 | 177.053 |
| auto128 / cold | 46.626 | 40.671 | 3.147 | 36.091 | 0.859 | 23.831 | 4843 | 133.804 |
| geometry128 / cold | 33.443 | 27.518 | 4.652 | 36.234 | 0.856 | 23.871 | 4845 | 108.909 |
| cpu64 / cold | 75.697 | 69.851 | 0.916 | 35.237 | 0.880 | 22.634 | 4843 | 177.272 |

All requests finished with exactly 32 tokens and zero reused prompt tokens.
All five 64-ID request outputs are token-identical. The two 128-ID outputs
are also token-identical. This is not a general numerical-equivalence test.

The CPU-axis decode reported 1,684 VRAM hits, 913 CPU expert routes and 4,843
PCIe routes, all 4,843 also counted as file routes. Setting PCIe fraction to
zero does **not** remove file-tier GPU transfers. The corresponding cold
auto64 decode reported 1,837 hits, zero CPU routes and 5,603 PCIe routes.

Process startup took 9.861 / 35.585 / 35.876 / 35.816 / 35.751 seconds in
table order. Minimum MemAvailable was respectively 226.085 / 177.224 /
174.855 / 173.566 / 178.195 GiB; no memory-floor stop occurred. The four
64-GiB-arena runs recorded peak process RSS of 67.619 / 69.659 / 69.662 /
67.650 GiB. Initial baseline instrumentation did not record peak RSS or a
harness-source hash; its protocol timing, I/O and memory-floor measurements
were already present. Later runs share the final harness hash in their JSON.

All cache labels refer to the **process**, never disk-cold storage. OS page
cache was uncontrolled and not flushed. Linux task `read_bytes` includes
storage reads during each complete request, including background arena work;
it is not expert payload or PCIe bytes. Warm repeats redo full prefill.
Single cold samples, only two warm repeats, short prompts and a shared host
do not establish an optimal configuration or a reliable speedup.

## Reproduction and raw artifacts

Persistent artifact root in this session:

```text
/home/algore/.copilot/session-state/644eb78d-44d2-4dd8-b49c-408d4c0652f7/files/worktrees/bench/build-bench-real
```

Each named subdirectory contains `result.json`, `stdout.log`, `stderr.log`.
JSON records exact argv, actual routing diagnostics, engine/config hashes,
shard sizes/mtimes, hardware, input/output IDs and individual request metrics.
The root also contains `prompt-{64,128}.json`, `prompt-source.txt`,
`input-provenance.json`, aggregate `summary.json`, console JSON files and
`run_remaining.py` (the bounded serial invocation script after the baseline).
Artifacts are local session files, not committed checkpoint data.

Validation:

- `python -m unittest discover -s tools/v41/bench -p 'test_*.py' -v`:
  passed, three focused tests, including seven child-process scenarios
  (success, protocol error, timeout, immediate/late diagnostics, memory floor,
  occupied GPU refusal), counter parsing and child cleanup.
- `python -m py_compile tools/v41/bench/benchmark.py tools/v41/bench/test_benchmark.py`:
  passed.
- `git diff --cached --check`: passed.
- Five real commands, exactly recorded in each `result.json`: all passed.
- `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader`
  and a `ps` model-process check after the matrix found no remaining model
  process; GPU0 and the model-RAM lease were released.

Larger arenas (96 GiB), long contexts, thread-count sweeps, prefetch, repeated
geometry/CPU trials, application-quality tests and other backends were not
tested. No core/backend files changed, so no backend rebuild was required.

## Integrated roadmap checkpoint comparison

Five further serial runs used the integrated `perf/ds41-roadmap` engine at
`b190925723e16c216da275135252c929f9ad7ca7`. The executable SHA256 was
`8a415bda075065eee33d30f1883283b5be125c30b754fc803e8289cb02b10944`
in every run. This private Release build used system GCC 13.3, CUDA 13.4,
SM75, and `STRATA_DS41_CUDA_GRAPHS=ON`, `STRATA_DS41_DSPARK=ON` and
`STRATA_DS41_MGPU=ON`. No draft sidecar or multi-GPU generation was used.
All runtime optimizations remain opt-in; enabling their build does not change
the repository defaults.

Hardware, checkpoint, tokenizer, input IDs and generation settings were the
same as `auto128` above: 128 input IDs, 32 output IDs, auto slots, 64 GiB arena,
batch 32 / chunk 16, short-turn cutoff 0, context 4096 and eight CPU threads.
Each run used the same binary and one process-cold request, without prefix
reuse. OS page-cache warmth was uncontrolled. These were ordered single
samples, not randomized or repeated trials.

| Run | Explicit harness switches | Prefill tok/s | Decode tok/s | TTFT s |
| --- | --- | ---: | ---: | ---: |
| eager | `--disable-cuda-graphs` | 3.110 | 0.853 | 47.067 |
| capture | none | 3.108 | 0.863 | 47.100 |
| batch | `--disable-cuda-graphs --chunk-batch` | 3.117 | 0.850 | 46.983 |
| device | `--disable-cuda-graphs --device-activations` | 3.114 | 0.852 | 47.028 |
| combined | `--chunk-batch --device-activations --tier-graph --dma-overlap` | 3.123 | 0.863 | 46.907 |

Every request completed with exactly the same 32 output IDs as the previous
`6d55d203` baseline. All five runs recorded 1,773 decode VRAM hits out of
7,440 expert lookups, 4,843 file routes and 133.804 GiB of request storage reads.
Minimum MemAvailable across the matrix was 178.045 GiB; peak process RSS
ranged from 69.654 to 69.715 GiB. All children exited with status zero, and
the final GPU/process checks found no remaining model process.

These runs establish a bounded generation regression check, not general logit
equivalence or application quality. They do **not** establish a meaningful
end-to-end acceleration: the observed decode range was only 0.850-0.863 tok/s
and prefill 3.108-3.123 tok/s. The large file-tier traffic makes residency and
storage worth investigating next; the synthetic kernel improvements alone
did not translate into a demonstrated model-throughput gain here. No defaults
were changed.

Combined validation at the same engine revision:

- `cmake --build build-roadmap-system --parallel 2` and
  `ctest --test-dir build-roadmap-system --output-on-failure`: passed,
  10/10 CPU tests.
- `cmake --build build-roadmap-cuda --parallel 2`: passed. The first link
  selected Conda's CUDA runtime with CUDA 13.4 headers; correcting only the
  private cache's CUDA library paths and linker flags resolved it.
- `CUDA_VISIBLE_DEVICES=2 ctest --test-dir build-roadmap-cuda -E 'mgpu_cuda' --output-on-failure`:
  passed, 19/19 tests, including full-width batch CLI dispatch and nonzero
  device-activation logit comparisons.
- `CUDA_VISIBLE_DEVICES=2,3 timeout 120 build-roadmap-cuda/test_ds41_mgpu_cuda --devices 0,1`:
  passed, synthetic uniform Q2_K/Q3_K resident-layer prototype only.
- `python -m unittest discover -s tools/v41/bench -p 'test_*.py' -v`:
  passed, 4/4 benchmark harness tests.
- `python -m unittest tools.v41.serve.test_serve_ds41 serve.test_frontend serve.test_detok -q`:
  passed, 44 ran with 5 optional-tokenizer skips.

Raw evidence is under the integration worktree's `build-roadmap-real/`, with
`summary.json`, per-run console JSON and each named run's `result.json`,
`stdout.log` and `stderr.log`. The session artifact
`files/validate_integrated_baseline.py` records the exact serial invocation
and fails on any output-ID mismatch. These local artifacts are not committed.
