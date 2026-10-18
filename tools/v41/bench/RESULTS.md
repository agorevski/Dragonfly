# Bounded local measurements, 2026-10-18

The initial single-GPU measurements below are five serial process runs, seven
complete requests, not a tuning sweep or a quality evaluation. They did not
change defaults. The later supported four-GPU section records the new DS41
CUDA default and its same-build comparison. No weights were downloaded.

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

## Supported four-GPU runtime and profile

The production layer-sharded runner at `54b6dd31fcda7367686f475c66deed57a9ed0547`
defaults to `--expert-gpus auto`. This selects all CUDA-visible cards for DS41;
one visible card retains the original tier. Core Strata/Qwen defaults and its
installer are unchanged. This is real checkpoint generation, not the earlier
synthetic multi-GPU prototype.

### Same-build comparison

Both processes used the same executable, SHA256
`1bcc184fdf50963f69362c9915ad11210396cb90150f97809f54f30c877e60e1`,
and the same 128 input IDs and 32 output IDs as the original `auto128` reference.
The four-card machine has four Quadro RTX 8000 48 GiB cards, PCIe Gen3 x16,
NVLink pairs 0/1 and 2/3, and cross-pair NODE connectivity. CPU, RAM, checkpoint,
config and tokenizer are unchanged. The private Release build uses GCC 13.3,
CUDA 13.4 and SM75. All runs were serialized; no other model occupied the GPUs.

Common settings: auto slots, 64 GiB global arena cap, 2 GiB VRAM margin per
device, batch 32 / chunk 16, step cutoff 0, context 4096, eight threads, PCIe
fraction 1, prediction prefetch 0, temperature 0, seed 1, routing biases 0,
no pruning, no prefix reuse, no DSpark and eager ggml CUDA. Chunk-batch,
device-activation, tier-graph and DMA-overlap opt-ins were not enabled.
Each process had a 600-second deadline and 40 GiB MemAvailable floor.

| GPUs / process state | Startup s | TTFT s | Prefill tok/s | Decode tok/s | Decode VRAM hits | Decode file routes | Request storage read GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPU0 / cold | 35.311 | 47.117 | 3.107 | 0.852 | 1,773 / 7,440 | 4,843 | 133.804 |
| GPU0-3 / cold | 80.222 | 2.254 | 56.816 | 21.917 | 7,193 / 7,440 | 0 | 0 |
| GPU0-3 / warm | same process | 1.953 | 65.567 | 21.846 | 7,193 / 7,440 | 0 | 0 |

The cold-request comparison is **18.3x prefill and 25.7x decode** on this
machine. Decode counts 31 steps for 32 emitted tokens. TTFT excludes process
startup. Warm repeats redo full prefill, without prompt reuse. All three
requests retained exactly the original baseline's 32 generated IDs.
These are ordered short single samples plus one warm repeat, not a tuning
sweep or a general quality/numerical-equivalence result.

The difference is residency, not four-way simultaneous matrix computation:
dense attention, Engram and KV stay on CUDA0, while contiguous expert-layer
ranges execute on their owning cards. This run placed 14,868 of 15,360
experts in VRAM and the remaining 492 in a 5.9 GiB pinned arena, with no file
tier. Resident/pinned counts were 3,529/311 on GPU0, 3,780/60 on GPU1,
3,780/60 on GPU2 and 3,779/61 on GPU3. Decode had 247 pinned PCIe misses,
zero CPU routes and zero file routes. Initialization still reads and uploads
weights: zero request storage reads does not mean zero startup I/O.

Peak sampled process RSS was 69.654 GiB for single GPU and 9.954 GiB for four
GPUs; minimum MemAvailable was 178.014 and 238.459 GiB respectively. This is
process/host instrumentation, not a measurement of total GPU allocation.
Automatic slot counts can vary slightly with free VRAM and profiler overhead.

### Nsight Systems evidence

A separate four-GPU one-shot 128-input / 32-output run was captured with
Nsight Systems 2026.3.2, CUDA and OS runtime tracing, no CPU sampling and no
context-switch sampling. It also produced the same 32 output IDs. Under
tracing, prefill was 54.0 tok/s and decode 17.54 tok/s; use the unprofiled
protocol table above for throughput comparisons, not the instrumented run.

The capture includes startup. Whole-process copy totals are dominated by
initial cache seeding and must not be presented as decode traffic. To exclude
that upload, the SQLite summary selects kernels and transfers starting at the
first `native_gu_kernel` launch (81.326 seconds into the trace). The last
kernel ends 3.998 seconds later. This approximate forward-only window omits
the first layer's earlier dense work; it is not a separately marked decode
range.

| CUDA device | Kernel launches in forward window | Summed GPU kernel time ms |
| --- | ---: | ---: |
| 0 | 221,427 | 1,254.425 |
| 1 | 2,059 | 413.533 |
| 2 | 1,984 | 406.519 |
| 3 | 1,990 | 419.267 |

All four devices executed real kernels. Native Q2_K gate/up took 895.776 ms
over 1,640 launches; native Q3_K down took 695.552 ms over 1,640 launches.
Together they account for about 63.8% of summed kernel time. Other leading
kernels were ggml Q2_K matrix-vector (230.232 ms), Q2_K matrix-matrix
(110.960 ms) and flash attention (36.573 ms).

The same window recorded 8,881 host-to-device copies, 7,861,744,860 bytes and
671.469 ms, plus 5,756 device-to-host copies (29.336 ms) and 22,290 copies
classified by CUDA as device-to-device (25.593 ms). These include staging and
dense work, not just expert payload. Host-side `cudaStreamSynchronize` took
2,314.533 ms over 9,049 calls; kernel launches took 774.775 ms over 227,141
calls. API durations include waiting for GPU work and are not additive to
kernel/copy durations.

This supports further investigation of expert kernels, pinned-tail transfers,
small-launch overhead and synchronization now that runtime file I/O is gone.
It does not establish the gain from any proposed fusion, overlap or graph
change, and does not justify enabling those separate opt-ins by default.

### Serving, validation and reproduction

The isolated HTTP adapter at `6bb676f8` accepts the production multi-GPU
configuration without injecting Qwen pipeline arguments. A real launcher run
on all four cards omitted `--expert-gpus`, `--pcie` and `--pf-b`, exercising
their defaults. On loopback, health reported the model loaded, an OpenAI
chat completion returned `4`, and an Anthropic SSE response returned `hello`
and ended with `message_stop`. Both model and HTTP processes exited afterward.
This is a serial smoke check, not a concurrency or long-context qualification.

CPU CTest passed 11/11 and CUDA CTest passed 20/20. Explicit
`CUDA_VISIBLE_DEVICES=0,1,2,3 build-roadmap-cuda/test_ds41_runner --cuda` and
the two-card physical 2/3 counterpart passed nonzero mixed Q2_K/Q3_K resident
and pinned references, cross-device output ordering and lifecycle checks.
The adapter/frontend/detokenizer unittest command listed above passed again
after the HTTP fix: 44 ran, five optional-tokenizer skips.

Reproduce each protocol case using the input JSON from the original `auto128`
artifacts and the corresponding physical mask (`0` or `0,1,2,3`):

```sh
python tools/v41/bench/benchmark.py \
  --engine build-roadmap-cuda/ds41_generate --engine-revision 54b6dd31 \
  --model /path/to/model-00001-of-00010.gguf --config /path/to/config.json \
  --ids-file /path/to/prompt-128.json --gpus 0,1,2,3 --expert-gpus auto \
  --slots auto --arena-gib 64 --vram-margin 2 --prompt-batch 32 \
  --chunk 16 --step-max 0 --output-tokens 32 --timeout 600 \
  --disable-cuda-graphs --repeats 2 --output /path/to/fresh-output

nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
  --format csv --output /path/to/nsys-stats /path/to/four-nsys.nsys-rep
```

Persistent evidence is in the integration worktree's `build-mgpu-real/`:
`summary.json`, `single/` and `four/` result JSON/logs, `nsys-command.json`
(exact profiler argv/environment), `four-nsys.nsys-rep`, its exported SQLite,
CSV summaries, `nsys-forward-summary.json` and `http/` request/config/log
artifacts. Session scripts `files/profile_mgpu_baseline.py` and
`files/check_mgpu_http.py` record the bounded serial commands and assertions.
Large profiles and checkpoint artifacts are not committed.

Supported here means the isolated Linux NVIDIA CUDA runner and adapter,
including the standard uniform-layer Q2_K gate/up plus Q3_K down checkpoint.
Other architectures, heterogeneous capacity balancing, long contexts,
concurrent serving, real DSpark and HIP/SYCL/Metal are not established by these
four-identical-card results. Insufficient VRAM-plus-pinned coverage fails
explicitly rather than falling back to file-tier generation.
