# DeepSeek V4.1 Flash (experimental, opt-in)

This add-on ports Xolotl's DeepSeek V4.1 inference path for
`deepseek-ai/DeepSeek-V4.1-Flash-1017`. It is separate from Strata's Qwen engine.
The normal installer, model choices, server entry point and engine defaults
remain unchanged. DeepSeek V4 Flash 0731 is a different architecture and is not
supported by this add-on.

## Model files

Use a compatible **GGUF** checkpoint of V4.1 Flash 1017. The original Hugging Face
safetensors files are not a drop-in input to the GGUF engine. Keep all GGUF
shards beside the first shard and supply the matching original `config.json`;
some V4.1 routing parameters are not present in GGUF metadata.

Xolotl documents the ten-shard Q2_K checkpoint at
[`mrfakename/DeepSeek-V4.1-Flash-1017-GGUF`](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-GGUF).
Its first shard is `DeepSeek-V4.1-Flash-1017-Q2_K-00001-of-00010.gguf`.
Obtain the matching `config.json` and tokenizer from the
[original model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
The pre-downloaded Q2_K checkpoint was used for the short generation checks
described below; no model weights were downloaded during the port.

Standard GGUF checkpoints can contain quantized Engram tables. These do not need
the separate FP8 Engram files used by Xolotl's custom GSQ checkpoints. A GSQ
checkpoint needs its matching Engram files and any associated rank data.
Do not mix artifacts from different checkpoints.

No model weights are included or downloaded automatically. Model storage,
system RAM and VRAM requirements depend on the chosen checkpoint and context
length; Strata's Qwen memory estimates do not apply to this model.

## Build separately from Strata

Use CMake 3.24 or newer, a C++20 compiler and, for GPU inference, the NVIDIA
CUDA toolkit. Configure the add-on directly so the normal Strata build does
not need to change:

```sh
cmake -S tools/v41 -B build-ds41 \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_DS41_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-ds41 --parallel 8
ctest --test-dir build-ds41 --output-on-failure
```

`75` is the compute capability of the Quadro RTX 8000 used for this port's
build checks; choose the architecture appropriate to your GPU. A CPU-only
build uses `-DSTRATA_DS41_CUDA=OFF`. The build fetches a pinned llama.cpp
source archive for ggml, not model weights. To use an existing compatible
llama.cpp checkout instead, pass
`-DSTRATA_DS41_GGML_DIR=/path/to/llama.cpp`.
Set `STRATA_DS41_CUDA_BUILD_JOBS=2` when configuring to bound the separate stock
CUDA backend build as well as using `--parallel 2` for the add-on targets.

The root Strata CMake project also exposes `STRATA_ENABLE_DS41`, which is
`OFF` by default. A root-project opt-in build additionally needs
`STRATA_DS41_CUDA=ON` for the DeepSeek CUDA targets; this is separate from
the normal Strata engine's backend settings.

The engine's default `--slots auto` sizes uniform Q2_K/Q3_K expert caches
against the free-VRAM budget, including slot alignment. On the single-GPU path, with batched
prefill, allocation is deferred until the prompt buffers are released.
If the automatic budget has no room, decode uses the RAM/file tiers
without a VRAM expert cache. An explicit `--slots N` remains a manual
capacity choice; it must fit the available VRAM.

## Supported multi-GPU generation and serving

**DS41 CUDA defaults to `--expert-gpus auto`: all CUDA-visible GPUs.** The
launcher uses the same default. One visible card uses the original single-GPU
tier. To keep that path on a multi-GPU machine, pass `--expert-gpus 0`; to
choose physical cards, set `CUDA_VISIBLE_DEVICES` before launch. IDs in
`--expert-gpus 0,1,2,3` refer to that visible list, not physical ordinals.
The separate Strata/Qwen installer and its defaults are unchanged.

This is the production layer-sharded runner adapted from Xolotl, not the
older all-resident prototype below. Dense attention, Engram and KV stay on
CUDA0. Contiguous expert-layer ranges execute on their owning GPUs. Each
owner keeps a persistent VRAM cache and pins only its nonresident experts
in host RAM. Every expert must fit in that union: initialization reports
the counts on each card and rejects an uncovered/file-tier tail.
Small routed outputs are copied back and synchronized before dense uses them.
This is capacity sharing, not concurrent tensor parallelism.

The standard checkpoint's **Q2_K gate/up plus Q3_K down** is supported.
Layer formats must be uniform across the model; per-expert mixed formats
and pruning are rejected. `--arena-gib` caps the **total** pinned tail across
the selected cards, while an explicit `--slots N` caps each card's cache.
`--vram-margin` must be at least 1 GiB and reserves prompt/workspace room
on every selected GPU; start with 2 GiB and a 4096-token context.
More context or larger prompt batches can require a larger margin.
The implementation retains a 40 GiB MemAvailable floor while building arenas.

For multi-GPU, omitted PCIe/prediction settings become `--pcie 1 --pf-b 0`;
incompatible explicit values fail instead of silently selecting a different
execution mode. Caches stay resident across prefill and repeated requests;
adaptive eviction, streaming prefetch and CPU expert execution are not used.
The normal routing-bias defaults are unchanged; use `--route-bias 0
--route-bias-file 0` for exact-routing comparisons. Real DSpark with multi-GPU
is not validated and is explicitly rejected. CPU execution remains available
with `--backend cpu --experts cpu` (or launcher `--backend cpu`).

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/v41/serve/launch.py \
  --engine build-ds41/ds41_generate \
  --model /path/to/model-00001-of-00010.gguf \
  --config /path/to/config.json --tokenizer /path/to/v41-tokenizer \
  --expert-gpus auto \
  --engine-arg=--vram-margin --engine-arg=2 \
  --engine-arg=--arena-gib --engine-arg=64 \
  --engine-arg=--prompt-batch --engine-arg=32 \
  --engine-arg=--prefill-chunk --engine-arg=16
```

Linux x86-64 NVIDIA CUDA is the supported backend for this runner.
Two- and four-device nonzero synthetic tests cover the mixed-format resident
and pinned paths, cross-device outputs, independent consumer streams,
changed inputs and lifecycle. Reserve the cards before running
`CUDA_VISIBLE_DEVICES=0,1,2,3 build-ds41/test_ds41_runner --cuda`;
this explicit multi-card test is not run by ordinary CTest.
Checkpoint timings and Nsight evidence are recorded in the
[benchmark report](bench/RESULTS.md#supported-four-gpu-runtime-and-profile).
On the local four-card system, the same-build 128-input / 32-output Q2_K
comparison measured 56.816 prefill and 21.917 decode tok/s, versus 3.107
and 0.852 on one card, with identical output IDs and zero runtime file routes.
Startup increased from 35.3 to 80.2 seconds; these are short local samples,
not a general throughput guarantee. OpenAI and Anthropic serving also passed
a real four-card smoke check.
Other GPU architectures, heterogeneous
capacity balancing and long-context/concurrent serving are not established
by the local four-Quadro-RTX-8000 measurements.

## Calibrating a smaller persistent expert cache

This opt-in tool counts **routed expert assignments**, not token hit/miss events.
It does not change router scores, routing IDs, or expert arithmetic. For an unbiased
calibration, explicitly disable the existing routing biases and pruning/skipping.
Use representative prompts and enough generated tokens; prefill-only data is not
a decode calibration. Keep a separate, independently collected held-out workload.

Add these arguments to an otherwise identical engine command:

```sh
--route-profile train.routes --route-bias 0 --route-bias-file 0 \
  --skip-miss 0 --skip-file 0 --no-enc-skip --no-prefix-cache
```

With `--serve`, counters accumulate across requests, are flushed **before every
`DONE`**, and are also exported on `QUIT`/EOF. One-shot runs export on exit.
No prompt text, token IDs, routing weights, or activations are recorded. Abrupt
termination can lose the current request or interrupt an export; read the file
after `DONE` or clean shutdown. Counts include routes actually executed, including
speculative verification/rejected rows and retried work, not merely accepted tokens.
Prefix reuse and encoder skipping naturally omit work that was not executed.
The profiler also records the currently resident IDs at export time.

```sh
python tools/v41/expert_residency.py train.routes other-train.routes \
  --phase decode --target .99 --held-out held.routes \
  --output residency.json --seed-output selected.seed
```

`--phase all` is the default; `prefill` and `decode` stay separate in the input.
Selection walks descending calibration frequency with deterministic `(layer,
expert)` ties until the **global** requested assignment coverage is reached.
The JSON lists exact IDs and per-layer coverage; 99% globally does not mean 99%
in every layer or on future prompts. This minimizes the number of experts without
capacity constraints, not the bytes of mixed-format experts. Optional
`--capacity-bytes 0:BYTES,1:BYTES,...` specifies an expert-only budget for every
profiled device; experts that cannot fit are skipped and an unattainable target
fails visibly. Dense weights, KV, scratch, and allocator overhead need separate
reserves. The runtime independently checks its real free VRAM.

The held-out input never affects selection and must not name a calibration input
(resolved path aliases are rejected). Its coverage is reported separately;
**exit code 3** means it missed the requested target (the report/selection remain
available for inspection). Invalid, zero-observation, incompatible, and overflowing
inputs return code 2 with an error on stderr. Compatible calibration profiles can
be merged; residency comparisons use the last profile's export snapshot.

Per-card estimates use each physical native expert blob rounded up to **256
bytes**, not a uniform average or dequantized tensor size. Reports distinguish
current resident bytes, selected bytes, and the all-resident upper bound; pinned
tail estimates use unpadded host blob bytes. **Dense/KV is on CUDA-visible device
0 in this runner. Only device 0's `current_minus_selected_physical_bytes`
directly releases memory for its KV budget.** Savings on other cards are not
additional local KV space. A negative difference means the recommendation needs
more expert VRAM than the exported baseline, not a saving.

Use the selection in the production persistent layer-sharded runner:

```sh
# Same model, visible device list/order, and geometry as the profile:
build-ds41/ds41_generate ... --expert-gpus 0,1,2,3 \
  --cache-seed selected.seed --arena-gib 100 --vram-margin 2 \
  --route-bias 0 --route-bias-file 0
```

`--cache-seed` is restricted to persistent multi-GPU mode and is mutually
exclusive with `--rank`. It reuses the existing ranked seeder: selected IDs come
first, every remaining owned expert follows into the pinned tail. Each card gets
exactly its selected slot quota (including zero); `--slots` remains an optional
per-card upper bound. Initialization rejects insufficient VRAM or aggregate
`--arena-gib`/arena-cap coverage. After allocation it verifies every selected ID
is actually resident, every unselected ID is pinned, and no file-tier expert
remains. Allocation shortfalls fail rather than silently claiming the target.
Caches and IDs stay fixed across prefill and requests. No automatic adjustment
of context length or KV geometry is made.

The aggregate ASCII format is `DS41_ROUTES_V1`, then `model HEADER_FNV1A64
FIRST_FILE_BYTES`, `geometry LAYERS EXPERTS TOP_K HIDDEN FF`, one
`expert L E DEVICE BLOB_BYTES ALIGNED_BYTES GATE_TYPE UP_TYPE DOWN_TYPE` per
routed expert, and `count L E PREFILL DECODE RESIDENT_0_OR_1` for every geometry
cell. `DS41_SEED_V1` repeats the model/geometry/expert metadata verbatim and adds
`select L E` lines. Geometry, quantization layouts, and device ownership must
match. Model identity is a first-64-KiB FNV-1a fingerprint plus the first file's
size: an accidental-mismatch check, **not** a cryptographic verification of every
model shard or external config. Keep those artifacts matched independently.

CPU-only focused checks:

```sh
python tools/v41/test_expert_residency.py
ctest --test-dir build-ds41 -R 'ds41_(runner_selection|expert_residency|standard)$' --output-on-failure
```

The exact-selection CUDA fixture is part of the explicitly leased
`test_ds41_runner --cuda` run. CPU checks do not establish a real-workload 99%
hit rate, long-context fit, or unchanged GPU performance.

## Experimental multi-GPU expert capacity prototype

This earlier diagnostic remains separate from the supported runner above.
`STRATA_DS41_MGPU=ON` builds a separate resident-layer expert runner. It is
off by default, is not linked into `ds41_generate`, and does **not** enable
multi-GPU generation or serving. No checkpoint was loaded for its validation.
The CPU `ds41_generate` executable was byte-identical with this option on
and off in the same build directory (SHA-256
`5dcc003bb6dd72817ce6914a82fe0da607610ae54ea40ae0a6f38c9cb65e7ddd`).

```sh
cmake -S tools/v41 -B build-ds41-mgpu \
  -DSTRATA_DS41_GGML_DIR=/path/to/llama.cpp \
  -DSTRATA_DS41_MGPU=ON -DSTRATA_DS41_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-ds41-mgpu \
  --target test_ds41_mgpu test_ds41_mgpu_cuda --parallel 2
ctest --test-dir build-ds41-mgpu -R '^ds41_mgpu_plan$' --output-on-failure
# Only after reserving physical GPUs 2 and 3:
CUDA_VISIBLE_DEVICES=2,3 timeout 120s \
  build-ds41-mgpu/test_ds41_mgpu_cuda --devices 0,1
```

For CPU planning only, use `STRATA_DS41_CUDA=OFF` and build
`test_ds41_mgpu`. The CUDA test is deliberately not registered with CTest:
it requires an explicit device lease. Device IDs are CUDA-visible ordinals.

The C++ interface is `engine/ds41_mgpu.hpp`: `ds41_mgpu_plan` validates and
sizes ownership without CUDA calls or model reads. `Ds41MultiGpuMoe::init`
loads every routed expert of each assigned layer onto its owner, under
explicit per-device caps for weights and workspace, while retaining a free
VRAM reserve. An empty owner map balances contiguous layer counts, not
heterogeneous card capacity; supply an explicit map for unequal caps.
Duplicate device IDs, invalid mappings, unsupported formats and insufficient
capacity fail rather than falling back to RAM or CPU execution.

`run` is blocking single-token expert execution with **host** input/output
pointers. Each owner has a nonblocking stream and bounded pinned staging;
operations restore the caller's current device. Host staging does not enable
peer access or assume NVLink/default-stream ordering. Initialization failure
cleans partially loaded shards on their allocating devices. Allocation
failures return an error; the reused native kernels retain their existing
fatal launch-error checks.

Supported inputs are uniform standard Q2_K or Q3_K gate/up/down, dimensions
256 through 32768 in multiples of 256, 2 through 256 layers, 1 through 1024
experts, and top-k at most 8. Every selected device must own a layer.
There are no mixed formats, cache misses, prompt batches, device-pointer
inputs, concurrent calls or graph support. Dense/attention/KV placement and
the generator/server connection remain unimplemented.

Synthetic tests on physical GPUs 2 and 3 (Quadro RTX 8000, SM75; one NVLink
pair) used four layers, four experts, width/FFN 256 and top-k 2. Combined
explicit device allocations were 1,053,072 bytes for Q2_K and 1,372,560
bytes for Q3_K, excluding CUDA context overhead. Maximum relative L2 error
against the existing CPU expert runner was 0.0230929 and 0.0182977,
respectively (CUDA q8_1 versus CPU q8_K activation rounding; threshold 0.05).
Tests cover layer-owner switching, duplicate expert IDs, repeated outputs,
caller-device restoration, invalid requests, capacity/free-reserve rejection,
and partial initialization cleanup followed by reinitialization.
Physical allocator OOM and fatal CUDA failures were not induced.
CPU tests cover explicit/noncontiguous ownership and invalid mappings.

This validates only that two-device synthetic expert prototype, not
four-device or cross-NODE execution, real-model capacity, throughput,
logits or model quality. Full integration still needs dense/attention/KV
ownership, inter-layer activation transfers, mixed-checkpoint validation,
end-to-end reference checks and a same-day single-device benchmark.

## Optional native expert batch tiles

`ds41_generate --experts gpu --chunk-batch` opts prompt chunks into a
four-activation-column DP4A kernel for standard 5120-wide, 2304-hidden
experts: Q2_K or Q3_K gate/up and Q3_K down. It covers both resident and
streamed experts without the fork MMQ ABI. Other layouts keep the grouped
kernel. The default is **off**; `--no-chunk-batch` explicitly disables it.
Decode, routing and the normal Strata engine are unchanged.

The batch kernel reuses the existing finite SwiGLU clamp and q8_1 activation
quantizer, including FP32 scale selection, integer rounding and FP16 scale
storage. It does not implement LoRA or tail-scale layouts.

`STRATA_DS41_BATCH_FUSED_GU=1` additionally opts the batch kernel into a
fused gate/up, SwiGLU and hidden-q8_1 launch. It requires `--chunk-batch`;
without that flag it has no effect. Unset or `=0` keeps the original
three-launch batch path. The fused path uses two launches: one CTA computes
32 gate/up rows for up to four tokens, exchanges their clamped SwiGLU
values in shared memory, and runs the existing q8_1 quantizer before the
unchanged down kernel. This removes one launch and the global FP32
gate/up writes and reads (36 KiB per routed token), but does not change
scratch allocation, transfers, expert residency or decode.
The same weight decoding, warp reduction order, explicitly rounded
SwiGLU product and q8_1 scale/rounding rules are retained.
The `ds41_batch` and `ds41_batch_fused_gu` CTests explicitly select off/on
and compare all output floats and hidden q8_1 storage, including untouched
entry guards, against grouped execution with nonzero synthetic weights.
Do not infer end-to-end improvement from the removed memory operations;
compare the same prompt and geometry with only this environment variable
changed before enabling it for a workload.

On physical GPU 0 (Quadro RTX 8000, SM75), Linux, GCC 13.3 and CUDA 13.4,
the off/on CTests both passed with seed 7319, full-width nonzero synthetic
weights and the cases below. All output floats and hidden q8_1 bytes were
bitwise equal to grouped execution; mixed gate/up FP64 reference relative
L2 errors were below `8.4e-8`. The two tests ran sequentially, not as an
interleaved off/on experiment. Each test retained the five alternating
grouped/batch pairs of ten launches described below.

| Q2_K/Q2_K/Q3_K routed entries | Unfused batch median [min, max], ms | Fused batch median [min, max], ms |
| --- | --- | --- |
| 3 | 0.153 [0.153, 0.153] | 0.181 [0.180, 0.181] |
| 15 | 0.530 [0.530, 0.531] | 0.567 [0.562, 0.571] |
| 65 | 2.236 [1.888, 2.272] | 1.910 [1.851, 2.210] |
| 257 | 7.502 [7.369, 7.657] | 7.498 [7.287, 7.929] |

These measurements do **not** establish a general speedup: sparse groups
regressed, the 65-entry ranges overlap, and the largest case was flat.
The fused path remains experimental and off by default. No real-model
throughput or quality comparison was run for this fusion.

Synthetic checks on physical GPU 1, a Quadro RTX 8000 (SM75), Linux and CUDA
13.4, used full 5120/2304 matrices, seed 7319 and no model weights. All output
floats and hidden q8_1 bytes matched the grouped path exactly, including
mixed Q2_K/Q3_K gate/up, zero activations, empty groups, nonzero entry offsets,
partial tiles and permuted token/destination indices. Independent dequantized
FP64 accumulation references include activation rounding and clamp 10.
The tier's `run_chunk` also matched default-off and enabled weighted outputs
bitwise for resident and streamed synthetic experts.
The standard CUDA GGUF fixture additionally runs the linked `ds41_generate`
with `--chunk-batch` and `--no-chunk-batch`: full 5120/2304 Q2_K/Q3_K experts
exercise the enabled dispatch, and a 256-wide fixture checks the grouped
fallback. `MIMO_CHUNK_PROF=1` reports `chunk batch:` only when a batch launch
is selected. The full-width fixture has zero weights and checks CLI wiring
and matching generated IDs, not nonzero-model logits or quality.

The microbenchmark measures only gate/up, clamped activation quantization and
down kernels, not weight transfers, routing, dense layers or generation.
It alternates execution order for five pairs of ten launches, after three
warmups per measurement, using CUDA events. One measured Q2_K/Q2_K/Q3_K run:

| Entries (two nonempty expert groups) | Grouped median [min, max], ms | Batch median [min, max], ms |
| --- | --- | --- |
| 3 | 0.162 [0.161, 0.162] | 0.153 [0.153, 0.154] |
| 7 | 0.336 [0.336, 0.336] | 0.261 [0.260, 0.261] |
| 15 | 0.697 [0.697, 0.698] | 0.529 [0.528, 0.529] |
| 65 | 2.487 [2.389, 2.962] | 1.873 [1.848, 2.237] |
| 257 | 9.813 [9.773, 11.792] | 7.337 [7.318, 7.493] |

These are synthetic kernel measurements, **not a whole-model speedup**.
No real-model quality, logits or throughput check was run for this option;
other GPU architectures and backends were not validated. Reproduce with a
private SM75 CUDA build using an existing ggml checkout:

```sh
cmake --build build-ds41 --target test_ds41_batch --parallel 2
CUDA_VISIBLE_DEVICES=1 MIMO_CHUNK_STAGE_MIB=32 \
  ctest --test-dir build-ds41 -R '^ds41_batch$' --output-on-failure -V
CMAKE_BUILD_PARALLEL_LEVEL=2 cmake --build build-ds41 --target ds41_generate --parallel 2
CUDA_VISIBLE_DEVICES=1 MIMO_CHUNK_STAGE_MIB=32 \
  ctest --test-dir build-ds41 -R '^ds41_standard_cuda$' --output-on-failure -V
```

The synthetic test stays below 3 GiB of device allocations even with the
default chunk staging size; the command above reduces that staging to
32 MiB per half. No checkpoint or download is needed.
The full CLI and private stock CUDA backend linked successfully with two
build jobs. The CPU and CUDA standard GGUF tests passed together in 7.28 s;
a repeated CUDA launcher test passed in 7.09 s, with physical GPU 1's total
used memory sampled every 0.2 s peaking at 665 MiB.

## Optional device-resident decode activations

`STRATA_DEVICE_ACTIVATIONS=1` keeps the one-token dense FFN input on the
GPU for routed experts. It is off when unset or `0`, and ignored for CPU
experts or the `STRATA_DEC_HOSTSUM` path. Prompt batching and multi-token
verification keep their existing activation path.

Router IDs and weights still reach the host. The dense backend synchronizes
before the tier reads the activation on its own stream; the tier completes
before the next dense call may reuse that buffer. CPU-served selected
experts retain a synchronized activation readback. Layers served entirely
on the GPU can omit that readback and the tier's activation upload.
`STRATA_X_HOST=1` forces the tier's host staging for diagnostics.

The opt-in prints skipped dense activation bytes and decode-tier input
transfer counters. These count only activation transfers, not expert weights,
router data or routed outputs. `STRATA_DUMP_DECODE_LOGITS=PATH` optionally
writes one raw vocabulary-wide F32 row per ordinary one-shot decode output
for numeric comparisons; it does not record speculative or server outputs.
The synthetic `ds41_device_activations_cuda` and
`ds41_device_activations_host_cuda` tests cover all-resident, mixed CPU-miss,
no-cache CPU and PCIe-served expert inputs. No full-checkpoint throughput
improvement is established by these tests.

On a Quadro RTX 8000 (SM75), Linux/CUDA 13.4, the 256-wide Q3_K
synthetic tier fixture produced bit-identical host/device-input outputs
in all four cases (dequantized F32 reference relative error 0.0092–0.0109).
Over 16 all-resident calls, host input uploaded 16,384 activation bytes;
device input transferred zero activation bytes in either direction.
Mixed and no-cache CPU cases retained 16,384 bytes in each direction.
These are input-transfer counts, not a model-throughput measurement.

## Synthetic graph and transfer diagnostics

`test_ds41_graphs` checks the composite activation rounding against an independent
format-code reference, then replays changed inputs and nonfinite blocks.
`test_ds41_tier_profile` uses four generated, nonzero 256-by-256 Q2_K or Q3_K
experts, two resident slots, and changing routes/activations. It checks against
the CPU tier (relative L2 below 0.05 for different activation quantizers) and
requires byte-identical repeated outputs. Neither executable loads a checkpoint.
Without `--cuda`, both run on CPU without probing a GPU.

Stock ggml CUDA capture is an isolated build opt-in, off by default:

```sh
cmake -S tools/v41 -B build-ds41-graphs \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_DS41_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DSTRATA_DS41_CUDA_GRAPHS=ON -DSTRATA_DS41_CUDA_BUILD_JOBS=2
cmake --build build-ds41-graphs \
  --target test_ds41_graphs test_ds41_tier_profile --parallel 2
ctest --test-dir build-ds41-graphs -R 'ds41_(graphs|tier_profile)$' --output-on-failure
```

Select an available GPU explicitly. Compare eager versus captured composite ops
in the **same binary**; the stock disable variable is presence-based (even `0`
disables capture):

```sh
CUDA_VISIBLE_DEVICES=3 GGML_CUDA_DISABLE_GRAPHS=1 \
  build-ds41-graphs/test_ds41_graphs --cuda --bench
env -u GGML_CUDA_DISABLE_GRAPHS CUDA_VISIBLE_DEVICES=3 \
  build-ds41-graphs/test_ds41_graphs --cuda --bench
```

Compare existing expert-tier switches separately:

```sh
env -u STRATA_TIER_GRAPH -u STRATA_DMA_OVERLAP CUDA_VISIBLE_DEVICES=3 \
  build-ds41-graphs/test_ds41_tier_profile --cuda --bench
env -u STRATA_DMA_OVERLAP CUDA_VISIBLE_DEVICES=3 STRATA_TIER_GRAPH=1 \
  build-ds41-graphs/test_ds41_tier_profile --cuda --bench
env -u STRATA_TIER_GRAPH CUDA_VISIBLE_DEVICES=3 STRATA_DMA_OVERLAP=1 \
  build-ds41-graphs/test_ds41_tier_profile --cuda --bench
CUDA_VISIBLE_DEVICES=3 STRATA_TIER_GRAPH=1 STRATA_DMA_OVERLAP=1 \
  build-ds41-graphs/test_ds41_tier_profile --cuda --bench
```

The two output digests should match across all four tier runs; a capture-refusal
message means the tier graph fell back, not that capture was measured. Each
timing reports 20 warmups and seven batches of 64 calls. `wall_sync` includes
completed calls; `event_submission` is an event-bracketed **submission envelope**,
not isolated kernel time. Explicit completion boundaries order the backend's
private nonblocking streams before the stop event. Tier timings include output
copy/checks, metadata, activation quantization and host reduction. Each measured
CUDA tier run requires 448 resident hits and 448 pinned-arena PCIe misses with no CPU
or file-tier work. The fakequant graph reports its node count and scratch bytes.
These small synthetic results are not model latency, large-transfer bandwidth,
or evidence to change a default.

No fused fakequant kernel is installed. The composite deliberately separates
rounding operations; reassociation or FMA contraction must not remove their
intermediate F32 rounding. Stock ggml's public CUDA interface does not expose its
compute stream or a CUDA custom-op registration hook. A local fused path first
needs a supported launch/stream integration point inside the cached attention
graph, plus bit-exact validation at each intermediate F32 rounding boundary.
Launching on the default stream or removing synchronization is not a substitute.

Measured on 2026-10-18: Quadro RTX 8000 (48 GiB, SM75), driver 580.178.04,
CUDA 13.4, GCC 13.3, Xeon W-2295. Same graph-enabled binary, one ordered
eager/capture sweep; median microseconds per completed composite call
(`wall_sync`, with submission-envelope event medians in parentheses):

| Composite mode | Elements | Eager us | Captured us |
| --- | ---: | ---: | ---: |
| FP8 power-of-two scale | 512 | 81.006 (81.106) | 46.063 (46.145) |
| FP8 power-of-two scale | 32768 | 88.709 (88.800) | 53.725 (53.823) |
| FP4 E4M3 scale | 512 | 107.942 (108.064) | 62.410 (62.500) |
| FP4 E4M3 scale | 32768 | 109.009 (109.115) | 67.181 (67.276) |
| FP4 power-of-two scale | 512 | 79.741 (79.835) | 46.711 (46.816) |
| FP4 power-of-two scale | 32768 | 79.604 (79.712) | 54.043 (54.125) |

All three modes passed the independent finite reference and nonfinite replay
checks in both settings. Stock capture reduces this synthetic launch overhead;
it does not fuse the rounding operations. Scratch allocations were at most
540672 bytes per graph.

The same day's tier sweep gave these median microseconds per call (wall,
with submission-envelope event medians in parentheses):

| Existing switches | Q2_K us | Q3_K us |
| --- | ---: | ---: |
| Neither | 57.841 (58.003) | 59.487 (59.614) |
| Tier graph only | 57.007 (57.118) | 58.928 (59.135) |
| DMA overlap only | 57.909 (58.033) | 57.527 (57.647) |
| Both | 52.517 (52.634) | 54.272 (54.393) |

All four runs matched digests `890bc232eb7bf106` (Q2_K) and
`040deed8d94e04b0` (Q3_K), with the required hit/PCIe counters and no
capture-refusal fallback. CPU-reference relative L2 maxima were 0.032147 and
0.038924, respectively; replay comparisons on the GPU were byte-identical.
These are tiny 64512/84480-byte expert transfers, not model-size DMA.
The four settings ran in the displayed order, not an interleaved randomized
trial. No checkpoint, end-to-end generation, or other GPU/backend was tested;
the measurements do not justify changing any default. CPU CTest passed 3/3
(including the existing cache regression); CUDA-build diagnostic CTest passed
4/4. Only build/test/documentation files changed, not the dense/tier runner
or global ggml/Strata sources.

## Integrated roadmap measurements

The opt-in roadmap paths were tested together on the local ten-shard Q2_K
checkpoint using one Quadro RTX 8000, a Xeon W-2295 and a 64 GiB host arena.
Five 128-input / 32-output trials retained identical generated IDs across
the eager control, CUDA capture, batch kernels, device activations and combined
options, and against the previous engine baseline. Decode measured
0.850-0.863 tok/s and prefill 3.108-3.123 tok/s: these single samples do not
establish an end-to-end speedup. Each request still read 133.804 GiB from
storage. No default changed.

The combined CPU suite passed 10/10 and the CUDA suite 19/19. See
[the benchmark report](bench/RESULTS.md#integrated-roadmap-checkpoint-comparison)
for exact settings, build revision, commands, results and limitations. This
single-GPU checkpoint does not validate real DSpark speculation or multi-GPU
generation; the later supported multi-GPU results are linked above.

## Serve through the isolated adapter

The adapter reuses Strata's HTTP server in a separate process with V4.1
tokenization, chat encoding, reasoning and DSML tool-call handling. Do not
launch a DeepSeek configuration through the normal `serve/server.py`
entry point: it uses Strata's Qwen chat conventions.

The Python dependencies are already listed in the root `requirements.txt`.
Export the original model's local Hugging Face tokenizer, then launch:

```sh
python tools/v41/serve/export_tokenizer.py \
  /path/to/DeepSeek-V4.1-Flash-1017 /path/to/v41-tokenizer

python tools/v41/serve/launch.py \
  --engine build-ds41/ds41_generate \
  --model /path/to/model-00001-of-00010.gguf \
  --config /path/to/DeepSeek-V4.1-Flash-1017/config.json \
  --tokenizer /path/to/v41-tokenizer \
  --write-config strata-ds41.json
```

The source directory for export needs `tokenizer.json` and `config.json`,
not the original safetensors weights. The launcher resolves paths and
supplies `STRATA_V41_CONFIG` to the engine. It does not enable an external
Engram store or a DSpark draft implicitly. Supply `--engram /path/to/engram`
only for a checkpoint that needs external Engram files.

The default address is `127.0.0.1:8150`. OpenAI-compatible clients can use
`http://127.0.0.1:8150/v1`; the Anthropic-compatible endpoint is
`http://127.0.0.1:8150/v1/messages`. For a non-loopback address you must
supply `--api-key` or set `STRATA_API_KEY`. Keys are not written into the
generated configuration.

Use `--configure-only` with `--write-config` to generate a reusable server
configuration without loading weights. Restart it with:

```sh
python tools/v41/serve/serve_ds41.py \
  --engine strata --config strata-ds41.json --port 8150
```

Start with the default 4096-token context. Engine options can be supplied
as repeated `--engine-arg=VALUE` arguments; option names and values are
separate arguments, for example
`--engine-arg=--arena-gib --engine-arg=24`. The CPU-only build requires
`--backend cpu` on the launcher.

The adapter supports text and completed tool calls, including streaming
OpenAI and Anthropic responses. It rejects unsupported forced tool choice,
vision, penalties and logit bias. Multi-GPU uses the same isolated process
and token protocol as single-GPU. Incremental tool
argument deltas, batching and session persistence are not implemented.

## DSpark readiness (optional, off by default)

`STRATA_DS41_DSPARK=ON` builds the experimental draft stack and its CPU synthetic
regression test. It does not enable speculative decoding by itself:

```sh
cmake -S tools/v41 -B build-ds41-cpu \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_DS41_CUDA=OFF \
  -DSTRATA_DS41_DSPARK=ON
cmake --build build-ds41-cpu --target ds41_generate test_ds41_dspark --parallel 2
ctest --test-dir build-ds41-cpu -R 'ds41_(standard|dspark_synthetic|dspark_rollback)$' --output-on-failure
```

The synthetic test uses small invented weights, not a real model. It checks
accepted-prefix accounting at every draft depth, deterministic-draft sampling
(including rejection and filtered-out drafts), rejection boundaries, batched
and single-row draft-window appends through wraparound, reset/reinitialization,
and failures for missing, truncated or mismatched sidecars and missing trunk
metadata. The rollback test compares nonzero synthetic target logits with
sequential decoding at every depth and rejection boundary, at both compressor
parities and past the SWA ring wrap. It includes embedded quantized Engram,
shared ratio-2 compressed KV and shared indexer state; routed expert output is
zero and activation quantization is off. Removing the odd-position re-feed
causes this test to fail. Both tests use CPU only, even in a CUDA build.
These checks do **not** verify target-model rollback with real weights, GPU
kernels, activation quantization, real-weight acceptance or speed.

Real speculative decoding still needs a matching 1017 DSpark sidecar, the trunk's
F16 token embeddings, and the GPU layer-major prompt path. No matching sidecar
was available for this tranche; real-weight speculative decoding remains
unvalidated. Structural tensor checks do not prove two artifacts came from the
same checkpoint. Do not substitute a V4 sidecar or infer compatibility from the
file name or tensor dimensions.

An explicit one-shot request uses `STRATA_DSPARK=/path/to/sidecar.gguf`;
serving uses `STRATA_SERVE_DSPARK=/path/to/sidecar.gguf` with `--serve`.
`STRATA_DSPARK_DEPTH` is 1 through 5. Invalid sidecars and failed draft-window
re-seeding are errors, not successful requests decoded silently without drafts.
The draft API appends accepted positions only, in increasing order; callers must
reset it before rewinding or after a backend execution failure.
Metrics report accepted/proposed drafts and actual verified rows (including
re-feeds). Accepted rows per pass are not emitted tokens per pass: a stop or
output budget can end inside an accepted prefix. Empty-draft passes are excluded
from the full-acceptance percentage. No acceleration claim is made by these tests.

## Keeping upstream updates

The implementation lives under `tools/v41/`. The build integration is opt-in;
leave `STRATA_ENABLE_DS41` off to retain the existing Strata build. The add-on
does not replace the normal installer or make DeepSeek the default model.
Use a separate build directory and the DeepSeek serving entry point instead
of changing an existing Strata model configuration.

Continue merging the Strata upstream branch normally. This isolation reduces
the shared-file merge surface, but cannot guarantee conflict-free updates:
changes to CMake targets, GGUF readers, expert-kernel interfaces or server
interfaces may require adapting the add-on. Rebuild and rerun its focused
tests after such updates. Do not apply Xolotl's entire fork diff or optional
ggml patches to the default Strata build.

## Provenance and scope

The source is the local Xolotl checkout at commit
`6e4bc8e5c09cf732dfa25ccf3ecfc81e672e6d69`, including the working-tree
changes needed for standard 1017 GGUF loading and quantized Engram support.
Those source changes were not committed in Xolotl at the time of the port.
The source checkout is not modified by this add-on.

This is an experimental Linux x86-64 CPU/NVIDIA path. It does not establish Windows,
AMD HIP, Intel SYCL or Apple Metal support, nor does it establish
reference-logit equivalence, model quality or throughput on this machine.
Multi-GPU support is the explicitly documented DS41 layer-sharded path above,
not Strata's Qwen pipeline or its optional speculative/fused paths.

## Original single-GPU port validation on this machine

Checked on Linux, CUDA 13.4, one Quadro RTX 8000 (48 GiB, SM75) selected with
`CUDA_VISIBLE_DEVICES=0`, and 251 GiB system RAM. The machine has four GPUs,
but these checks used only one. The existing ten Q2_K shards occupy about
247 GiB. Only the original model's small config/tokenizer sidecars were fetched.

| Check | Result |
| --- | --- |
| `cmake --build build-ds41/cpu --parallel 8` | Passed |
| `ctest --test-dir build-ds41/cpu --output-on-failure` | 4 passed: quantized Engram, PTQ1, CPU expert pool, synthetic standard GGUF generation |
| `cmake --build build-ds41/cuda --parallel 8` | Passed |
| `ctest --test-dir build-ds41/cuda --output-on-failure` | 7 passed, including CUDA standard GGUF generation, transcode and PTQ1 CUDA reference comparisons |
| `python -m unittest tools.v41.serve.test_serve_ds41 serve.test_frontend serve.test_detok -q` | 44 ran, 5 skipped because the optional Qwen pack tokenizer was absent |
| `python tools/test_setup_choices.py` | 47 passed |
| `python tools/test_setup_amd.py` | 40 passed |
| `python -m pytest serve/test_q36_config.py -q` | 4 passed |

The real checkpoint passed `test_ds41_geometry`: all 40 layers' tensors,
Q2_K gate/up and Q3_K down expert formats, shared KV/index sources and
Engram metadata were accepted. The local adapter's health endpoint reported
the loaded model with a 4096-token context.

Two real HTTP requests passed at temperature 0 with thinking disabled:

- OpenAI `/v1/chat/completions`: "What is 2 + 2? Answer briefly." returned
  `4`, 15 prompt tokens and 2 generated tokens, with `finish_reason: stop`.
- Anthropic `/v1/messages` streaming: "Reply with only the word hello."
  returned `hello` and completed with `end_turn` and `message_stop`.

The bounded smoke configuration used `--arena-gib 16 --slots 512 --threads 8
--prompt-batch 32 --prefill-chunk 16 --route-bias 0 --route-bias-file 0
--pcie 1 --pf-b 0`. This disables Xolotl's default routing biases; other
add-on engine settings can trade routing accuracy for speed and should be
chosen explicitly. Engine flags go through repeated `--engine-arg=VALUE`
arguments on the launcher. The normal Strata engine is not affected.

These are short loading/generation and API compatibility checks, not model
quality, long-context, throughput or reference-logit benchmarks. Real-model
tool calling, DSpark, fused ggml operations and multi-GPU were not validated
in these original smoke checks; subsequent multi-GPU evidence is linked above.
HIP, SYCL and Metal were not built because the add-on does not support them;
their existing Strata sources were left unchanged.

A subsequent cache-sizing correction passed focused CPU tests (2/2) and
CUDA tests (4/4). Small uniform Q2_K/Q3_K fixtures cover bounded allocation
(two slots rather than eight model experts), explicit slot counts,
zero/sub-slot budgets, deferred allocation and release/reopen. Standard
GGUF launcher fixtures also check automatic, explicit and no-room budgets
through generation. These regressions do not establish a throughput gain.
