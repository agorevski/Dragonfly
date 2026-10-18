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

The root Strata CMake project also exposes `STRATA_ENABLE_DS41`, which is
`OFF` by default. A root-project opt-in build additionally needs
`STRATA_DS41_CUDA=ON` for the DeepSeek CUDA targets; this is separate from
the normal Strata engine's backend settings.

The engine's default `--slots auto` sizes uniform Q2_K/Q3_K expert caches
against the free-VRAM budget, including slot alignment. With batched
prefill, allocation is deferred until the prompt buffers are released.
If the automatic budget has no room, decode uses the RAM/file tiers
without a VRAM expert cache. An explicit `--slots N` remains a manual
capacity choice; it must fit the available VRAM.

## Experimental multi-GPU expert capacity prototype

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
vision, multi-GPU serving, penalties and logit bias. Incremental tool
argument deltas, batching and session persistence are not implemented.

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
Multi-GPU operation and optional speculative/fused paths should not be
inferred from Strata's support for those features on its Qwen models.

## Validation on this machine

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
tool calling, DSpark, fused ggml operations and multi-GPU were not validated.
HIP, SYCL and Metal were not built because the add-on does not support them;
their existing Strata sources were left unchanged.

A subsequent cache-sizing correction passed focused CPU tests (2/2) and
CUDA tests (4/4). Small uniform Q2_K/Q3_K fixtures cover bounded allocation
(two slots rather than eight model experts), explicit slot counts,
zero/sub-slot budgets, deferred allocation and release/reopen. Standard
GGUF launcher fixtures also check automatic, explicit and no-room budgets
through generation. These regressions do not establish a throughput gain.
