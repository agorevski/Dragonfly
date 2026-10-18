# Bounded V4.1 benchmarks

`benchmark.py` drives the existing `ds41_generate --serve` token protocol directly.
It does not modify the engine or the default server path and needs no downloads.
Only physical GPU0 is visible by default. Use `--gpus 0,1,2,3` for the supported
four-GPU path; the engine defaults to `--expert-gpus auto` within that mask.
Reserve every selected GPU and the full-checkpoint host RAM
before starting; do not run multiple instances concurrently.
The bounded single- and multi-GPU runs and their limitations are in [RESULTS.md](RESULTS.md).

```sh
python -m unittest discover -s tools/v41/bench -p 'test_*.py' -v
python tools/v41/bench/benchmark.py \
  --engine /path/to/ds41_generate --engine-revision BUILD_COMMIT \
  --model /path/to/model-00001-of-00010.gguf \
  --config /path/to/original/config.json --ids-file prompt-64.json \
  --output build-bench/baseline64 --slots 512 --arena-gib 16 \
  --repeats 2 --timeout 600
```

The input file is a JSON array of nonnegative token IDs from the matching local
tokenizer. Record its source/tokenizer and use the same IDs for each comparison.
The output directory must not already exist. Defaults are 64/128-token input
(provided by the caller), 32 output tokens, context 4096, eight CPU threads,
prompt batch 32, chunk 16, PCIe fraction 1 and prediction prefetch 0. The harness
requires at least two output tokens and a full length finish; early EOS, partial
generation, engine errors, nonfinite timings and prefix reuse are failures.

Suggested finite matrix, keeping all other flags identical:

1. Baseline: `--slots 512 --arena-gib 16`.
2. Residency: `--slots auto --vram-margin 2 --arena-gib 64`.
3. Prefill control: 128 IDs, residency flags plus `--step-max 0`.
4. Prefill geometry: the same 128 IDs and control flags plus
   `--prompt-batch 64 --chunk 32`.
5. CPU/PCIe axis: 64 IDs, residency flags plus `--pcie 0` (same thread count).
6. Multi-GPU: residency flags plus `--gpus 0,1,2,3 --expert-gpus auto`.
   Keep `--pcie 1 --pf-b 0`; the pinned tail must fit the total arena budget.

For the integrated roadmap engine, compare each opt-in separately against the
same binary and input: `--device-activations`, `--chunk-batch` (with
`--step-max 0`), `--tier-graph`, and `--dma-overlap`. A graph-enabled build can
be compared with eager ggml execution using `--disable-cuda-graphs`; this does
not disable the independent expert-tier graph switch. Record build-time CUDA
graph settings with the engine revision. All harness switches default off,
and inherited switch values are removed before explicit settings are applied.

The engine normally uses its short-turn passes through 768 new tokens, so
changing prompt batch alone at 64/128 tokens does not exercise layer-major
geometry. `--step-max 0` explicitly sets `STRATA_SERVE_STEP_MAX=0` for the
128-token pair only; inspect the logged actual prefill path.
Each process has one absolute deadline, no retry/fallback, and terminates then
kills its own child if necessary. Arena size is capped at 96 GiB and the
MemAvailable floor cannot be lowered below 40 GiB. Linux MemAvailable, rather
than free RAM, accounts for reclaimable page cache. Quantized Engram tables
share the model mapping and can fault in additional pages: the arena budget
alone is not the process's memory footprint.

Artifacts include exact commands, token IDs, engine/config hashes, declared
build revision, harness revision/hash, model shard sizes/mtimes (not weight
hashes), hardware, raw stdout/stderr and `result.json`. TTFT is request write to
first observed token, including pipe/scheduling overhead; prefill/decode timings
come from DONE. Decode throughput uses output tokens minus one, matching the
engine. Routing diagnostics preserve CPU/PCIe/file expert counts; DONE exposes
decode hit/lookup counts and file routes, **not bytes**. Linux task I/O deltas
are storage/syscall counters, not PCIe traffic; they can include background
arena work and do not measure total system/page-cache traffic.

Repeat zero is **process-cold**, not disk-cold. Later repeats keep the process,
expert caches, and adaptive arena warm but redo full prefill with prefix reuse
disabled. OS page-cache warmth is uncontrolled across all runs. Do not compare
these labels as cold-disk measurements. Route and file biases are zero,
pruning/skipping and encoder skipping are disabled, and no DSpark draft is
loaded. Other STRATA/DS41/GGML environment overrides are removed (except the
explicit settings above). Standard
checkpoint Engram is retained. Keep raw failures and do not report a speedup
from incomplete generation or change defaults based on these short runs.
