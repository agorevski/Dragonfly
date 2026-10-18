"""Exercise the standard 1017 GGUF path without model downloads or Python packages."""

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def fixture(directory):
    prefix = "deepseek41."
    metadata = {"general.architecture": "deepseek41"}
    integers = {
        "embedding_length": 128, "block_count": 2, "attention.head_count": 1,
        "attention.key_length": 64, "rope.dimension_count": 16,
        "attention.q_lora_rank": 128, "attention.output_lora_rank": 64,
        "attention.output_group_count": 1, "attention.sliding_window": 4,
        "expert_count": 2, "expert_used_count": 1, "expert_feed_forward_length": 128,
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
        "candidate_block_size": 2, "hidden_size": 128, "num_hidden_layers": 2,
        "n_routed_experts": 2, "vocab_size": 4, "engram_head_dim": 32,
        "engram_n_heads": 1, "engram_max_ngram_size": 3, "swiglu_limit": 10.,
    }
    tensors = []

    def add(name, shape, type_id=0):
        count = 1
        for dim in shape:
            count *= dim
        if type_id == 8:
            payload = bytes(count // 32 * 34)
        else:
            value = 1. if "norm" in name or "engram_q." in name or "engram_k." in name else 0.
            payload = struct.pack("<f", value) * count
        tensors.append((name, shape, type_id, payload))

    add("token_embd.weight", [128, 4])
    add("output_norm.weight", [128])
    add("output.weight", [128, 4])
    for layer in range(2):
        base = f"blk.{layer}."
        for site in ("attn", "ffn"):
            add(base + f"hc_{site}_fn.weight", [512, 24])
            add(base + f"hc_{site}_base.weight", [24])
            add(base + f"hc_{site}_scale.weight", [3])
            add(base + f"{site}_norm.weight", [128])
        for name, shape in (
            ("attn_sinks", [1]), ("attn_q_a", [128, 128]), ("attn_q_b", [128, 64]),
            ("attn_q_a_norm", [128]), ("attn_kv", [128, 64]), ("attn_kv_a_norm", [64]),
            ("attn_output_a", [64, 64]), ("attn_output_b", [64, 128]),
            ("ffn_gate_inp", [128, 2]),
            ("ffn_gate_shexp", [128, 128]), ("ffn_up_shexp", [128, 128]),
            ("ffn_down_shexp", [128, 128]),
            ("ffn_gate_exps", [128, 128, 2]), ("ffn_up_exps", [128, 128, 2]),
            ("ffn_down_exps", [128, 128, 2]),
        ):
            add(base + name + ".weight", shape)
        add(base + "exp_probs_b.bias", [2])
        add(base + "exp_probs_b_vl.bias", [2])
    for name, shape in (
        ("attn_compressor_kv", [128, 64]), ("attn_compressor_norm", [64]),
        ("attn_compressor_gate", [128, 64]), ("indexer.attn_k", [64, 128]),
        ("indexer.k_norm", [128]), ("indexer.attn_q_b", [128, 4096]),
        ("indexer.proj", [128, 32]), ("engram_q", [128, 4]),
        ("engram_k", [128, 4]), ("engram_wkv", [64, 640]),
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
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    directory = args.work.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    try:
        model, config_path, config = fixture(directory)
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
        for extra, message in (
            (["--chunk-mmq"], "--chunk-mmq is unavailable"),
            (["--expert-gpus", "0,1"], "unknown argument --expert-gpus"),
            (["--experts", "invalid"], "--experts must be cpu|gpu"),
        ):
            result = subprocess.run(command + extra, env=env, text=True, capture_output=True, timeout=60)
            if result.returncode != 2 or message not in result.stderr:
                raise AssertionError("Unsupported option was not rejected: " + result.stderr)
        if not args.dspark_built:
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
