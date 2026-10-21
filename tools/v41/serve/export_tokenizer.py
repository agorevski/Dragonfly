#!/usr/bin/env python3
"""Export a local V4.1 HF tokenizer.json to Strata's small tokenizer directory.

Usage: python tools/v41/serve/export_tokenizer.py HF_DIRECTORY OUTPUT_DIRECTORY
No model weights, network access, transformers or tokenizers package is needed.
"""
import argparse
import json
from pathlib import Path


def export_tokenizer(source, output, config=None):
    source, output = Path(source), Path(output)
    if source.is_dir():
        config = Path(config) if config else source / "config.json"
        source = source / "tokenizer.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    model = data["model"]
    if model["type"] != "BPE":
        raise ValueError("expected a byte-level BPE tokenizer")
    vocab = dict(model["vocab"])
    added = data.get("added_tokens", [])
    for item in added:
        if item["content"] in vocab and vocab[item["content"]] != item["id"]:
            raise ValueError("added token disagrees with BPE vocabulary")
        vocab[item["content"]] = item["id"]
    ids = list(vocab.values())
    if sorted(ids) != list(range(len(ids))):
        raise ValueError("tokenizer IDs must be unique and contiguous")
    if config:
        cfg = json.loads(Path(config).read_text(encoding="utf-8"))
        expected = cfg.get("text_config", cfg)["vocab_size"]
        if expected != len(ids):
            raise ValueError(f"config vocab_size {expected} != tokenizer size {len(ids)}")
    types = [1] * len(ids)
    for item in added:
        types[item["id"]] = 3 if item.get("special") else 4
    merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in model["merges"]]
    output.mkdir(parents=True, exist_ok=True)
    (output / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (output / "merges.txt").write_text("\n".join(merges) + ("\n" if merges else ""), encoding="utf-8")
    (output / "token_type.json").write_text(json.dumps(types), encoding="utf-8")
    (output / "chat_template.jinja").write_text(
        "{# V4.1 uses tools/v41/dsv41_encoding.py via serve_ds41.py #}\n", encoding="utf-8")
    (output / "tokenizer_config.json").write_text(
        json.dumps({"pre": "joyai-llm", "source": source.name}), encoding="utf-8")
    return len(ids), len(merges)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--config", type=Path, help="model config.json when source is a tokenizer.json file")
    args = ap.parse_args()
    try:
        tokens, merges = export_tokenizer(args.source, args.output, args.config)
    except (ValueError, KeyError, OSError) as exc:
        ap.error(str(exc))
    print(f"{tokens} tokens, {merges} merges -> {args.output}")


if __name__ == "__main__":
    main()
