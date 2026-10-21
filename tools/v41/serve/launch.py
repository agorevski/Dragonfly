#!/usr/bin/env python3
"""Generate an opt-in V4.1 server config and launch the local adapter.

python tools/v41/serve/launch.py --engine build-ds41/ds41_generate \
  --model /models/model.gguf --config /models/config.json --tokenizer /models/tokenizer

--write-config FILE --configure-only generates a reusable config without loading weights.
Then: python tools/v41/serve/serve_ds41.py --engine strata --config FILE --port 8150
Extra engine flags must each use --engine-arg=VALUE; paths are resolved before launch.
External FP8 Engram directories and DSpark are never enabled implicitly.
"""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def existing(path, directory=False):
    path = Path(path).expanduser().resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"{'directory' if directory else 'file'} does not exist: {path}")
    return str(path)


def make_config(args):
    engine = existing(args.engine)
    if not os.access(engine, os.X_OK):
        raise ValueError(f"engine is not executable: {engine}")
    model = existing(args.model)
    config = existing(args.config)
    model_config = json.loads(Path(config).read_text(encoding="utf-8"))
    tokenizer = existing(args.tokenizer, directory=True)
    for name in ("vocab.json", "merges.txt", "token_type.json"):
        existing(Path(tokenizer) / name)
    if args.context < 1:
        raise ValueError("--context must be positive")
    vocab = json.loads((Path(tokenizer) / "vocab.json").read_text(encoding="utf-8"))
    from tools.v41 import dsv41_encoding as encoding
    eos = vocab.get(encoding.eos_token)
    if not isinstance(eos, int) or isinstance(eos, bool):
        raise ValueError("exported V4.1 tokenizer is missing its EOS token")
    expected = model_config.get("text_config", model_config).get("vocab_size")
    if expected != len(vocab):
        raise ValueError("model config vocab_size does not match the exported tokenizer")
    command = ["-m", model, "--backend", args.backend, "--experts",
               "cpu" if args.backend == "cpu" else "gpu", "--ctx", str(args.context), "--stop", str(eos)]
    if args.engram:
        command += ["--engram", ":".join(existing(p, directory=True) for p in args.engram)]
    command += args.engine_arg
    return {"exe": engine, "args": command, "cwd": str(ROOT), "tokenizer": tokenizer,
            "model_name": args.model_name, "arch": "deepseek-v4.1", "backend": args.backend,
            "host": args.host, "env": {"STRATA_V41_CONFIG": config},
            "answer_room": False, "reasoning_budget_tokens": 0,
            "sampling": {"temperature": 1.0, "top_p": 0.95}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="built ds41_generate executable")
    ap.add_argument("--model", required=True, help="1017 GGUF (embedded quantized Engram is supported)")
    ap.add_argument("--config", required=True, help="DeepSeek model config.json, not a server config")
    ap.add_argument("--tokenizer", required=True, help="directory exported by export_tokenizer.py")
    ap.add_argument("--engram", action="append", default=[], help="optional external Engram directory (repeatable)")
    ap.add_argument("--backend", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--model-name", default="deepseek-ai/DeepSeek-V4.1-Flash-1017")
    ap.add_argument("--engine-arg", action="append", default=[])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8150)
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY", ""))
    ap.add_argument("--write-config", type=Path)
    ap.add_argument("--configure-only", action="store_true")
    args = ap.parse_args()
    try:
        sys.path.insert(0, str(ROOT))
        from tools.v41.serve.serve_ds41 import validate_binding
        validate_binding(args.host, args.api_key)
        cfg = make_config(args)
        if args.configure_only and not args.write_config:
            raise ValueError("--configure-only requires --write-config")
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    output = (args.write_config or ROOT / "tools/v41/serve/server.local.json").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"V4.1 server config: {output}", flush=True)
    if args.configure_only:
        return 0
    command = [sys.executable, str(HERE / "serve_ds41.py"), "--engine", "strata", "--config", str(output),
               "--host", args.host, "--port", str(args.port)]
    env = dict(os.environ)
    if args.api_key:
        env["STRATA_API_KEY"] = args.api_key
    os.execve(sys.executable, command, env)


if __name__ == "__main__":
    sys.exit(main())
