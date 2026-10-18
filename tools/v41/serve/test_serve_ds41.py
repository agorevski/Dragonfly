"""CPU-only adapter tests: python -m unittest discover -s tools/v41/serve -p 'test_*.py' -v"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.v41.serve import export_tokenizer as X
from tools.v41.serve import launch as L
from tools.v41.serve import serve_ds41 as D
from strata_tokenizer import BYTE_TO_UNICODE, Tokenizer


@contextmanager
def fixture():
    # Fixtures deliberately stay in the checkout, never the system temporary directory.
    path = ROOT / "tools/v41/serve" / (".test-" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def make_tokenizer(path):
    specials = [D.E.bos_token, D.E.eos_token, D.E.USER_SP_TOKEN, D.E.ASSISTANT_SP_TOKEN,
                "<｜System｜>", D.E.thinking_start_token, D.E.thinking_end_token]
    tokens = list(BYTE_TO_UNICODE.values()) + specials
    data = {"model": {"type": "BPE", "vocab": {t: i for i, t in enumerate(tokens)}, "merges": []},
            "added_tokens": [{"id": i + 256, "content": t, "special": t != D.E.USER_SP_TOKEN}
                             for i, t in enumerate(specials)]}
    (path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    (path / "config.json").write_text(json.dumps({"text_config": {"vocab_size": len(tokens)}}))
    X.export_tokenizer(path, path / "export")
    types = json.loads((path / "export/token_type.json").read_text())
    return D.Ds41Tokenizer(tokens, [], types)


def dsml(calls):
    return D.CALLS_OPEN + "\n" + "\n".join(
        D.INVOKE_OPEN + f' name="{name}">\n'
        + D.E.encode_arguments_to_dsml({"arguments": arguments})
        + "\n" + D.INVOKE_CLOSE for name, arguments in calls) + "\n" + D.CALLS_CLOSE


def request(port, path, data=None, key=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 None if data is None else json.dumps(data).encode(), headers)
    return urllib.request.urlopen(req, timeout=10)


@contextmanager
def server(path, scripts, real_engine=False):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    command = [sys.executable, str(L.HERE / "serve_ds41.py"), "--engine",
               "strata" if real_engine else "mock", "--port", str(port), "--api-key", "test-key"]
    if real_engine:
        command = [sys.executable, str(L.HERE / "launch.py"), "--engine", str(path / "mock-engine"),
                   "--model", str(path / "model.gguf"), "--config", str(path / "config.json"),
                   "--tokenizer", str(path / "export"), "--backend", "cpu",
                   "--port", str(port), "--api-key", "test-key",
                   "--write-config", str(path / "server.json")]
    else:
        command += ["--tokenizer", str(path / "export")]
        for script in scripts:
            command += ["--script", script]
    with (path / "http.log").open("w+") as log:
        env = dict(os.environ)
        env.pop("STRATA_API_KEY", None)
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            for _ in range(150):
                if proc.poll() is not None:
                    log.seek(0)
                    raise AssertionError("server exited:\n" + log.read())
                try:
                    with request(port, "/health", key="test-key") as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.HTTPError):
                    time.sleep(.1)
            else:
                log.seek(0)
                raise AssertionError("server not responsive:\n" + log.read())
            yield port
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


class EncodingTests(unittest.TestCase):
    def test_import_does_not_patch_core(self):
        self.assertIs(D.S.OutputParser, D.F.OutputParser)
        self.assertIsNot(D.S.ChatTemplate, D.Ds41Template)
        self.assertIsNot(Tokenizer, D.Ds41Tokenizer)
        self.assertEqual(D.S.IM_END, "<|im_end|>")

    def test_sequential_presplit_and_special_token_roundtrip(self):
        with fixture() as path:
            tok = make_tokenizer(path)
            self.assertEqual(tok.pre_split("abc漢字def01234"), ["abc", "漢字", "def", "012", "34"])
            text = "abc漢字def 01234 café 👋\n" + D.E.USER_SP_TOKEN + "</think>"
            self.assertEqual(tok.decode(tok.encode(text, parse_special=True)), text)
            self.assertEqual(len(tok.encode(D.E.eos_token, parse_special=True)), 1)
            self.assertGreater(len(tok.encode(D.E.eos_token, parse_special=False)), 1)
            self.assertEqual(len(tok.encode(D.E.USER_SP_TOKEN, parse_special=False)), 1)
            # A merge across the CJK/digit boundaries must not be applied.
            tokens = list(BYTE_TO_UNICODE.values()) + ["12", "123", "1234"]
            merged = D.Ds41Tokenizer(tokens, ["1 2", "12 3", "123 4"])
            self.assertEqual(merged.encode("1234"), [merged.ids["123"], merged.ids["4"]])

    def test_template_reasoning_tools_and_history(self):
        template = D.Ds41Template()
        tools = [{"name": "read", "parameters": {"type": "object"}}]
        messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "file"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "b", "function": {"name": "read", "arguments": {"path": "b"}}},
                        {"id": "a", "function": {"name": "read", "arguments": '{"path":"a"}'}}]},
                    {"role": "tool", "tool_call_id": "a", "content": "AAA"},
                    {"role": "tool", "tool_call_id": "b", "content": "BBB"},
                    {"role": "system", "content": "late system"}]
        prompt = template.render(messages, tools, reasoning_effort=100)
        self.assertIn("Reasoning Effort: 100 ", prompt)
        self.assertIn('<｜DSML｜ invoke name="read">', prompt)
        self.assertLess(prompt.index("BBB"), prompt.index("AAA"))
        self.assertIn("<｜System｜>late system", prompt)
        self.assertTrue(prompt.endswith("<｜Assistant｜><think>"))
        self.assertTrue(template.render(messages, enable_thinking=False).endswith("<｜Assistant｜></think>"))
        self.assertNotIn("Reasoning Effort:", template.render(messages, reasoning_effort="none"))
        self.assertFalse(template.render(messages, add_generation_prompt=False).endswith("<think>"))
        for value in ("turbo", 101, -1, True):
            with self.assertRaises(ValueError):
                D.effort_kwargs(value)
        with self.assertRaisesRegex(ValueError, "text only"):
            template.render([{"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}])

    def test_dsml_incremental_reasoning_and_parallel_calls(self):
        calls = [("read", {"path": "a b.txt", "lines": 3}),
                 ("read", {"path": "x</y>", "nested": {"yes": True}, "null": None})]
        for thinking in (True, False):
            text = ("inspect</think>" if thinking else "") + "Answer." + dsml(calls)
            for split in range(1, 50):
                parser = D.Ds41OutputParser(thinking=thinking, tools=[{"name": "read"}])
                events = []
                for i in range(0, len(text), split):
                    events.extend(parser.feed(text[i:i + split]))
                events.extend(parser.finish())
                self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"),
                                 "inspect" if thinking else "")
                self.assertEqual("".join(e.text for e in events if e.kind == "content").strip(), "Answer.")
                self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"], calls)
        parser = D.Ds41OutputParser(thinking=False, tools=[{"name": "read"}])
        events = parser.feed(dsml(calls)[:40]) + parser.finish("length")
        self.assertFalse(any(e.kind == "tool_call" for e in events))
        with self.assertRaises(ValueError):
            D.parse_tool_call(' name="x"><｜DSML｜ parameter name="x" string="false">no</｜DSML｜ parameter>')
        self.assertEqual(D.unescape("a&lt;" + D.P_CLOSE[1:]), "a" + D.P_CLOSE)
        parser = D.Ds41OutputParser(thinking=False, tools=[{"name": "other"}])
        self.assertFalse(any(e.kind == "tool_call" for e in parser.feed(dsml(calls)) + parser.finish()))

    def test_api_converters_preserve_parallel_tool_ids(self):
        request = {"messages": [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "b", "function": {"name": "read", "arguments": "{}"}},
                {"id": "a", "function": {"name": "read", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "AAA"},
            {"role": "tool", "tool_call_id": "b", "content": "BBB"}]}
        messages, _, _ = D.openai_to_messages(request)
        self.assertEqual([c["id"] for c in messages[0]["tool_calls"]], ["b", "a"])
        self.assertEqual([m["tool_call_id"] for m in messages[1:]], ["a", "b"])
        anthropic = {"messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "b", "name": "read", "input": {}},
                                               {"type": "tool_use", "id": "a", "name": "read", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "AAA"},
                                        {"type": "tool_result", "tool_use_id": "b", "content": "BBB"}]}]}
        messages, _, _ = D.anthropic_to_messages(anthropic)
        self.assertEqual([c["id"] for c in messages[0]["tool_calls"]], ["b", "a"])
        self.assertEqual([m["tool_call_id"] for m in messages[1:]], ["a", "b"])


class LaunchTests(unittest.TestCase):
    def test_export_validation(self):
        with fixture() as path:
            make_tokenizer(path)
            (path / "config.json").write_text('{"vocab_size":1}')
            with self.assertRaisesRegex(ValueError, "vocab_size"):
                X.export_tokenizer(path, path / "bad")
            data = json.loads((path / "tokenizer.json").read_text())
            data["model"]["vocab"]["bad"] = 9999
            (path / "tokenizer.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "contiguous"):
                X.export_tokenizer(path / "tokenizer.json", path / "bad")

    def test_config_absolute_paths_embedded_engram_and_optional_external(self):
        with fixture() as path:
            make_tokenizer(path)
            model = path / "model with spaces.gguf"
            model.write_bytes(b"GGUF")
            args = argparse.Namespace(engine=sys.executable, model=str(model), config=str(path / "config.json"),
                                      tokenizer=str(path / "export"), context=4096, backend="cuda", expert_gpus="auto",
                                      engram=[], engine_arg=[], model_name="test", host="127.0.0.1")
            cfg = L.make_config(args)
            self.assertEqual(cfg["args"][cfg["args"].index("--expert-gpus") + 1], "auto")
            args.expert_gpus = "0"
            self.assertEqual(L.make_config(args)["args"][-1], "0")
            args.backend = "cpu"
            with self.assertRaisesRegex(ValueError, "--expert-gpus requires"):
                L.make_config(args)
            args.backend, args.expert_gpus = "cuda", "auto"
            self.assertEqual(cfg["env"]["STRATA_V41_CONFIG"], str(path / "config.json"))
            self.assertNotIn("--engram", cfg["args"])
            self.assertNotIn("STRATA_SERVE_DSPARK", cfg["env"])
            args.engram = [str(path)]
            self.assertIn("--engram", L.make_config(args)["args"])
            output = path / "runtime.json"
            proc = subprocess.run([sys.executable, str(L.HERE / "launch.py"), "--engine", sys.executable,
                                   "--model", str(model), "--config", str(path / "config.json"),
                                   "--tokenizer", str(path / "export"), "--write-config", str(output),
                                   "--configure-only"], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(output.read_text())["exe"], str(Path(sys.executable).resolve()))

    def test_bind_security(self):
        for host in ("127.0.0.1", "::1", "localhost"):
            D.validate_binding(host, None)
        for host in ("0.0.0.0", "::", "192.168.1.1", "example.test"):
            with self.assertRaises(ValueError):
                D.validate_binding(host, None)
            D.validate_binding(host, "secret")
        proc = subprocess.run([sys.executable, str(L.HERE / "serve_ds41.py"), "--host", "0.0.0.0"],
                              env={k: v for k, v in os.environ.items() if k != "STRATA_API_KEY"},
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("requires --api-key", proc.stderr)

    def test_mock_http_openai_anthropic_streaming_auth(self):
        calls = [("read", {"path": "a.txt", "lines": 3})]
        script = "inspect</think>Answer." + dsml(calls)
        with fixture() as path:
            make_tokenizer(path)
            with server(path, [script, script, script, script, "plain answer"]) as port:
                tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
                req = {"model": "test", "messages": [{"role": "user", "content": "read a"}],
                       "tools": tools, "max_tokens": 600}
                with self.assertRaises(urllib.error.HTTPError) as denied:
                    request(port, "/v1/chat/completions", req)
                self.assertEqual(denied.exception.code, 401)
                with request(port, "/v1/chat/completions", req, "test-key") as response:
                    result = json.load(response)["choices"][0]
                self.assertEqual(result["finish_reason"], "tool_calls")
                self.assertEqual(result["message"]["reasoning_content"], "inspect")
                self.assertEqual(json.loads(result["message"]["tool_calls"][0]["function"]["arguments"]), calls[0][1])
                anth = {"model": "test", "max_tokens": 600, "messages": req["messages"],
                        "tools": [{"name": "read", "input_schema": {"type": "object"}}]}
                with request(port, "/v1/messages", anth, "test-key") as response:
                    result = json.load(response)
                self.assertEqual(result["stop_reason"], "tool_use")
                self.assertEqual([b["input"] for b in result["content"] if b["type"] == "tool_use"], [calls[0][1]])
                with request(port, "/v1/chat/completions", dict(req, stream=True), "test-key") as response:
                    streamed = response.read().decode()
                self.assertIn("[DONE]", streamed)
                chunks = [json.loads(line[6:]) for line in streamed.splitlines()
                          if line.startswith("data: ") and line != "data: [DONE]"]
                deltas = [c["choices"][0]["delta"] for c in chunks]
                self.assertEqual("".join(d.get("reasoning_content", "") for d in deltas), "inspect")
                self.assertEqual("".join(d.get("content", "") for d in deltas).strip(), "Answer.")
                streamed_calls = [c for delta in deltas for c in delta.get("tool_calls", [])]
                self.assertEqual(len(streamed_calls), 1)
                self.assertEqual(streamed_calls[0]["function"]["name"], "read")
                self.assertEqual(json.loads(streamed_calls[0]["function"]["arguments"]), calls[0][1])
                self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "tool_calls")
                self.assertNotIn("DSML", streamed)
                with request(port, "/v1/messages", dict(anth, stream=True), "test-key") as response:
                    streamed = response.read().decode()
                events = [json.loads(line[6:]) for line in streamed.splitlines() if line.startswith("data: ")]
                starts = [e["content_block"] for e in events if e.get("type") == "content_block_start"]
                self.assertEqual([b["name"] for b in starts if b["type"] == "tool_use"], ["read"])
                arguments = "".join(e.get("delta", {}).get("partial_json", "") for e in events)
                self.assertEqual(json.loads(arguments), calls[0][1])
                self.assertEqual("".join(e.get("delta", {}).get("thinking", "") for e in events), "inspect")
                self.assertEqual([e["delta"]["stop_reason"] for e in events
                                  if e.get("type") == "message_delta"], ["tool_use"])
                self.assertEqual(events[-1]["type"], "message_stop")
                self.assertNotIn("DSML", streamed)
                with request(port, "/v1/chat/completions",
                             dict(req, reasoning_effort="none", tool_choice="none"), "test-key") as response:
                    result = json.load(response)["choices"][0]
                self.assertEqual(result["message"]["content"], "plain answer")
                self.assertFalse(result["message"].get("reasoning_content"))
                with self.assertRaises(urllib.error.HTTPError) as unsupported:
                    request(port, "/v1/chat/completions", dict(req, tool_choice="required"), "test-key")
                self.assertEqual(unsupported.exception.code, 400)
                with self.assertRaises(urllib.error.HTTPError) as unsupported:
                    request(port, "/v1/chat/completions", dict(req, presence_penalty=1), "test-key")
                self.assertEqual(unsupported.exception.code, 400)

    def test_real_engine_protocol_with_synthetic_executable(self):
        with fixture() as path:
            tok = make_tokenizer(path)
            ids = tok.encode("ready</think>hello", parse_special=True) + tok.encode(D.E.eos_token, parse_special=True)
            engine = path / "mock-engine"
            engine.write_text(
                f"#!{sys.executable}\nimport os,sys\n"
                "assert os.path.isfile(os.environ['STRATA_V41_CONFIG'])\n"
                "assert '--engram' not in sys.argv\n"
                "assert '--serve' in sys.argv\n"
                "print('READY 4096 stop', flush=True)\n"
                "for line in sys.stdin:\n"
                " if line.startswith('GEN '):\n"
                "  assert ' temperature=0.0 ' in line and ' seed=0 ' in line\n"
                f"  open({str(path / 'prompt.txt')!r}, 'w').write(line)\n"
                f"  for token in {ids!r}: print('T', token, flush=True)\n"
                f"  print('DONE {len(ids)} 100 1 1 stop 0 0 0 0 0 0 0 0 0', flush=True)\n"
                " elif line.strip() == 'QUIT': break\n")
            engine.chmod(0o755)
            model = path / "model.gguf"
            model.write_bytes(b"GGUF")
            args = argparse.Namespace(engine=str(engine), model=str(model), config=str(path / "config.json"),
                                      tokenizer=str(path / "export"), context=4096, backend="cpu", expert_gpus="auto",
                                      engram=[], engine_arg=[], model_name="test", host="127.0.0.1")
            (path / "server.json").write_text(json.dumps(L.make_config(args)))
            with server(path, [], real_engine=True) as port:
                with request(port, "/v1/chat/completions",
                             {"messages": [{"role": "user", "content": "hi"},
                                           {"role": "assistant", "content": "previous"},
                                           {"role": "system", "content": "late instruction"}],
                              "max_tokens": 100, "reasoning_effort": 100,
                              "temperature": 0, "seed": 0}, "test-key") as response:
                    result = json.load(response)["choices"][0]
                self.assertEqual(result["message"]["content"], "hello")
                self.assertEqual(result["message"]["reasoning_content"], "ready")
                wire = (path / "prompt.txt").read_text().split()[-1]
                prompt = tok.decode([int(i) for i in wire.split(",")])
                self.assertIn("<｜System｜>late instruction", prompt)
                self.assertIn("Reasoning Effort: 100 ", prompt)
                self.assertNotIn("<|im_start|>", prompt)


if __name__ == "__main__":
    unittest.main()
