#!/usr/bin/env python3
"""
test_server.py — Smoke test for dotLLM Server API endpoints.

Starts the server with a model, runs a suite of HTTP tests against the
OpenAI-compatible API, and reports results. Requires a model to be
pre-downloaded (uses the first cached model found).

Usage:
    python scripts/test_server.py                             # auto-detect model
    python scripts/test_server.py --model bartowski/Llama-3.2-1B-Instruct-GGUF --quant Q8_0
    python scripts/test_server.py --base-url http://localhost:8080  # test already-running server
    python scripts/test_server.py --device gpu
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PORT = 18080  # non-standard port to avoid conflicts
DEFAULT_MODELS = [
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Q8_0"),
    ("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "Q8_0"),
    ("QuantFactory/SmolLM-135M-GGUF", "Q8_0"),
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str
    elapsed: float


def api_call(base_url: str, path: str, body: dict | None = None,
             method: str = "GET", timeout: float = 120) -> tuple[int, dict | str]:
    """Make an HTTP request and return (status_code, parsed_json_or_text)."""
    url = f"{base_url}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    headers = {"Content-Type": "application/json"} if body else {}

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        return e.code, body_text
    except urllib.error.URLError as e:
        return 0, str(e)


def api_stream(base_url: str, path: str, body: dict,
               timeout: float = 120) -> tuple[int, list[dict], str | None]:
    """Make a streaming SSE request, collect chunks, return (status, chunks, error)."""
    url = f"{base_url}{path}"
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            chunks = []
            for line in resp:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunks.append(json.loads(payload))
                    except json.JSONDecodeError:
                        pass
            return resp.status, chunks, None
    except Exception as e:
        return 0, [], str(e)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_health(base_url: str) -> TestResult:
    start = time.monotonic()
    status, body = api_call(base_url, "/health")
    elapsed = time.monotonic() - start
    if status == 200 and isinstance(body, dict) and body.get("status") == "ok":
        return TestResult("GET /health", True, "ok", elapsed)
    return TestResult("GET /health", False, f"status={status}", elapsed)


def test_ready(base_url: str) -> TestResult:
    start = time.monotonic()
    status, body = api_call(base_url, "/ready")
    elapsed = time.monotonic() - start
    if status == 200:
        return TestResult("GET /ready", True, "ready", elapsed)
    return TestResult("GET /ready", False, f"status={status}", elapsed)


def test_models(base_url: str) -> TestResult:
    start = time.monotonic()
    status, body = api_call(base_url, "/v1/models")
    elapsed = time.monotonic() - start
    if status == 200 and isinstance(body, dict) and "data" in body and len(body["data"]) > 0:
        model_id = body["data"][0]["id"]
        return TestResult("GET /v1/models", True, f"model={model_id}", elapsed)
    return TestResult("GET /v1/models", False, f"status={status}", elapsed)


def test_tokenize(base_url: str) -> TestResult:
    start = time.monotonic()
    status, body = api_call(base_url, "/v1/tokenize",
                            {"text": "Hello world"}, method="POST")
    elapsed = time.monotonic() - start
    if status == 200 and isinstance(body, dict) and "tokens" in body:
        count = body.get("count", len(body["tokens"]))
        return TestResult("POST /v1/tokenize", True, f"{count} tokens", elapsed)
    return TestResult("POST /v1/tokenize", False, f"status={status}", elapsed)


def test_detokenize(base_url: str) -> TestResult:
    # First tokenize to get token IDs
    _, tok_resp = api_call(base_url, "/v1/tokenize",
                           {"text": "Hello world"}, method="POST")
    if not isinstance(tok_resp, dict) or "tokens" not in tok_resp:
        return TestResult("POST /v1/detokenize", False, "tokenize failed", 0)

    start = time.monotonic()
    status, body = api_call(base_url, "/v1/detokenize",
                            {"tokens": tok_resp["tokens"]}, method="POST")
    elapsed = time.monotonic() - start
    if status == 200 and isinstance(body, dict) and "text" in body:
        text = body["text"]
        ok = "hello" in text.lower()
        return TestResult("POST /v1/detokenize", ok,
                         f"'{text}'" if ok else f"unexpected: '{text}'", elapsed)
    return TestResult("POST /v1/detokenize", False, f"status={status}", elapsed)


def test_chat_completion(base_url: str) -> TestResult:
    start = time.monotonic()
    status, body = api_call(base_url, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        "max_tokens": 10,
        "temperature": 0,
    }, method="POST")
    elapsed = time.monotonic() - start

    if status != 200:
        detail = body[:100] if isinstance(body, str) else f"status={status}"
        return TestResult("POST /v1/chat/completions", False, detail, elapsed)

    if not isinstance(body, dict) or "choices" not in body:
        return TestResult("POST /v1/chat/completions", False, "missing choices", elapsed)

    choice = body["choices"][0]
    content = choice.get("message", {}).get("content", "")
    finish = choice.get("finish_reason", "")
    has_usage = "usage" in body
    has_4 = "4" in content

    detail = f"'{content.strip()[:40]}' finish={finish} usage={'yes' if has_usage else 'no'}"
    return TestResult("POST /v1/chat/completions", has_4 and has_usage, detail, elapsed)


def test_chat_streaming(base_url: str) -> TestResult:
    start = time.monotonic()
    status, chunks, error = api_stream(base_url, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10,
        "temperature": 0,
        "stream": True,
    })
    elapsed = time.monotonic() - start

    if error:
        return TestResult("POST /v1/chat/completions (stream)", False, error[:60], elapsed)
    if status != 200:
        return TestResult("POST /v1/chat/completions (stream)", False, f"status={status}", elapsed)

    # Check we got chunks with content deltas
    content_chunks = [c for c in chunks
                      if c.get("choices", [{}])[0].get("delta", {}).get("content")]
    text = "".join(
        c["choices"][0]["delta"]["content"] for c in content_chunks)

    # Check last chunk has finish_reason
    has_finish = any(
        c.get("choices", [{}])[0].get("finish_reason") is not None
        for c in chunks)

    detail = f"{len(chunks)} chunks, text='{text.strip()[:30]}', finish={'yes' if has_finish else 'no'}"
    ok = len(content_chunks) > 0 and has_finish
    return TestResult("POST /v1/chat/completions (stream)", ok, detail, elapsed)


def test_completion(base_url: str) -> TestResult:
    start = time.monotonic()
    status, body = api_call(base_url, "/v1/completions", {
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0,
    }, method="POST")
    elapsed = time.monotonic() - start

    if status != 200:
        return TestResult("POST /v1/completions", False, f"status={status}", elapsed)
    if not isinstance(body, dict) or "choices" not in body:
        return TestResult("POST /v1/completions", False, "missing choices", elapsed)

    text = body["choices"][0].get("text", "")
    ok = len(text.strip()) > 0
    return TestResult("POST /v1/completions", ok, f"'{text.strip()[:40]}'", elapsed)


def test_chat_tools(base_url: str) -> TestResult:
    start = time.monotonic()
    status, body = api_call(base_url, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "max_tokens": 60,
        "temperature": 0,
        "tools": TOOLS,
    }, method="POST", timeout=180)
    elapsed = time.monotonic() - start

    if status != 200:
        detail = body[:100] if isinstance(body, str) else f"status={status}"
        return TestResult("POST /v1/chat/completions (tools)", False, detail, elapsed)

    if not isinstance(body, dict) or "choices" not in body:
        return TestResult("POST /v1/chat/completions (tools)", False, "missing choices", elapsed)

    choice = body["choices"][0]
    finish = choice.get("finish_reason", "")
    content = choice.get("message", {}).get("content", "") or ""
    tool_calls = choice.get("message", {}).get("tool_calls")

    # Tool call detected
    if finish == "tool_calls" and tool_calls and len(tool_calls) > 0:
        tc = tool_calls[0]
        name = tc.get("function", {}).get("name", "")
        detail = f"tool_call: {name}, finish={finish}"
        return TestResult("POST /v1/chat/completions (tools)", True, detail, elapsed)

    # Model mentioned weather in text (some models don't produce structured calls)
    if "weather" in content.lower() or "get_weather" in content:
        detail = f"text match: '{content[:40]}', finish={finish}"
        return TestResult("POST /v1/chat/completions (tools)", True, detail, elapsed)

    detail = f"no tool call, text='{content[:40]}', finish={finish}"
    return TestResult("POST /v1/chat/completions (tools)", False, detail, elapsed)


def test_response_format_json(base_url: str) -> TestResult:
    start = time.monotonic()
    status, body = api_call(base_url, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "Return a JSON object with key 'answer' set to 42."}],
        "max_tokens": 30,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }, method="POST")
    elapsed = time.monotonic() - start

    if status != 200:
        return TestResult("POST /v1/chat/completions (json_object)", False, f"status={status}", elapsed)

    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
        ok = isinstance(parsed, dict)
        detail = f"valid JSON: {json.dumps(parsed)[:50]}"
        return TestResult("POST /v1/chat/completions (json_object)", ok, detail, elapsed)
    except json.JSONDecodeError:
        return TestResult("POST /v1/chat/completions (json_object)", False,
                         f"invalid JSON: {content[:50]}", elapsed)


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

def find_cached_model() -> tuple[str, str | None] | None:
    """Find the first cached model from DEFAULT_MODELS."""
    models_dir = Path.home() / ".dotllm" / "models"
    for repo, quant in DEFAULT_MODELS:
        repo_dir = models_dir / repo.replace("/", os.sep)
        if not repo_dir.exists():
            continue
        gguf_files = [f for f in repo_dir.iterdir() if f.suffix == ".gguf"]
        if quant:
            gguf_files = [f for f in gguf_files if quant.lower() in f.name.lower()]
        if gguf_files:
            return repo, quant
    return None


def start_server(model: str, quant: str | None, port: int,
                 device: str = "cpu") -> subprocess.Popen:
    """Start the dotLLM server as a subprocess."""
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        "dotnet", "run",
        "--project", str(repo_root / "src" / "DotLLM.Server"),
        "-c", "Release", "--",
        "--model", model,
        "--port", str(port),
        "--device", device,
    ]
    if quant:
        cmd += ["--quant", quant]

    print(f"[server] Starting: {' '.join(cmd[-6:])}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )
    return proc


def wait_for_server(base_url: str, timeout: float = 120) -> bool:
    """Wait until the server responds to /health."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _ = api_call(base_url, "/health", timeout=2)
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_health,
    test_ready,
    test_models,
    test_tokenize,
    test_detokenize,
    test_completion,
    test_chat_completion,
    test_chat_streaming,
    test_response_format_json,
    test_chat_tools,
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for dotLLM Server API.")
    parser.add_argument("--model", type=str, help="Model repo or path")
    parser.add_argument("--quant", type=str, help="Quantization filter")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Test an already-running server (skip startup)")
    args = parser.parse_args()

    server_proc = None
    base_url = args.base_url

    try:
        if base_url is None:
            # Start server
            model = args.model
            quant = args.quant
            if model is None:
                found = find_cached_model()
                if found is None:
                    print("No cached models found. Use --model to specify.")
                    return 1
                model, quant = found
                print(f"[server] Auto-detected model: {model} ({quant})")

            server_proc = start_server(model, quant, args.port, args.device)
            base_url = f"http://localhost:{args.port}"

            print(f"[server] Waiting for server at {base_url}...")
            if not wait_for_server(base_url, timeout=120):
                print("[server] FAILED — server did not start in time")
                # Print server output for debugging
                server_proc.terminate()
                stdout, _ = server_proc.communicate(timeout=5)
                if stdout:
                    print(stdout[-500:])
                return 1
            print(f"[server] Ready!")
        else:
            print(f"[server] Using existing server at {base_url}")

        # Run tests
        print()
        print(f"{'Test':<45} {'Result':<8} {'Time':>8}  Details")
        print("=" * 100)

        passed = 0
        failed = 0

        for test_fn in ALL_TESTS:
            try:
                result = test_fn(base_url)
            except Exception as e:
                result = TestResult(test_fn.__name__, False, str(e)[:60], 0)

            status = "PASS" if result.passed else "FAIL"
            time_str = f"{result.elapsed:.1f}s"
            detail = result.detail[:50]

            if result.passed:
                passed += 1
            else:
                failed += 1

            print(f"{result.name:<45} {status:<8} {time_str:>8}  {detail}")

        print("=" * 100)
        total = passed + failed
        print(f"\n{passed}/{total} passed, {failed} failed")
        return 1 if failed > 0 else 0

    finally:
        if server_proc is not None:
            print("\n[server] Shutting down...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
