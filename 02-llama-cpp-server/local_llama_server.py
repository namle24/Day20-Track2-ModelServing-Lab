#!/usr/bin/env python3
"""Minimal OpenAI-compatible llama.cpp server with Prometheus metrics.

The packaged `llama_cpp.server` version available in this environment does not
expose `/metrics`. This wrapper keeps the lab surface the same for clients:
`POST /v1/chat/completions` and `GET /metrics`, backed by `llama_cpp.Llama`.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from llama_cpp import Llama


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def make_app(model_path: str, n_threads: int, n_ctx: int, n_batch: int, n_gpu_layers: int) -> FastAPI:
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )
    lock = threading.Lock()
    stats = {
        "requests_total": 0,
        "requests_processing": 0,
        "requests_deferred": 0,
        "tokens_predicted_total": 0,
        "prompt_tokens_total": 0,
        "kv_cache_usage_ratio": 0.0,
        "kv_cache_tokens": 0,
        "n_decode_total": 0,
        "n_busy_slots_per_decode": 0,
    }

    app = FastAPI(title="Day20 local llama.cpp server")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": Path(model_path).name}

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="messages must be a non-empty list")

        max_tokens = int(payload.get("max_tokens") or 128)
        temperature = float(payload.get("temperature") or 0.3)
        stats["requests_total"] += 1
        if lock.locked():
            stats["requests_deferred"] += 1

        started = int(time.time())
        with lock:
            stats["requests_processing"] += 1
            try:
                result = llm.create_chat_completion(
                    messages=messages,
                    model=payload.get("model") or "local",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False,
                )
            finally:
                stats["requests_processing"] -= 1

        usage = result.get("usage", {}) if isinstance(result, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        stats["prompt_tokens_total"] += prompt_tokens
        stats["tokens_predicted_total"] += completion_tokens
        stats["n_decode_total"] += completion_tokens
        stats["kv_cache_tokens"] = min(n_ctx, max(stats["kv_cache_tokens"], total_tokens))
        stats["kv_cache_usage_ratio"] = min(1.0, stats["kv_cache_tokens"] / max(n_ctx, 1))
        stats["n_busy_slots_per_decode"] = max(stats["n_busy_slots_per_decode"], 1)

        if isinstance(result, dict):
            result.setdefault("id", f"chatcmpl-{started}")
            result.setdefault("object", "chat.completion")
            result.setdefault("created", started)
            result.setdefault("model", payload.get("model") or "local")
            return result
        raise HTTPException(status_code=500, detail="unexpected llama.cpp response")

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        lines = [
            "# HELP llamacpp:tokens_predicted_total Total generated tokens.",
            "# TYPE llamacpp:tokens_predicted_total counter",
            f"llamacpp:tokens_predicted_total {stats['tokens_predicted_total']}",
            "# HELP llamacpp:prompt_tokens_total Total prompt tokens.",
            "# TYPE llamacpp:prompt_tokens_total counter",
            f"llamacpp:prompt_tokens_total {stats['prompt_tokens_total']}",
            "# HELP llamacpp:kv_cache_usage_ratio Approximate peak KV cache occupancy ratio.",
            "# TYPE llamacpp:kv_cache_usage_ratio gauge",
            f"llamacpp:kv_cache_usage_ratio {stats['kv_cache_usage_ratio']:.6f}",
            "# HELP llamacpp:kv_cache_tokens Approximate peak KV cache tokens.",
            "# TYPE llamacpp:kv_cache_tokens gauge",
            f"llamacpp:kv_cache_tokens {stats['kv_cache_tokens']}",
            "# HELP llamacpp:requests_processing Requests currently in inference.",
            "# TYPE llamacpp:requests_processing gauge",
            f"llamacpp:requests_processing {stats['requests_processing']}",
            "# HELP llamacpp:requests_deferred Requests observed while another inference was active.",
            "# TYPE llamacpp:requests_deferred counter",
            f"llamacpp:requests_deferred {stats['requests_deferred']}",
            "# HELP llamacpp:n_decode_total Total decode steps.",
            "# TYPE llamacpp:n_decode_total counter",
            f"llamacpp:n_decode_total {stats['n_decode_total']}",
            "# HELP llamacpp:n_busy_slots_per_decode Busy slots per decode step.",
            "# TYPE llamacpp:n_busy_slots_per_decode gauge",
            f"llamacpp:n_busy_slots_per_decode {stats['n_busy_slots_per_decode']}",
        ]
        return "\n".join(lines) + "\n"

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--n_batch", type=int, default=512)
    parser.add_argument("--n_gpu_layers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = make_app(
        model_path=args.model,
        n_threads=args.n_threads,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_gpu_layers=args.n_gpu_layers,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
