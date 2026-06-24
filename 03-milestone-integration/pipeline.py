#!/usr/bin/env python3
"""Skeleton RAG pipeline gluing N19 retrieval + N20 llama-server.

Replace the STUB markers with your actual N18/N19 code. Runs as-is using
in-memory toy data so you can confirm the OpenAI-compat call before wiring
in your real lakehouse + vector store.
"""
from __future__ import annotations

import time
import os
from dataclasses import dataclass
from typing import Iterable

import httpx

LLAMA_SERVER_BASE = "http://localhost:8080/v1"
PIPELINE_MAX_TOKENS = int(os.environ.get("LAB_PIPELINE_MAX_TOKENS", "64"))
SYSTEM_PROMPT = (
    "You are a serving-engineering tutor. Answer using only the documents provided. "
    "If the documents don't contain the answer, say so."
)


# ────────────────────────────────────────────────────────────────────────
# Replace this STUB with retrieval against your N19 vector index.
# ────────────────────────────────────────────────────────────────────────

TOY_DOCS = [
    {"id": "n20-paged", "text": "PagedAttention treats KV cache like virtual memory pages, eliminating 60-80% fragmentation."},
    {"id": "n20-radix", "text": "RadixAttention stores KV in a prefix trie; cache hit on shared prefix lets the engine skip prefill."},
    {"id": "n20-disagg", "text": "Disaggregated serving (Mooncake, llm-d, Dynamo) splits prefill and decode onto separate GPU pools."},
    {"id": "n20-goodput", "text": "Goodput@SLO = req/s satisfying TTFT and TPOT SLOs. Throughput at saturation ignores SLO."},
    {"id": "n20-quant", "text": "GGUF Q4_K_M is the production-quality default for laptop/edge serving via llama.cpp."},
]


@dataclass
class Doc:
    id: str
    text: str
    score: float


def retrieve(query: str, k: int = 3) -> list[Doc]:
    """STUB: replace with your N19 vector index call."""
    # Toy keyword overlap so the demo does *something* sensible without an embedder.
    q_terms = {w.lower() for w in query.split() if len(w) > 3}
    scored = [
        Doc(d["id"], d["text"], score=len(q_terms & {w.lower() for w in d["text"].split()}))
        for d in TOY_DOCS
    ]
    return sorted(scored, key=lambda d: d.score, reverse=True)[:k]


# ────────────────────────────────────────────────────────────────────────
# Prompt assembly
# ────────────────────────────────────────────────────────────────────────


def build_prompt(query: str, contexts: Iterable[Doc]) -> list[dict]:
    ctx_block = "\n".join(f"[{c.id}] {c.text}" for c in contexts)
    user = f"Context:\n{ctx_block}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ────────────────────────────────────────────────────────────────────────
# llama-server call
# ────────────────────────────────────────────────────────────────────────


def call_llm(messages: list[dict]) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = httpx.post(
        f"{LLAMA_SERVER_BASE}/chat/completions",
        json={"model": "local", "messages": messages, "max_tokens": PIPELINE_MAX_TOKENS, "temperature": 0.3},
        timeout=120.0,
    )
    r.raise_for_status()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return r.json()["choices"][0]["message"]["content"], elapsed_ms


def answer(query: str) -> dict:
    t_total = time.perf_counter()

    t = time.perf_counter()
    docs = retrieve(query, k=3)
    t_retrieve_ms = (time.perf_counter() - t) * 1000.0

    messages = build_prompt(query, docs)

    text, t_llm_ms = call_llm(messages)

    return {
        "query": query,
        "answer": text,
        "contexts": [{"id": d.id, "score": d.score} for d in docs],
        "timings_ms": {
            "retrieve": round(t_retrieve_ms, 1),
            "llm": round(t_llm_ms, 1),
            "total": round((time.perf_counter() - t_total) * 1000.0, 1),
        },
    }


def main() -> None:
    queries = [
        "Why is goodput more useful than throughput?",
        "What problem does PagedAttention actually solve?",
        "When should I think about disaggregated serving?",
    ]
    for q in queries:
        print(f"\n=== {q} ===")
        result = answer(q)
        print(f"  contexts: {[c['id'] for c in result['contexts']]}")
        print(f"  timings : {result['timings_ms']}")
        print(f"  answer  : {result['answer'].strip()[:300]}")


if __name__ == "__main__":
    main()
