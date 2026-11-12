#!/usr/bin/env python3
"""Reranker cost, offline: no API calls.

Loads chunks.jsonl and the cross-encoder ONCE, then over all 15 questions from
questions.json measures p50/p95 ms for
  - bm25 over the full corpus
  - rerank of 25 pairs
  - rerank of 8 pairs
  - rerank of 25 pairs with torch.set_num_threads(1)

    python scripts/time_rerank.py [--chunks chunks.jsonl] [--questions questions.json]
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieve import build_index, tokenize  # noqa: E402

RERANK_MODEL = "cross-encoder/ms-marco-minilm-l-6-v2"


def pctl(vals: list[float], p: float) -> float:
    v = sorted(vals)
    if not v:
        return float("nan")
    k = max(0, min(len(v) - 1, int(round((p / 100) * (len(v) - 1)))))
    return v[k]


def summarise(name: str, times_ms: list[float], pairs: int | None = None) -> None:
    p50 = pctl(times_ms, 50)
    p95 = pctl(times_ms, 95)
    line = (f"{name:28s} n={len(times_ms):2d}  p50={p50:7.2f}ms  p95={p95:7.2f}ms  "
            f"max={max(times_ms):7.2f}ms  min={min(times_ms):7.2f}ms")
    if pairs:
        line += f"  per-pair: p50={p50 / pairs:6.2f}ms p95={p95 / pairs:6.2f}ms"
    print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline reranker cost measurement")
    ap.add_argument("--chunks", default="chunks.jsonl")
    ap.add_argument("--questions", default="questions.json")
    args = ap.parse_args()

    cpus = os.cpu_count()
    aff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    print(f"torch.get_num_threads() = {torch.get_num_threads()}  (as it stands)")
    print(f"cpu_count() = {cpus} | usable/affinity = {aff} | "
          f"torch.get_num_interop_threads() = {torch.get_num_interop_threads()}")
    print(f"model = {RERANK_MODEL}")

    chunks = [json.loads(line) for line in Path(args.chunks).open(encoding="utf-8") if line.strip()]
    qspec = json.load(open(args.questions, encoding="utf-8"))
    questions = [q["question"] for q in qspec["questions"]]
    print(f"corpus = {len(chunks)} chunks | questions = {len(questions)}")

    t0 = time.perf_counter()
    bm25 = build_index(chunks)
    print(f"bm25 index built once in {(time.perf_counter() - t0) * 1000:.1f}ms")

    from sentence_transformers import CrossEncoder
    t0 = time.perf_counter()
    model = CrossEncoder(RERANK_MODEL)
    print(f"cross-encoder loaded once in {(time.perf_counter() - t0) * 1000:.1f}ms")

    q_tokens = [tokenize(q) for q in questions]
    top25: list[list[tuple[str, str]]] = []
    bm25_ms: list[float] = []
    for terms in q_tokens:
        t0 = time.perf_counter()
        scores = bm25.get_scores(terms)
        bm25_ms.append((time.perf_counter() - t0) * 1000)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:25]
        top25.append([chunks[i]["text"] for i in order])

    # one warmup predict so the first timed call is not paying model init
    model.predict([("warmup query", chunks[0]["text"]), ("warmup query", chunks[1]["text"])])

    variants: dict[str, list[float]] = {"rerank_25": [], "rerank_8": [], "rerank_25_1thread": []}
    for idx, pairs in enumerate(top25):
        q = questions[idx]
        p25 = [[q, t] for t in pairs]
        t0 = time.perf_counter()
        model.predict(p25)
        variants["rerank_25"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        model.predict(p25[:8])
        variants["rerank_8"].append((time.perf_counter() - t0) * 1000)

    torch.set_num_threads(1)
    print(f"torch.get_num_threads() now = {torch.get_num_threads()} (single-thread variant)")
    for idx, pairs in enumerate(top25):
        p25 = [[questions[idx], t] for t in pairs]
        t0 = time.perf_counter()
        model.predict(p25)
        variants["rerank_25_1thread"].append((time.perf_counter() - t0) * 1000)

    print()
    summarise("bm25_full_corpus", bm25_ms)
    summarise("rerank_25_pairs", variants["rerank_25"], 25)
    summarise("rerank_8_pairs", variants["rerank_8"], 8)
    summarise("rerank_25_pairs_1thread", variants["rerank_25_1thread"], 25)

    per_pair = statistics.median(variants["rerank_25"]) / 25
    per_pair_1 = statistics.median(variants["rerank_25_1thread"]) / 25
    print(f"\nmedian per-pair (default threads): {per_pair:.2f} ms  ->  "
          f"8 pairs ≈ {per_pair * 8:.0f} ms, 25 pairs ≈ {per_pair * 25:.0f} ms")
    print(f"median per-pair (1 thread)      : {per_pair_1:.2f} ms  ->  "
          f"{per_pair_1 / per_pair:.2f}x the default-thread cost per pair")
    print(f"peak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    torch.set_num_threads(max(1, torch.get_num_threads()))


if __name__ == "__main__":
    main()
