#!/usr/bin/env python3
"""TTFT sweep: input length, cross-model, prompt caching, reasoning_effort=minimal.

~11 requests, gpt-oss-20b with reasoning_effort="low" and max_tokens=320 unless
stated otherwise. Every request records finish_reason.

    python scripts/time_ttft.py [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat  # noqa: E402

MAIN = "openai/gpt-oss-20b"
QWEN = "qwen/qwen3.8-27b"
LOG = Path("ttft.jsonl")
QUESTION = ("In one short paragraph, explain what a B-tree is and why databases "
            "use it instead of a hash index for range queries.")
FILLER = ("The archive records daily temperature readings from coastal weather stations, "
          "noting humidity, wind direction, and the time of each observation. ")

COUNT = {"n": 0}
RECORDS: list[dict] = []


def filler_tokens(n_tokens: int) -> str:
    """Roughly n_tokens of context (English is ~4 chars per token)."""
    target = 4 * n_tokens
    out = FILLER * (target // len(FILLER) + 1)
    return out[:target]


def prompt(input_tokens: int) -> list[dict]:
    return [{"role": "user", "content": filler_tokens(input_tokens) + "\n\n" + QUESTION}]


async def one(label: str, model: str, messages, *, reasoning_effort="low",
              max_tokens=320, deadline_s=120.0) -> dict:
    COUNT["n"] += 1
    print(f"  [{COUNT['n']:02d}] {label:24s} {model}", flush=True)
    t0 = time.monotonic()
    r = await chat(model, messages, deadline_s=deadline_s, stream=True,
                   reasoning_effort=reasoning_effort, max_tokens=max_tokens)
    rec = {"label": label, "t_monotonic": t0, "t_monotonic_end": time.monotonic(), **r}
    RECORDS.append(rec)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"        -> {r['status']} http={r['http_status']} "
          f"ttft={None if r['first_token_ms'] is None else round(r['first_token_ms'])}ms "
          f"in={r['tokens_in']} out={r['tokens_out']} reason={r['reasoning_tokens']} "
          f"finish={r['finish_reason']}", flush=True)
    return r


def slope_ms_per_1k(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares slope of ttft(ms) against input tokens, per 1000 tokens."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0:
        return None
    return 1000 * sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


async def main() -> None:
    ap = argparse.ArgumentParser(description="TTFT sweep (~11 requests)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    plan = ["in~250", "in~500", "in~1000", "in~2000", "in~4000", "in~1000(2nd)",
            "qwen~1000 x2", "cache-1500 x2", "minimal x1"]
    if args.dry_run:
        print(f"plan ({len(plan)} groups, 11 requests): " + ", ".join(plan))
        return
    if LOG.exists():
        LOG.unlink()

    print("input-length sweep (gpt-oss-20b, low, max_tokens=320)")
    sweep = [(250, "in~250"), (500, "in~500"), (1000, "in~1000"),
             (2000, "in~2000"), (4000, "in~4000"), (1000, "in~1000(2nd)")]
    for n, label in sweep:
        await one(label, MAIN, prompt(n))

    print("cross-model at equal input (qwen/qwen3.8-27b, ~1000 tokens)")
    for i in range(2):
        await one(f"qwen~1000#{i+1}", QWEN, prompt(1000))

    print("prompt caching: identical 1,500-token prompt twice")
    cache_msgs = prompt(1500)
    for i in range(2):
        await one(f"cache-1500#{i+1}", MAIN, cache_msgs)

    print("reasoning_effort=minimal (single attempt)")
    await one("minimal", MAIN, prompt(1000), reasoning_effort="minimal")

    # ---- reports ----
    print("\nTTFT vs input tokens (gpt-oss-20b)")
    print(f"  {'label':16s} {'tokens_in':>9s} {'ttft_ms':>9s} {'total_ms':>9s} "
          f"{'out':>4s} {'reason':>6s} {'finish':>10s}")
    pts: list[tuple[float, float]] = []
    for r in RECORDS:
        if r["model"] != MAIN or not r["label"].startswith("in~"):
            continue
        if r["status"] != "ok":
            print(f"  {r['label']:16s} {str(r['tokens_in']):>9s} {'-':>9s} "
                  f"{r['status']:>9s}")
            continue
        pts.append((float(r["tokens_in"]), float(r["first_token_ms"])))
        print(f"  {r['label']:16s} {r['tokens_in']:9d} {r['first_token_ms']:9.0f} "
              f"{r['total_ms']:9.0f} {r['tokens_out']:4d} {str(r['reasoning_tokens']):>6s} "
              f"{str(r['finish_reason']):>10s}")
    if len(pts) >= 2:
        s = slope_ms_per_1k([p[0] for p in pts], [p[1] for p in pts])
        print(f"  fitted slope: {s:.2f} ms per 1000 input tokens "
              f"(over {len(pts)} points, {min(p[0] for p in pts):.0f}-{max(p[0] for p in pts):.0f} tokens)")

    base = next((r for r in RECORDS if r["label"] == "in~1000" and r["status"] == "ok"), None)
    print("\nqwen vs gpt-oss at equal input length")
    print(f"  {'model':28s} {'tokens_in':>9s} {'ttft_ms':>9s} {'total_ms':>9s} "
          f"{'reasoning':>9s} {'finish':>12s} {'status':>10s}")
    for r in [base] + [x for x in RECORDS if x["model"] == QWEN]:
        if r is None:
            continue
        ttft = f"{r['first_token_ms']:.0f}" if r["first_token_ms"] else "-"
        tin = str(r["tokens_in"]) if r["tokens_in"] else "-"
        print(f"  {r['model']:28s} {tin:>9s} {ttft:>9s} {r['total_ms']:9.0f} "
              f"{str(r['reasoning_tokens']):>9s} {str(r['finish_reason']):>12s} "
              f"{r['status']:>10s}")
        if r["status"] != "ok":
            print(f"     error_body: {r['error_body']}")

    print("\nprompt caching (identical 1,500-token prompt, back to back)")
    cache = [r for r in RECORDS if r["label"].startswith("cache-1500")]
    for r in cache:
        cached = None
        if r.get("usage"):
            details = r["usage"].get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens")
        print(f"  {r['label']:14s} ttft={r['first_token_ms'] and round(r['first_token_ms'])}ms "
              f"in={r['tokens_in']} cached_tokens={cached} reason={r['reasoning_tokens']} "
              f"finish={r['finish_reason']} status={r['status']}")
    if len(cache) == 2 and all(r["first_token_ms"] for r in cache):
        print(f"  ttft delta second vs first: {cache[1]['first_token_ms'] - cache[0]['first_token_ms']:+.0f} ms")
    fields = set()
    for r in cache:
        fields |= set((r.get("usage") or {}).keys())
        fields |= set(((r.get("usage") or {}).get("prompt_tokens_details") or {}).keys())
    print(f"  usage fields seen: {sorted(fields)}")

    print("\nreasoning_effort=minimal")
    m = next((r for r in RECORDS if r["label"] == "minimal"), None)
    if m:
        print(f"  status={m['status']} http={m['http_status']} "
              f"finish={m['finish_reason']} reason_tokens={m['reasoning_tokens']} "
              f"ttft={m['first_token_ms'] and round(m['first_token_ms'])}ms")
        if m["status"] != "ok":
            print(f"  error_body verbatim: {m['error_body']}")
    print(f"\nrequests used: {COUNT['n']}  log: {LOG}")


if __name__ == "__main__":
    asyncio.run(main())
