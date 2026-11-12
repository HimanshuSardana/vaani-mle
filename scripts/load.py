#!/usr/bin/env python3
"""Load scenarios: fire pipeline requests on a schedule, one CSV row per request.

    python scripts/load.py --scenario steady --rate 3 --duration 120 --concurrency 4 --source real
    LLM_BASE_URL=http://127.0.0.1:8801/openai/v1 python scripts/load.py \
        --scenario halved --rate 20 --duration 120 --concurrency 8 --source stand-in

Rate is offered requests per minute, so `--rate 600 --duration 4` is a burst of 40.
Request i is scheduled at i * 60 / rate seconds after start and fires whether or not
an earlier request has finished, which is what makes the offered rate real.

Writes the same columns as grid.py plus queued_ms, so peak queue length is
recoverable from the CSV alone: a request waited from start_time - queued_ms to
start_time, and the peak is the largest overlap of those intervals.

answer_grade stays empty: these rows measure latency and outcomes, not accuracy.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import grid  # noqa: E402
import pipeline  # noqa: E402

COLS = grid.COLUMNS + ["queued_ms"]


async def emit_row(w, lock, *, scenario, source, setting, q, tr, queued_ms, total_ms):
    pipeline.write_trace(tr)  # the 429 body and window live only in the trace
    cits = tr.get("citations") or []
    first = tr.get("ttft_user_visible_ms")
    row = {
        "request_id": tr.get("answer_id"),
        "scenario": scenario,
        "source": source,
        "start_time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "first_token_ms": first if first is not None else "",
        "total_ms": total_ms,
        "status": grid.brief_status(tr.get("status")),
        "retries": tr.get("retries", 0),
        "tokens_in": tr.get("draft_tokens_in") or "",
        "tokens_out": tr.get("draft_tokens_out") or "",
        "retrieval_setting": setting,
        "answer_grade": "",
        "qid": q["id"],
        "expect": q["expect"],
        "raw_status": tr.get("status"),
        "first_token_visible_ms": first if first is not None else "",
        "grade_reason": "",
        "citations": len(cits),
        "resolved": sum(1 for c in cits if c.get("chunk_id")),
        "auto_cited": tr.get("auto_cited_count", 0),
        "refusal_retry": tr.get("refusal_retry", False),
        "rounds": tr.get("retrieval_rounds", 0),
        "rung": tr.get("rung"),
        "answer_text": (tr.get("answer_text") or "")[:120],
        "queued_ms": queued_ms,
    }
    async with lock:
        w.writerow(row)


async def one(sem, *, delay, scenario, source, setting, q, w, lock):
    await asyncio.sleep(delay)
    queued_at = time.perf_counter()
    async with sem:
        started = time.perf_counter()
        queued_ms = round((started - queued_at) * 1000, 1)
        try:
            tr = await pipeline.run(q["question"], setting, source)
        except Exception as exc:
            tr = {"status": "error", "error_body": f"{type(exc).__name__}: {exc}",
                  "answer_text": "", "citations": [], "answer_id": uuid_hex()}
        total_ms = round((time.perf_counter() - started) * 1000, 1)
    await emit_row(w, lock, scenario=scenario, source=source, setting=setting, q=q, tr=tr,
             queued_ms=queued_ms, total_ms=total_ms)
    print(f"  {q['id']} {tr.get('status'):13s} rung={tr.get('rung')} "
          f"queue={queued_ms:6.0f}ms total={total_ms:7.0f}ms", flush=True)


def uuid_hex() -> str:
    import uuid
    return uuid.uuid4().hex[:16]


async def main_async(args) -> None:
    spec = json.load(open(ROOT / "questions.json", encoding="utf-8"))
    answerable = [q for q in spec["questions"] if q["expect"] == "answerable"]

    pipeline.decide_rerank_at_startup(pipeline.corpus()[0])
    out = Path(args.out)
    new = not out.exists()
    f = out.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=COLS)
    if new:
        w.writeheader()
    else:
        head = out.open(encoding="utf-8").readline().strip().split(",")
        if head != COLS:
            sys.exit(f"{out} header does not match load.py columns; move it aside first")

    n = max(1, int(round(args.rate * args.duration / 60)))
    interval = 60.0 / args.rate
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    print(f"{args.scenario}: {n} requests over {args.duration}s at {args.rate}/min, "
          f"concurrency {args.concurrency}, source {args.source}", flush=True)

    t0 = time.perf_counter()
    tasks = [
        asyncio.create_task(one(sem, delay=i * interval, scenario=args.scenario,
                                source=args.source, setting=args.setting,
                                q=answerable[i % len(answerable)], w=w, lock=lock))
        for i in range(n)
    ]
    await asyncio.gather(*tasks)
    f.close()
    print(f"{args.scenario} done in {time.perf_counter() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one load scenario into results.csv")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--rate", type=float, required=True, help="offered requests per minute")
    ap.add_argument("--duration", type=float, required=True, help="seconds of arrivals")
    ap.add_argument("--concurrency", type=int, default=4, help="max requests in flight")
    ap.add_argument("--source", choices=["real", "stand-in"], default="real")
    ap.add_argument("--setting", default="S2")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()
    if args.source == "stand-in" and not os.environ.get("LLM_BASE_URL"):
        sys.exit("--source stand-in needs LLM_BASE_URL pointing at the stand-in")
    if args.source == "real" and not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set (load .env first)")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
