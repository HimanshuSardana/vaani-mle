#!/usr/bin/env python3
"""Run the 15-question grid over the three declared retrieval settings.

    python scripts/grid.py [--settings S1 S2 S3] [--spacing 20] [--limit N]

Writes results.csv (one row per question x setting) and appends every request's
trace to traces/grid_trace.jsonl. answer_grade stays empty here: a model judges
answers afterwards and scripts/apply_judge.py fills the column, so one run of the
grid never decides its own correctness.

Spacing: ~1.3k tokens per request against an 8k TPM budget means one request every
~20 s sustained; retries honour retry-after.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import pipeline  # noqa: E402

# Columns required by the brief, in its order, then the extra columns a reviewer
# needs to recompute a row. answer_grade stays empty until a model judges it.
COLUMNS = ["request_id", "scenario", "source", "start_time", "first_token_ms",
           "total_ms", "status", "retries", "tokens_in", "tokens_out",
           "retrieval_setting", "answer_grade", "qid", "expect", "raw_status",
           "first_token_visible_ms", "grade_reason", "citations", "resolved",
           "auto_cited", "refusal_retry", "rounds", "rung", "answer_text"]
BRIEF_STATUS = {"ok", "rate_limited", "timeout", "error"}


def brief_status(raw: str | None) -> str:
    """Map the pipeline's status onto the four values the brief allows."""
    raw = raw or "error"
    if raw in BRIEF_STATUS:
        return raw
    return "ok" if raw in ("refused_ok",) else "error"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", default=["S1", "S2", "S3"])
    ap.add_argument("--spacing", type=float, default=20.0, help="seconds between requests")
    ap.add_argument("--limit", type=int, default=0, help="stop after N questions (debug)")
    ap.add_argument("--qids", nargs="+", default=None,
                    help="only these question ids, e.g. --qids Q13 Q15 Q16")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    pipeline.MAX_RETRIES = 2  # 429/timeout/5xx only, honours retry-after
    spec = json.load(open(ROOT / "questions.json", encoding="utf-8"))
    questions = spec["questions"][: args.limit or None]
    if args.qids:
        want = [q.strip().upper() for q in args.qids]
        questions = [q for q in questions if q["id"] in want]
        missing = [q for q in want if q not in {x["id"] for x in questions}]
        if missing:
            sys.exit(f"unknown question id(s): {', '.join(missing)}")
    pipeline.decide_rerank_at_startup(pipeline.corpus()[0])

    out = Path(args.out)
    new = not out.exists()
    f = out.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=COLUMNS)
    if new:
        w.writeheader()

    done = 0
    total = len(questions) * len(args.settings)
    for q in questions:
        for setting in args.settings:
            done += 1
            t0 = time.monotonic()
            try:
                tr = asyncio.run(pipeline.run(q["question"], setting, "groq"))
            except Exception as exc:  # never lose the row
                tr = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                      "answer_text": "", "citations": [], "retrieval_rounds": 0}
            pipeline.write_trace(tr)
            cits = tr.get("citations") or []
            w.writerow({
                "request_id": tr.get("answer_id"),
                "scenario": f"quality_{setting}",
                "source": "groq",
                "start_time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "first_token_ms": round(tr.get("draft_ttft_ms") or 0, 1),
                "total_ms": round(tr.get("draft_total_ms") or 0, 1),
                "status": brief_status(tr.get("status")),
                "retries": tr.get("retries", 0),
                "tokens_in": tr.get("draft_tokens_in"),
                "tokens_out": tr.get("draft_tokens_out"),
                "retrieval_setting": setting,
                "answer_grade": "",
                "qid": q["id"], "expect": q["expect"],
                "raw_status": tr.get("status"),
                "first_token_visible_ms": tr.get("ttft_user_visible_ms"),
                "grade_reason": "",
                "citations": len(cits),
                "resolved": sum(1 for c in cits if c.get("chunk_id")),
                "auto_cited": tr.get("auto_cited_count", 0),
                "refusal_retry": tr.get("refusal_retry", False),
                "rounds": tr.get("retrieval_rounds", 0), "rung": tr.get("rung"),
                "answer_text": tr.get("answer_text") or "",
            })
            f.flush()
            print(f"[{done}/{total}] {q['id']} {setting} {tr.get('status'):14s} "
                  f"{time.monotonic() - t0:.0f}s", flush=True)
            if done < total:
                time.sleep(args.spacing)

    f.close()
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"\nrows={len(rows)} ok={len(ok)} failed={len(rows) - len(ok)}")
    print("answer_grade is empty until a model judges the answers: "
          "python scripts/apply_judge.py")


if __name__ == "__main__":
    main()
