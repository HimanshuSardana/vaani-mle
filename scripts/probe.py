#!/usr/bin/env python3
"""Rate-limit / latency probe for Groq. HARD CAP: 25 requests.

    python scripts/probe.py                 # writes probe.jsonl, prints tables
    python scripts/probe.py --dry-run       # no network: prints the plan only

Phases
  1  12 req  openai/gpt-oss-20b, reasoning_effort="low": 6 short (~40 tokens),
             6 medium (~300 tokens) - same instruction, only length varies.
  2   3 req  same model + same short prompt, reasoning_effort=None (default).
  3   3 req  allam-2-7b, short prompts.
  4  <=5 req  openai/gpt-oss-safeguard-20b (RPM 3, TPM 2K): 5 rapid sequential
             requests to trip the limit. Record code, verbatim body, retry-after
             and the wall-clock wait until a request succeeds. No retries.
             Never run against gpt-oss-20b.

No request is retried here except by phase 4's own "keep firing until one
succeeds" rule, and the process exits non-zero rather than exceed the cap.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat  # noqa: E402

CAP = 25
LOG = Path("probe.jsonl")

MODEL_MAIN = "openai/gpt-oss-20b"
MODEL_SMALL = "allam-2-7b"
MODEL_LIMIT = "openai/gpt-oss-safeguard-20b"  # RPM 3, TPM 2K - only for phase 4

COUNT = {"n": 0}
RECORDS: list[dict] = []


async def call(phase: str, model: str, messages, *, deadline_s: float = 60.0,
               reasoning_effort: str | None = None,
               max_tokens: int = 40) -> dict:
    if COUNT["n"] >= CAP:
        sys.exit(f"request cap reached ({CAP}); refusing to continue")
    COUNT["n"] += 1
    print(f"  [{COUNT['n']:02d}/{CAP}] phase {phase} {model} "
          f"effort={reasoning_effort}", flush=True)
    t0 = time.monotonic()
    res = await chat(model, messages, deadline_s=deadline_s, stream=True,
                     reasoning_effort=reasoning_effort, max_tokens=max_tokens)
    rec = {"phase": phase, "t_monotonic": t0, "t_monotonic_end": time.monotonic(),
           "n_request": COUNT["n"], **res}
    RECORDS.append(rec)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"        -> {res['status']} http={res['http_status']} "
          f"ttft={res['first_token_ms']}ms total={res['total_ms']:.0f}ms "
          f"out={res['tokens_out']}", flush=True)
    return res


def prompt(n_tokens: int) -> list[dict]:
    return [{"role": "user",
             "content": f"Explain how a hash table handles collisions, in roughly "
                        f"{n_tokens} tokens. Plain prose, no bullet points."}]


async def phase1() -> None:
    print("phase 1: gpt-oss-20b, reasoning_effort=low, 6 short + 6 medium")
    for i in range(6):
        await call("1-short", MODEL_MAIN, prompt(40), reasoning_effort="low", max_tokens=40)
    for i in range(6):
        await call("1-medium", MODEL_MAIN, prompt(300), reasoning_effort="low", max_tokens=320)


async def phase2() -> None:
    print("phase 2: same model + short prompt, reasoning_effort omitted (default)")
    for i in range(3):
        await call("2-default", MODEL_MAIN, prompt(40), reasoning_effort=None)


async def phase3() -> None:
    print("phase 3: allam-2-7b, short prompts")
    for i in range(3):
        await call("3-small", MODEL_SMALL, prompt(40), reasoning_effort=None)


async def phase4() -> None:
    print("phase 4: trip the rate limit on gpt-oss-safeguard-20b (max 5 req, no retry)")
    first_hit = None
    saw_limit = False
    for i in range(5):
        res = await call("4-limit", MODEL_LIMIT, prompt(40), reasoning_effort="low")
        now = time.monotonic()
        if res["status"] == "rate_limited":
            saw_limit = True
            if first_hit is None:
                first_hit = now
        # only a success AFTER the limit counts as recovery
        if saw_limit and res["status"] == "ok":
            break


async def phase_inflight(n: int = 8) -> dict:
    """Fire n concurrent short requests at allam-2-7b; count 429 vs ok."""
    print(f"phase 5: in-flight - {n} concurrent short requests on {MODEL_SMALL}")
    tasks = [call("5-inflight", MODEL_SMALL, prompt(40), reasoning_effort=None,
                  max_tokens=40) for _ in range(n)]
    res = await asyncio.gather(*tasks)
    ok = [r for r in res if r["status"] == "ok"]
    limited = [r for r in res if r["status"] == "rate_limited"]
    print(f"   issued {n} concurrently: {len(ok)} ok, {len(limited)} 429")
    if limited:
        print(f"   -> in-flight limit below {n} (first 429 had retry_after="
              f"{limited[0]['retry_after']})")
    else:
        print(f"   -> at least {n}, no limit observed")
    return {"issued": n, "ok": len(ok), "limited": len(limited),
            "peak_ok": len(ok)}


async def phase_recover() -> dict:
    """Trip one 429 on the safeguard model, wait retry_after, retry exactly once."""
    print(f"phase 6: time-to-recover on {MODEL_LIMIT} (trip, wait, retry once)")
    hit = None
    for _ in range(4):
        r = await call("6-recover", MODEL_LIMIT, prompt(40), reasoning_effort="low",
                       max_tokens=40)
        if r["status"] == "rate_limited":
            hit = r
            break
    if hit is None:
        print("   no 429 within 4 requests; recovery not measured")
        return {"recovered_s": None, "retry_after": None}
    wait = float(hit.get("retry_after") or 20)
    print(f"   got 429, retry_after={wait}s -> sleeping")
    t_hit = time.monotonic()
    await asyncio.sleep(wait)
    r = await call("6-recover", MODEL_LIMIT, prompt(40), reasoning_effort="low",
                   max_tokens=40)
    if r["status"] == "ok":
        dt = time.monotonic() - t_hit
        print(f"   recovered after {dt:.1f}s")
        return {"recovered_s": dt, "retry_after": wait}
    print(f"   retry still {r['status']}; not recovered (no further retry)")
    return {"recovered_s": None, "retry_after": wait}


def pct(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = max(0, min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1)))))
    return vals[k]


def rolling_token_ceiling(records: list[dict], window_s: float = 60.0) -> float | None:
    """Largest (tokens_in + tokens_out) that completed inside any rolling window."""
    done = sorted((r["t_monotonic_end"], (r["tokens_in"] or 0) + (r["tokens_out"] or 0))
                  for r in records if r.get("t_monotonic_end"))
    best = 0
    for i, (t0, _) in enumerate(done):
        total = sum(tok for t, tok in done[i:] if t - t0 <= window_s)
        best = max(best, total)
    return float(best)


def report_models() -> None:
    print("\n1) per-model timing and usage")
    print(f"{'model':32s} {'n':>3s} {'ttft_p50':>9s} {'ttft_p95':>9s} {'ttft_max':>9s} "
          f"{'tot_p50':>9s} {'tot_p95':>9s} {'tot_max':>9s} {'in':>5s} {'out':>5s} "
          f"{'reason':>7s} {'est%':>5s}")
    models = sorted({r["model"] for r in RECORDS})
    for m in models:
        rs = [r for r in RECORDS if r["model"] == m]
        ok = [r for r in rs if r["status"] == "ok"]
        ttft = [r["first_token_ms"] for r in ok if r["first_token_ms"] is not None]
        tot = [r["total_ms"] for r in ok]
        tin = sum(r["tokens_in"] or 0 for r in ok)
        tout = sum(r["tokens_out"] or 0 for r in ok)
        reas = sum(r["reasoning_tokens"] or 0 for r in ok)
        est = 100 * sum(1 for r in ok if r["tokens_estimated"]) / len(ok) if ok else 0
        f = lambda v: f"{v:.0f}" if v is not None else "-"
        print(f"{m:32s} {len(rs):3d} {f(pct(ttft,50)):>9s} {f(pct(ttft,95)):>9s} "
              f"{f(max(ttft) if ttft else None):>9s} {f(pct(tot,50)):>9s} "
              f"{f(pct(tot,95)):>9s} {f(max(tot) if tot else None):>9s} {tin:5d} "
              f"{tout:5d} {reas:7d} {est:5.0f}")


def report_timeline() -> None:
    print("\n2) rate-limit buckets over time (t=s since first request)")
    if not RECORDS:
        print("   no requests")
        return
    t0 = RECORDS[0]["t_monotonic"]
    print(f"   {'t':>7s} {'phase':10s} {'status':12s} {'tpm_remaining':>14s} "
          f"{'rpd_remaining':>14s} {'reset_tokens':>13s}")
    for r in RECORDS:
        h = r.get("ratelimit_headers") or {}
        print(f"   {r['t_monotonic']-t0:7.1f} {r['phase']:10s} {r['status']:12s} "
              f"{h.get('tpm_remaining','-'):>14s} {h.get('rpd_remaining','-'):>14s} "
              f"{h.get('tpm_reset','-'):>13s}")


def report_phase2() -> None:
    print("\n3) phase 2 (reasoning_effort omitted) vs phase 1 short (effort=low)")
    p1 = [r for r in RECORDS if r["phase"] == "1-short" and r["status"] == "ok"]
    p2 = [r for r in RECORDS if r["phase"] == "2-default" and r["status"] == "ok"]
    def mean(xs, key):
        vals = [x[key] for x in xs if x.get(key) is not None]
        return statistics.mean(vals) if vals else None
    rows = [("reasoning_tokens", mean(p1, "reasoning_tokens"), mean(p2, "reasoning_tokens")),
            ("tokens_out", mean(p1, "tokens_out"), mean(p2, "tokens_out")),
            ("first_token_ms", mean(p1, "first_token_ms"), mean(p2, "first_token_ms")),
            ("total_ms", mean(p1, "total_ms"), mean(p2, "total_ms"))]
    print(f"   {'metric':18s} {'low':>10s} {'default':>10s} {'delta':>10s} {'delta%':>8s}")
    for name, a, b in rows:
        if a is None or b is None:
            print(f"   {name:18s} {'-':>10s} {'-':>10s}")
            continue
        d = b - a
        print(f"   {name:18s} {a:10.1f} {b:10.1f} {d:+10.1f} "
              f"{(100*d/a if a else 0):+7.0f}%")


def report_limit() -> None:
    print("\n4) rate-limit behaviour (phase 4, gpt-oss-safeguard-20b)")
    p4 = [r for r in RECORDS if r["phase"] == "4-limit"]
    if not p4:
        print("   not run")
        return
    hit = None
    for r in p4:
        print(f"   req {r['n_request']}: {r['status']} http={r['http_status']} "
              f"retry_after={r['retry_after']}")
        if r["status"] == "rate_limited" and hit is None:
            hit = r
    if hit:
        print("\n   body verbatim:")
        print("   " + (hit["error_body"] or "").replace("\n", "\n   "))
        print(f"\n   retry-after header: {hit['retry_after']}")
    after = [r for r in p4 if hit and r["status"] == "ok"
             and r["t_monotonic"] > hit["t_monotonic"]]
    if hit and after:
        print(f"   recovered after {after[0]['t_monotonic'] - hit['t_monotonic']:.1f}s "
              f"(wall clock, measured)")
    elif hit:
        print("   NOT recovered: the phase-4 budget ran out while still limited "
              f"(retry_after said {hit['retry_after']}s)")


def report_table1() -> None:
    print("\n5) Table 1 draft - per-model limits and what the API does at the limit")
    print(f"   {'property':22s} {'value':26s} {'source':14s} {'confidence':10s}")
    hdrs: dict[str, str] = {}
    for r in RECORDS:  # last response that carried headers (main model preferred)
        h = r.get("ratelimit_headers") or {}
        if h:
            hdrs = h
            if r["model"] == MODEL_MAIN:
                break
    tpm_limit = hdrs.get("tpm_limit", "not measured")
    rpd_limit = hdrs.get("rpd_limit", "not measured")
    ok = [r for r in RECORDS if r["status"] == "ok"]
    tpm_ceiling = rolling_token_ceiling(ok, 60.0)
    p4 = [r for r in RECORDS if r["phase"] in ("4-limit", "6-recover")]
    hit = next((r for r in RECORDS if r["status"] == "rate_limited"), None)
    rec = None
    if hit:  # a success after the 429, whichever phase produced it
        okp4 = [r for r in RECORDS if r["status"] == "ok"
                and r["t_monotonic"] > hit["t_monotonic"]]
        rec = f"{okp4[0]['t_monotonic'] - hit['t_monotonic']:.1f}s" if okp4 else None
    inflight = next((r.get("inflight") for r in RECORDS if r.get("inflight")), {})
    inflight_cell = ("not measured" if not inflight else
                     (f"{inflight['ok']}/{inflight['issued']} ok (limit < {inflight['issued']})"
                      if inflight["limited"] else
                      f"at least {inflight['issued']}, no limit observed"))
    recover = next((r["recovered_s"] for r in RECORDS
                    if r.get("recovered_s") is not None), None)
    if recover is None and rec:
        recover = float(rec.rstrip("s"))
    rows = [
        ("requests/min", "30", "console (user-provided)", "medium"),
        ("tokens/min", tpm_limit if tpm_limit not in (None, "-") else "not measured",
         "header (x-ratelimit-limit-tokens)", "high"),
        ("requests/day", rpd_limit if rpd_limit not in (None, "-") else "not measured",
         "header (x-ratelimit-limit-requests)", "high"),
        ("tokens/day", "200000", "console (user-provided)", "high"),
        ("in-flight", inflight_cell, "measured (phase 5, 8 concurrent)", "high" if inflight else "low"),
        ("on-limit response", (f"{hit['http_status']} {hit['status']}" if hit else "not measured"),
         "measured (phase 4/6)", "high" if hit else "low"),
        ("time-to-recover", f"{recover:.1f}" if recover else "not measured",
         "measured (phase 6 wall clock)", "high" if recover else "low"),
        ("measured_tpm_ceiling", f"{tpm_ceiling:.0f}" if tpm_ceiling else "not measured",
         "measured (max rolling 60s window)", "medium"),
    ]
    for name, val, src, conf in rows:
        print(f"   {name:22s} {val:26s} {src:14s} {conf:10s}")
    if hit:
        print("   on-limit body (verbatim):")
        print("     " + (hit.get("error_body") or "").replace("\n", "\n     "))
    print("   note: no rpm field exists - requests are bucketed per DAY, tokens per MINUTE.")
    if tpm_ceiling and tpm_limit not in (None, "not measured"):
        try:
            over = tpm_ceiling > float(str(tpm_limit).rstrip("s"))
        except ValueError:
            over = False
        why = ("the rolling window is global wall-clock and spans model switches plus bucket "
               "resets, while the header is a per-model sliding minute bucket - so the ceiling "
               "can sit above it" if over else
               "the probe never saturates the minute bucket, so measured < header")
        print(f"   note: measured_tpm_ceiling ({tpm_ceiling:.0f} tok in the busiest 60s) vs "
              f"{tpm_limit} header: {why}.")


async def main() -> None:
    ap = argparse.ArgumentParser(description="Groq latency / rate-limit probe (cap 25)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, no network")
    ap.add_argument("--measure-missing", action="store_true",
                    help="only measure in-flight + time-to-recover (<=15 requests) and print Table 1")
    args = ap.parse_args()

    plan = [("1-short", MODEL_MAIN, 6), ("1-medium", MODEL_MAIN, 6),
            ("2-default", MODEL_MAIN, 3), ("3-small", MODEL_SMALL, 3),
            ("4-limit", MODEL_LIMIT, 5)]
    if args.dry_run:
        print(f"plan: {sum(n for _, _, n in plan)} requests (cap {CAP})")
        for ph, m, n in plan:
            print(f"  {ph:10s} {m:32s} x{n}")
        return
    if args.measure_missing:
        if not os.environ.get("GROQ_API_KEY"):
            sys.exit("GROQ_API_KEY is not set - no requests sent")
        LOG.touch()
        inflight = await phase_inflight(8)
        recover = await phase_recover()
        rec = {"phase": "5/6-summary", "model": "measurement", "status": "summary",
               "t_monotonic": time.monotonic(), "inflight": inflight,
               "recovered_s": recover["recovered_s"],
               "retry_after": recover["retry_after"]}
        RECORDS.append(rec)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nused {COUNT['n']} requests (cap {CAP})")
        report_table1()
        return

    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set - this probe needs a real key (no requests sent)")

    if LOG.exists():
        LOG.unlink()
    t_start = time.monotonic()
    await phase1()
    await phase2()
    await phase3()
    await phase4()
    print(f"\nused {COUNT['n']} of {CAP} requests in {time.monotonic()-t_start:.1f}s; "
          f"log: {LOG}")

    report_models()
    report_timeline()
    report_phase2()
    report_limit()
    report_table1()


if __name__ == "__main__":
    asyncio.run(main())
