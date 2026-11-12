#!/usr/bin/env python3
"""Stand-in model service: OpenAI-compatible, SSE-streaming, calibrated on real logs.

    python scripts/standin/server.py --port 8787 [--limits] [--fail-rate 0.05]
                                     [--slow-success-rate 0.05] [--load-slowdown]
                                     [--eof]

Point the pipeline at it with LLM_BASE_URL=http://127.0.0.1:8787/openai/v1
(GROQ_API_KEY can be any non-empty string).

Calibration comes from scripts/standin/calibration.json, fitted by fit.py from
ttft.jsonl/probe.jsonl: TTFT is a two-mode mixture, inter-token gap and completion
length are sampled from measured quantiles. Nothing is hardcoded from taste.

Every request is logged to standin.log as one JSON line.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

from aiohttp import web

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CALIB = json.loads((HERE / "calibration.json").read_text())
LIMITS_CFG = json.loads((HERE / "limits.json").read_text())
LOG = ROOT / "standin.log"

# cl100k_base + a per-model constant: the real prompts measure a fixed ~70-token
# chat-template overhead above the tokenizer count (see fit.py residuals), so the
# raw tokenizer under-counts short prompts by up to 60%.
_ENC = None
_T0 = time.monotonic()
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception as exc:  # pragma: no cover
    print(f"[tokenizer] tiktoken unavailable ({exc}); falling back to chars/4", flush=True)

OVERHEAD = CALIB.get("tokenizer", {}).get("template_overhead_tokens", {})


def overhead_for(model: str) -> float:
    short = model.split("/")[-1]
    return float(OVERHEAD.get(short, OVERHEAD.get(model, 70.0)))


def count_tokens(text: str, model: str = "") -> int:
    """Token charge for the rate limiter: tokenizer count + template overhead."""
    if _ENC is None:
        return max(1, int(round(len(text) / 4.0)))
    return max(1, len(_ENC.encode(text)) + int(overhead_for(model)))


BANDS = [("<400", 0, 400), ("400-900", 400, 900), ("900-1600", 900, 1600),
         (">1600", 1600, 10 ** 9)]
RNG = random.Random(int(CALIB.get("seed", 1337)))
BAND_SAMPLES = CALIB["ttft_bands"]


def band_of(tokens: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= tokens < hi:
            return name
    return ">1600"


def sample_ttft_ms(tokens_in: float) -> float:
    """Bootstrap with replacement from this band's observations.

    A sparse band (n < threshold) is pooled with its neighbours: a 6-sample band
    cannot otherwise produce the slow tail the real service showed.
    """
    order = [b[0] for b in BANDS]
    name = band_of(tokens_in)
    idx = order.index(name)
    pool = list(BAND_SAMPLES.get(name, {}).get("samples_ms") or [])
    threshold = int(CALIB.get("sparse_band_threshold", 8))
    if len(pool) < threshold:  # too few draws: bootstrap the whole observed set
        pool = [s for b in order for s in (BAND_SAMPLES.get(b, {}).get("samples_ms") or [])]
    return float(RNG.choice(pool)) if pool else 800.0


def sample_tokens_out(max_tokens: int, reasoning_tokens: int) -> int:
    """Bootstrap an observed completion length; no interpolated quantile curve.

    Interpolating a p10..p90 range produced a 152-token mean against the real
    drafts' 123, which inflated tokens/sec by 23%.
    """
    key = "tokens_out_short" if max_tokens <= 64 else "tokens_out_draft"
    d = CALIB.get(key) or CALIB["tokens_out"]
    pool = d.get("samples") or [d["p50"]]
    return max(1, min(max_tokens, RNG.choice(pool) + reasoning_tokens))


def limits_for(model: str) -> dict:
    return LIMITS_CFG["models"].get(model, LIMITS_CFG["default"])


class Windows:
    """Sliding per-minute/day buckets, seeded from request bodies, not from us."""

    def __init__(self) -> None:
        self.per_model: dict[str, dict] = {}

    def _m(self, model: str) -> dict:
        return self.per_model.setdefault(model, {"min": [], "day": [], "inflight": 0})

    def check(self, model: str, tokens: int) -> tuple[str, float] | None:
        cfg = limits_for(model)
        st = self._m(model)
        now = time.monotonic()
        st["min"] = [e for e in st["min"] if now - e[0] < 60]
        st["day"] = [e for e in st["day"] if now - e[0] < 86400]
        rpm_used = len(st["min"])
        tpm_used = sum(e[1] for e in st["min"])
        rpd_used = len(st["day"])
        tpd_used = sum(e[1] for e in st["day"])
        for kind, used, limit in (("requests", rpm_used, cfg["rpm"]),
                                  ("tokens", tpm_used, cfg["tpm"]),
                                  ("requests", rpd_used, cfg["rpd"]),
                                  ("tokens", tpd_used, cfg["tpd"])):
            if used + (1 if kind == "requests" else tokens) > limit:
                window = "per minute" if (kind, used, limit) in (
                    ("requests", rpm_used, cfg["rpm"]), ("tokens", tpm_used, cfg["tpm"])) else "per day"
                reset = 60 - (now - st["min"][0][0]) if st["min"] and window == "per minute" else 60
                return (f"{kind} {window}", max(1.0, round(reset)))
        return None

    def record(self, model: str, tokens: int) -> None:
        st = self._m(model)
        now = time.monotonic()
        st["min"].append((now, tokens))
        st["day"].append((now, tokens))

    def headers(self, model: str) -> dict[str, str]:
        cfg = limits_for(model)
        st = self._m(model)
        now = time.monotonic()
        spent_min = sum(e[1] for e in st["min"] if now - e[0] < 60)
        spent_day = len([e for e in st["day"] if now - e[0] < 86400])
        reset_min = 60 - (now - st["min"][-1][0]) if st["min"] else 60
        return {
            "x-ratelimit-limit-requests": str(cfg["rpd"]),
            "x-ratelimit-remaining-requests": str(max(0, cfg["rpd"] - spent_day)),
            "x-ratelimit-reset-requests": f"{max(1, int(reset_min))}s",
            "x-ratelimit-limit-tokens": str(cfg["tpm"]),
            "x-ratelimit-remaining-tokens": str(max(0, cfg["tpm"] - spent_min)),
            "x-ratelimit-reset-tokens": f"{reset_min:.3f}s",
        }


WINDOWS = Windows()
INFLIGHT = {"n": 0}


def rate_limit_body(model: str, kind: str, used: int, limit: int, retry_after: int) -> dict:
    """Same shape and wording as the real 429 body (client parses this)."""
    unit = "requests per minute (RPM)" if kind.startswith("requests per minute") else (
        "tokens per minute (TPM)" if kind.startswith("tokens per minute") else
        "requests per day (RPD)" if kind.startswith("requests") else "tokens per day (TPD)")
    return {"error": {
        "message": (f"Rate limit reached for model `{model}` in organization "
                    f"`{LIMITS_CFG['org']}` service tier `{LIMITS_CFG['service_tier']}` "
                    f"on {unit}: Limit {limit}, Used {used}, Requested 1. "
                    f"Please try again in {retry_after}s. Need more tokens? Upgrade to "
                    f"Dev Tier today at https://console.groq.com/settings/billing"),
        "type": "requests" if "requests" in kind else "tokens",
        "code": "rate_limit_exceeded"}}


def log_line(**kw) -> None:
    kw["ts"] = time.time()
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kw, ensure_ascii=False) + "\n")


def make_text(prompt: str, n_tokens: int, reasoning_tokens: int) -> str:
    """Extractive-ish filler: echoes the first passage so citations can resolve."""
    head = " ".join(prompt.split())
    for marker in ("Passages:", "Question:"):
        if marker in head:
            head = head.split(marker, 1)[1]
            break
    head = head.strip()
    seed = head[:400] if head else "The passages do not contain this information."
    words = seed.split()
    need = max(1, n_tokens * 4 // 5)
    out = []
    while len(out) < need:
        out.extend(words or ["no", "content"])
    return " ".join(out[:need])


async def chat_completions(request: web.Request) -> web.StreamResponse:
    t0 = time.monotonic()
    body = await request.json()
    model = body.get("model", "unknown")
    prompt = "\n".join(str(m.get("content", "")) for m in body.get("messages") or [])
    max_tokens = int(body.get("max_tokens") or 320)
    reasoning_effort = body.get("reasoning_effort")
    stream = bool(body.get("stream"))
    prompt_tokens = count_tokens(prompt, model)

    def headers() -> dict[str, str]:
        return WINDOWS.headers(model)

    # --- limits -----------------------------------------------------------
    if ARGS.limits:
        blocked = WINDOWS.check(model, prompt_tokens)
        if blocked:
            kind, retry_after = blocked
            cfg = limits_for(model)
            used = len([e for e in WINDOWS._m(model)["min"] if time.monotonic() - e[0] < 60])
            limit = cfg["rpm"] if kind.startswith("requests per minute") else (
                cfg["tpm"] if kind.startswith("tokens per minute") else (
                    cfg["rpd"] if kind.startswith("requests") else cfg["tpd"]))
            payload = rate_limit_body(model, kind, used, limit, int(retry_after))
            log_line(model=model, status="rate_limited", kind=kind, retry_after=int(retry_after),
                     tokens_in=prompt_tokens, inflight=INFLIGHT["n"])
            return web.json_response(payload, status=429,
                                     headers={**headers(), "retry-after": str(int(retry_after))})
        WINDOWS.record(model, prompt_tokens)

    # --- injected failures ------------------------------------------------
    if random.random() < ARGS.fail_rate:
        log_line(model=model, status="error", injected="fail_rate",
                 tokens_in=prompt_tokens, inflight=INFLIGHT["n"])
        return web.json_response({"error": {"message": "internal server error",
                                            "type": "server_error", "code": "internal_error"}},
                                 status=500, headers=headers())

    # --- reasoning + length budget (the measured eof failure) --------------
    reasoning_tokens = 0
    if reasoning_effort:
        reasoning_tokens = random.choice([4, 5, 5, 6, 7]) if max_tokens > 60 else (
            random.randint(max_tokens - 4, max_tokens))
    if ARGS.eof and reasoning_effort and reasoning_tokens >= max_tokens:
        finish_reason, completion_tokens, text = "length", 0, ""
    else:
        completion_tokens = sample_tokens_out(max_tokens, reasoning_tokens)
        finish_reason = "length" if completion_tokens >= max_tokens else "stop"
        text = make_text(prompt, max(0, completion_tokens - reasoning_tokens),
                         reasoning_tokens)

    # --- timing: TTFT mixture, per-token cadence, load slowdown -----------
    INFLIGHT["n"] += 1
    cap = limits_for(model)["concurrency"]
    rho = min(0.95, INFLIGHT["n"] / max(1, cap))
    slowdown = 1.0 / (1.0 - rho) if ARGS.load_slowdown and rho > 0 else 1.0
    ttft_ms = sample_ttft_ms(prompt_tokens) * slowdown
    it = CALIB["inter_token"]
    # measured per-token cadence (0.31 ms); the 128 tok/s figure is an aggregate
    # that includes TTFT, so it is not used as a per-token gap
    gap_ms = max(float(it.get("gap_ms", 0.3057)),
                 float(it.get("measured_gap_ms_p50", 0.3057))) * slowdown
    if random.random() < ARGS.slow_success_rate:
        ttft_ms += random.uniform(8000, 15000)  # success, just very late

    queue_ms = 0.053
    prompt_time = max(0.001, ttft_ms / 1000.0)
    completion_time = max(0.001, (completion_tokens * gap_ms) / 1000.0)

    usage = {
        "queue_time": queue_ms,
        "prompt_tokens": prompt_tokens,
        "prompt_time": round(prompt_time, 6),
        "completion_tokens": completion_tokens,
        "completion_time": round(completion_time, 6),
        "total_tokens": prompt_tokens + completion_tokens,
        "total_time": round(queue_ms + prompt_time + completion_time, 6),
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }

    if not stream:
        await asyncio.sleep(ttft_ms / 1000.0 + completion_time)
        INFLIGHT["n"] -= 1
        log_line(model=model, status="ok", stream=False, tokens_in=prompt_tokens,
                 tokens_out=completion_tokens, ttft_ms=round(ttft_ms, 1),
                 inflight=INFLIGHT["n"], finish_reason=finish_reason)
        return web.json_response({
            "id": f"chatcmpl-standin-{int(time.time()*1000)}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "finish_reason": finish_reason,
                         "message": {"role": "assistant", "content": text}}],
            "usage": usage}, headers=headers())

    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream", "Cache-Control": "no-cache",
        "Connection": "keep-alive", **headers()})
    await resp.prepare(request)

    def chunk(delta: dict, finish=None) -> bytes:
        payload = {"id": f"chatcmpl-standin-{int(time.time()*1000)}",
                   "object": "chat.completion.chunk", "created": int(time.time()),
                   "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                   "x_groq": {"id": "standin"}}
        return f"data: {json.dumps(payload)}\n\n".encode()

    await asyncio.sleep(ttft_ms / 1000.0)
    if reasoning_tokens:
        for _ in range(reasoning_tokens):
            await resp.write(chunk({"reasoning": "r"}))
    words = text.split()
    per = max(1, len(words) // max(1, completion_tokens - reasoning_tokens)) if text else 1
    written = 0
    for i in range(0, len(words), max(1, per)):
        await resp.write(chunk({"content": " ".join(words[i:i + per]) + " "}))
        written += 1
        await asyncio.sleep(gap_ms / 1000.0)
    await resp.write(chunk({}, finish=finish_reason))
    if (body.get("stream_options") or {}).get("include_usage"):
        await resp.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    await resp.write_eof()
    INFLIGHT["n"] -= 1
    log_line(model=model, status="ok", stream=True, tokens_in=prompt_tokens,
             tokens_out=completion_tokens, reasoning_tokens=reasoning_tokens,
             ttft_ms=round(ttft_ms, 1), completion_tokens_written=written,
             inflight=INFLIGHT["n"], finish_reason=finish_reason,
             wall_ms=round((time.monotonic() - t0) * 1000, 1))
    return resp


async def health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "inflight": INFLIGHT["n"],
                              "calibration": {"method": CALIB.get("method"),
                                              "bands": {k: v["n"] for k, v in BAND_SAMPLES.items()}}})


ARGS: argparse.Namespace


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser(description="stand-in model service")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--fail-rate", type=float, default=0.0)
    ap.add_argument("--slow-success-rate", type=float, default=0.0)
    ap.add_argument("--limits", action="store_true")
    ap.add_argument("--load-slowdown", action="store_true")
    ap.add_argument("--eof", action="store_true")
    ARGS = ap.parse_args()

    app = web.Application()
    app.router.add_post("/openai/v1/chat/completions", chat_completions)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/health", health)
    print(f"stand-in on :{ARGS.port}  flags: fail={ARGS.fail_rate} slow={ARGS.slow_success_rate} "
          f"limits={ARGS.limits} slowdown={ARGS.load_slowdown} eof={ARGS.eof}", flush=True)
    print(f"[calibration] bootstrap bands: "
          + ", ".join(f"{k}:n={v['n']}" for k, v in BAND_SAMPLES.items()), flush=True)
    print(f"[calibration] tokenizer={'cl100k_base' if _ENC else 'chars/4'} "
          f"overhead={OVERHEAD} inter-token={CALIB['inter_token']['gap_ms']}ms "
          f"({1000 / CALIB['inter_token']['gap_ms']:.0f} tok/s) "
          f"load={time.monotonic() - _T0:.1f}s seed={CALIB.get('seed')}", flush=True)
    web.run_app(app, port=ARGS.port, print=None)


if __name__ == "__main__":
    main()
