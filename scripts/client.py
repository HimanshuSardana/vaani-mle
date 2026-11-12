#!/usr/bin/env python3
"""Minimal async client for an OpenAI-compatible endpoint (Groq by default).

    from client import chat
    r = await chat("openai/gpt-oss-20b", [{"role": "user", "content": "hi"}],
                   deadline_s=60, reasoning_effort="low")

`chat` never raises for HTTP-level problems: timeouts, 429s and transport errors
are returned as a status. Everything rate-limit related is reported with its real
meaning (requests per DAY / tokens per MINUTE) - there is no "rpm" field, because
Groq does not send one.

Env: LLM_BASE_URL (default https://api.groq.com/openai/v1), GROQ_API_KEY (only).
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

# x-ratelimit-<name> -> what the bucket actually means
RATELIMIT_RENAME = {
    "limit_requests": "rpd_limit",
    "remaining_requests": "rpd_remaining",
    "reset_requests": "rpd_reset",
    "limit_tokens": "tpm_limit",
    "remaining_tokens": "tpm_remaining",
    "reset_tokens": "tpm_reset",
}
# headers we must never echo back
SECRET_HEADERS = {"authorization", "api-key", "x-api-key", "cookie", "proxy-authorization"}


def _base_url() -> str:
    return os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set (this client reads that env var only)")
    return key


def _scrub(headers) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in SECRET_HEADERS:
            continue
        if any(tok in kl for tok in ("token", "secret", "key")) and not kl.startswith("x-ratelimit"):
            continue
        out[kl] = v
    return out


def _ratelimits(headers) -> dict[str, str]:
    """Every x-ratelimit-* header, renamed to what the bucket means."""
    out: dict[str, str] = {}
    for k, v in headers.items():
        kl = k.lower()
        if not kl.startswith("x-ratelimit-"):
            continue
        name = kl[len("x-ratelimit-"):].replace("-", "_")
        out[RATELIMIT_RENAME.get(name, name)] = v
    return out


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def _empty(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "status": "error",
        "http_status": None,
        "first_token_ms": None,
        "total_ms": 0.0,
        "tokens_in": None,
        "tokens_out": None,
        "reasoning_tokens": None,
        "reasoning_text": "",
        "tokens_estimated": False,
        "text": "",
        "chunks": [],
        "retry_after": None,
        "ratelimit_headers": {},
        "retries": 0,
        "max_tokens": None,
        "finish_reason": None,
        "usage": None,
        "error_body": None,
        "headers": {},
    }


async def chat(model: str, messages: list[dict], *, deadline_s: float,
               stream: bool = True, reasoning_effort: str | None = None,
               max_tokens: int = 320) -> dict[str, Any]:
    """One completion. Returns timings, usage, text, status and rate-limit state.

    max_tokens is always sent: prose instructions like "about 40 tokens" are not
    obeyed by the model (a 40-token ask produced 307 tokens without a cap).
    Default 320 suits drafting; pass 40 for short routing-style calls.
    """
    res = _empty(model)
    res["max_tokens"] = max_tokens
    payload: dict[str, Any] = {"model": model, "messages": messages,
                               "stream": stream, "max_tokens": max_tokens}
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if reasoning_effort is not None:  # omitted entirely when None, by design
        payload["reasoning_effort"] = reasoning_effort

    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    url = f"{_base_url()}/chat/completions"
    start_ns = time.perf_counter_ns()
    usage: dict | None = None

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(deadline_s)) as cx:
            async with cx.stream("POST", url, json=payload, headers=headers) as resp:
                res["http_status"] = resp.status_code
                res["ratelimit_headers"] = _ratelimits(resp.headers)
                res["headers"] = _scrub(resp.headers)
                if resp.status_code == 429:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    res.update(status="rate_limited", error_body=body,
                               retry_after=resp.headers.get("retry-after"))
                    res["total_ms"] = (time.perf_counter_ns() - start_ns) / 1e6
                    return res
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    res.update(status="error", error_body=body,
                               retry_after=resp.headers.get("retry-after"))
                    res["total_ms"] = (time.perf_counter_ns() - start_ns) / 1e6
                    return res

                if stream:
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("usage"):
                            usage = obj["usage"]
                        for choice in obj.get("choices") or []:
                            if choice.get("finish_reason"):
                                res["finish_reason"] = choice["finish_reason"]
                            delta = choice.get("delta") or {}
                            piece = delta.get("content")
                            rpiece = delta.get("reasoning")
                            if rpiece:
                                res["reasoning_text"] += rpiece
                            if piece:
                                if res["first_token_ms"] is None:
                                    res["first_token_ms"] = (
                                        time.perf_counter_ns() - start_ns) / 1e6
                                res["text"] += piece
                                res["chunks"].append(piece)
                else:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    obj = json.loads(body)
                    usage = obj.get("usage")
                    for choice in obj.get("choices") or []:
                        if choice.get("finish_reason"):
                            res["finish_reason"] = choice["finish_reason"]
                        msg = choice.get("message") or {}
                        res["text"] += msg.get("content") or ""
                        res["reasoning_text"] += msg.get("reasoning") or ""
                    if res["text"]:
                        res["first_token_ms"] = (time.perf_counter_ns() - start_ns) / 1e6
                        res["chunks"] = [res["text"]]
                res["status"] = "ok"

    except httpx.TimeoutException as exc:
        res.update(status="timeout", error_body=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # transport / auth / bad json - never raise
        res.update(status="error", error_body=f"{type(exc).__name__}: {exc}")

    res["total_ms"] = (time.perf_counter_ns() - start_ns) / 1e6

    if res["status"] == "ok":
        if usage:
            res["usage"] = usage
            res["tokens_in"] = usage.get("prompt_tokens")
            res["tokens_out"] = usage.get("completion_tokens")
            details = usage.get("completion_tokens_details") or {}
            res["reasoning_tokens"] = details.get("reasoning_tokens")
        if res["tokens_in"] is None or res["tokens_out"] is None:
            res["tokens_in"] = res["tokens_in"] or _estimate_tokens(
                " ".join(str(m.get("content", "")) for m in messages))
            res["tokens_out"] = res["tokens_out"] or _estimate_tokens(res["text"])
            res["tokens_estimated"] = True
    return res
