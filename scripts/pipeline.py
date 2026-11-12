#!/usr/bin/env python3
"""End-to-end pipeline: route -> retrieve -> draft -> verify, with a rung ladder.

    python scripts/pipeline.py "What BLEU does the big Transformer get on German?" --setting S1
    python scripts/pipeline.py "..." --setting S1 --source standin

Constants come from measurement (see probe/ttft/rerank runs), not guesses:
  route   allam-2-7b,        max_tokens 64,  no reasoning_effort
  draft   gpt-oss-20b,       max_tokens 320, reasoning_effort="low"
  retrieve BM25 + hint + table boost -> 12 candidates -> rerank(384-char pairs) -> 6 chunks

Every request appends one line to traces/trace.jsonl. Nothing here computes its
own token totals: quota comes from x-ratelimit-remaining-tokens, usage from the
API's prompt/completion counts.
"""
from __future__ import annotations

import argparse
import asyncio
import uuid
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import chat  # noqa: E402
from retrieve import (PAPER_ALIASES, build_index, detect_hint,  # noqa: E402
                      is_appendix_like, load_chunks, load_or_build_index,
                      pair_text, wants_compare, wants_table)

CHUNKS_PATH = Path(os.environ.get("CHUNKS_FILE", "chunks.jsonl"))
# Loaded once per process: parsing chunks.jsonl and rebuilding BM25 on every
# request cost ~54 ms of the request budget for identical results.
_CORPUS: tuple[list[dict], object] | None = None

PAPER_IDS = set(PAPER_ALIASES)

ROUTE_MODEL = "allam-2-7b"
DRAFT_MODEL = "openai/gpt-oss-20b"
ROUTE_MAX_TOKENS = 64
DRAFT_MAX_TOKENS = 320
CANDIDATES = 12
KEEP_CHUNKS = 6
# Declared retrieval settings for the grid (uniform across all 15 questions).
SETTINGS = {
    "S1": {"rerank": False, "keep": 6, "note": "BM25 only"},
    "S2": {"rerank": True,  "keep": 6, "note": "BM25 + cross-encoder rerank (12 -> 6)"},
    "S3": {"rerank": True,  "keep": 8, "note": "BM25 + rerank, wider keep (12 -> 8)"},
    # Table 3 settings: piece size, pieces fetched and rerank each differ. A and B
    # read their own chunks file (CHUNKS_FILE), C is S3 on the 6000-char pieces.
    "A": {"rerank": False, "keep": 4,
          "note": "small pieces (max-chars 1200), 4 fetched, no rerank"},
    "B": {"rerank": True,  "keep": 6,
          "note": "medium pieces (max-chars 3000), 6 fetched, rerank"},
}
RERANK_CHAR_LIMIT = 384
RERANK_BUDGET_MS = 400.0
REQUEST_DEADLINE_MS = float(os.environ.get("REQUEST_DEADLINE_MS", 8000))
MIN_DEADLINE_MS = 400.0
MAX_RETRIES = 2
MAX_ROUNDS = 2
TRACE = Path("traces/trace.jsonl")

REFUSAL = "Not stated in the provided papers."
CITATION_RE = re.compile(
    r"\[(\d{4}\.\d{4,5})(?:v\d+)?\s*§\s*([^\]]{1,120}?)\s*(?:pp?\.?|page)s?\s?(\d+)\]"
    r"(?:\s*[\"\u201c]([^\"\u201d]{2,300})[\"\u201d])?",
    re.IGNORECASE)
DASHES = {"\u2010": "-", "\u2011": "-", "\u2212": "-", "\u2013": "-", "\u2014": "-"}
SUPERSCRIPTS = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁻': '-', '⁺': '+'}
MATH_OPS = "+\\-=<>\u00b7\u00d7/,"

SHORT_CITE_RE = re.compile(
    r"\[\s*(\d{1,2})\s*§\s*([^\]]{1,120}?)\s*(?:pp?\.?|page)s?\s?(\d+)\s*\]",
    re.IGNORECASE)
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
# numbered-passage form the draft is asked to produce: [3] "verbatim quote"
PASSAGE_CITE_RE = re.compile(
    r"\[\s*(\d{1,2})\s*\](?:\s*[\"\u201c]([^\"\u201d]{2,300})[\"\u201d])?")
LIST_MARKER = re.compile(r"^\s*\d{1,3}[.)]\s+", re.M)
BRACKET_CITE = re.compile(r"\[[^\]]*§[^\]]*\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
REFERENCE_RE = re.compile(
    r"\b\d{4}\.\d{4,5}(?:v\d+)?\b"
    r"|(?:section|sec\.|\u00a7|table|figure|fig\.|eq\.|equation|appendix)\s*"
    r"\d+(?:\.\d+)*", re.IGNORECASE)

RERANK_TIMES: list[float] = []
RERANK_DISABLED = False
_RERANKER = None
RERANK_STARTUP_MS: float | None = None


def get_reranker():
    """Model load is ~3.7 s; it must not count against REQUEST_DEADLINE_MS."""
    global _RERANKER
    if _RERANKER is None:
        try:
            from sentence_transformers import CrossEncoder
            _RERANKER = CrossEncoder("cross-encoder/ms-marco-minilm-l-6-v2")
            _RERANKER.predict([("warmup", "warmup passage")])
        except Exception as exc:
            log(f"[rerank] unavailable ({exc}); BM25 only")
            _RERANKER = False
    return _RERANKER or None


def decide_rerank_at_startup(chunks: list[dict], question: str = "startup probe") -> bool:
    """Time 12 truncated pairs once, then fix the decision for the whole process."""
    global RERANK_DISABLED, RERANK_STARTUP_MS
    model = get_reranker()
    if model is None:
        RERANK_DISABLED = True
        log("[rerank] startup: no cross-encoder available -> BM25 order")
        return RERANK_DISABLED
    pairs = [(question, pair_text(c, RERANK_CHAR_LIMIT)) for c in chunks[:CANDIDATES]]
    t0 = now_ms()
    model.predict(pairs)
    RERANK_STARTUP_MS = now_ms() - t0
    RERANK_DISABLED = RERANK_STARTUP_MS > RERANK_BUDGET_MS
    log(f"[rerank] startup: {len(pairs)} truncated pairs in {RERANK_STARTUP_MS:.0f}ms "
        f"(budget {RERANK_BUDGET_MS:.0f}ms) -> "
        f"{'BM25 order (rerank disabled for all requests)' if RERANK_DISABLED else 'rerank enabled'}")
    return RERANK_DISABLED


def norm_quote(text: str) -> str:
    """Quote normalisation: maths-friendly, digits exact.

    Collapses whitespace around + - = < > · × / and commas, unifies minus/dash/
    en-dash, strips "$" and LaTeX braces, folds unicode superscripts to ASCII, and
    maps \\cdot|\\times to one character. Digits are never altered, so a changed
    digit still fails the substring check.
    """
    t = text.lower().replace("$", "").replace("{", "").replace("}", "").replace("^", "")
    for bad, good in DASHES.items():
        t = t.replace(bad, good)
    t = "".join(SUPERSCRIPTS.get(ch, ch) for ch in t)
    t = re.sub(r"\\cdot|\\times|\u00b7|\u00d7|\u22c5", "\u00b7", t)
    t = re.sub(r"\s*([" + re.escape(MATH_OPS) + r"])\s*", r"\1", t)
    return " ".join(t.split())


def now_ms() -> float:
    return time.perf_counter_ns() / 1e6


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# quota: per-model buckets from response headers + local rpm counter
# --------------------------------------------------------------------------- #
STATE_FILE = Path(".state/buckets.json")


class Buckets:
    """Last-known bucket state, persisted so a cold start still knows the quota."""

    def __init__(self) -> None:
        self.state: dict[str, dict] = {}
        self.calls: dict[str, list[float]] = {}
        if STATE_FILE.exists():
            try:
                self.state = json.loads(STATE_FILE.read_text())
            except (ValueError, OSError):
                self.state = {}

    def _save(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.state, indent=1))

    def observe(self, model: str, res: dict) -> None:
        h = res.get("ratelimit_headers") or {}
        def num(key):
            v = h.get(key)
            try:
                return float(str(v).rstrip("s").split("m")[-1])
            except (TypeError, ValueError):
                return None
        self.state[model] = {
            "tpm_remaining": num("tpm_remaining"),
            "tpm_limit": num("tpm_limit"),
            "rpd_remaining": num("rpd_remaining"),
            "tpm_reset": h.get("tpm_reset"),
            "rpd_reset": h.get("rpd_reset"),
            "observed_at": time.monotonic(),
            "observed_at_wall": time.time(),
        }
        self.calls.setdefault(model, []).append(time.monotonic())
        self._save()

    def rpm(self, model: str) -> int:
        cutoff = time.monotonic() - 60
        self.calls[model] = [t for t in self.calls.get(model, []) if t >= cutoff]
        return len(self.calls.get(model, []))

    def pressure(self, model: str) -> bool:
        s = self.state.get(model) or {}
        tpm, rpd = s.get("tpm_remaining"), s.get("rpd_remaining")
        return bool((tpm is not None and tpm < 1500) or (rpd is not None and rpd < 3))

    def snapshot(self, model: str) -> float | None:
        """Remaining TPM, refilled linearly by elapsed time (the bucket is per minute).

        Without the refill the ladder keeps believing the bucket is empty long after
        the minute window has rolled, and every request degrades to extractive.
        """
        s = self.state.get(model) or {}
        if s.get("tpm_remaining") is None:
            return None
        age = max(0.0, time.time() - s.get("observed_at_wall", 0))
        limit = s.get("tpm_limit")
        remaining = s["tpm_remaining"]
        if limit:
            remaining = min(limit, remaining + (age / 60.0) * limit)
        if age > 180:
            return None  # too stale to trust
        return remaining


BUCKETS = Buckets()


async def call_with_retries(model, messages, *, max_tokens, reasoning_effort,
                            deadline_ns, buckets, trace) -> dict:
    """Retries only on 429/timeout/5xx, exponential backoff with full jitter."""
    attempt = 0
    while True:
        if (deadline_ns - time.perf_counter_ns()) / 1e6 < MIN_DEADLINE_MS:
            return {"status": "timeout", "error_body": "deadline exhausted before call",
                    "http_status": None, "first_token_ms": None, "total_ms": 0.0,
                    "tokens_in": None, "tokens_out": None, "reasoning_tokens": None,
                    "text": "", "finish_reason": None, "ratelimit_headers": {}}
        remaining_ms = (deadline_ns - time.perf_counter_ns()) / 1e6
        res = await chat(model, messages, deadline_s=max(0.5, remaining_ms / 1000),
                         stream=True, reasoning_effort=reasoning_effort,
                         max_tokens=max_tokens)
        buckets.observe(model, res)
        retryable = (res["status"] in ("rate_limited", "timeout")
                     or (res["http_status"] or 0) >= 500)
        if not retryable or attempt >= MAX_RETRIES:
            return res
        attempt += 1
        trace["retries"] += 1
        hinted = 0.0
        try:
            hinted = float(res.get("retry_after") or 0)
        except (TypeError, ValueError):
            hinted = 0.0
        # Groq's retry-after header can name a longer wait than the same 429's body
        # does, and it is the body that says which window ran out. Read both.
        body = res.get("error_body") or ""
        if "tokens per day" in body:
            trace["ratelimit_window"] = "tpd"
        elif "tokens per minute" in body:
            trace["ratelimit_window"] = "tpm"
        elif "requests per" in body:
            trace["ratelimit_window"] = "rpm"
        body_hint = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", body)
        if body_hint:
            body_s = float(body_hint.group(1) or 0) * 60 + float(body_hint.group(2))
            hinted = min(hinted, body_s) if hinted > 0 else body_s
        if hinted > 0:
            delay = hinted + random.uniform(0.0, 0.5)  # obey the server's own number
        else:
            delay = random.uniform(0, min(4.0, 0.4 * 2 ** attempt))
        # A retry-after longer than the time left cannot help: sleeping through it
        # misses the request deadline, so the request fails later than it had to.
        # Measured: a 429 with retry-after 868s slept 14.5 min and then timed out.
        left_ms = (deadline_ns - time.perf_counter_ns()) / 1e6
        if delay * 1000 > left_ms - MIN_DEADLINE_MS:
            trace["retry_skipped"] = (f"retry-after {delay:.0f}s exceeds "
                                     f"{left_ms:.0f}ms left on the deadline")
            log(f"[retry] {model} {res['status']} wait {delay:.0f}s > "
                f"{left_ms:.0f}ms left; returning now instead of sleeping")
            return res
        log(f"[retry] {model} {res['status']} attempt {attempt} backoff {delay:.2f}s")
        await asyncio.sleep(delay)


# --------------------------------------------------------------------------- #
# step 1: route
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# step 2: BM25 (no model)
# --------------------------------------------------------------------------- #
def bm25_candidates(question: str, chunks: list[dict],
                    index=None) -> tuple[list[dict], float]:
    t0 = now_ms()
    bm25 = index if index is not None else build_index(chunks)
    scores = bm25.get_scores(re.findall(r"[a-z0-9]+", question.lower()))
    scored = [dict(c, score=float(s)) for c, s in zip(chunks, scores)]
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored, now_ms() - t0


def corpus() -> tuple[list[dict], object]:
    """Chunks and BM25 index for this process, from the persisted index when valid."""
    global _CORPUS
    if _CORPUS is None:
        chunks = load_chunks(CHUNKS_PATH)
        cache = Path(".state") / f"bm25_{CHUNKS_PATH.stem}.pkl"
        _CORPUS = (chunks, load_or_build_index(CHUNKS_PATH, chunks, cache))
    return _CORPUS


def apply_boosts(scored: list[dict], hint: list[str], table_intent: bool,
                 use_hint: bool) -> list[dict]:
    out = []
    for c in scored:
        s = c["score"]
        if use_hint and hint and c["paper"] in hint:
            s *= 1.5
        if table_intent and c["is_table"]:
            s *= 1.5
        out.append(dict(c, score=s))
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def rerank(question: str, cands: list[dict], model, deadline_ns: int,
           trace: dict, force: bool = False) -> tuple[list[dict], float]:
    """Cross-encoder over truncated pairs, unless the setting or startup says no."""
    global RERANK_DISABLED
    head = cands[:CANDIDATES]
    if RERANK_DISABLED and not force:
        trace["rerank_path"] = "bm25_fallback"
        trace["rerank_disabled"] = True
        if RERANK_TIMES:
            trace["rerank_p50_ms"] = round(sorted(RERANK_TIMES)[len(RERANK_TIMES) // 2], 2)
        return head, 0.0
    t0 = now_ms()
    if (deadline_ns - time.perf_counter_ns()) / 1e6 < MIN_DEADLINE_MS:
        trace["rerank_path"] = "skipped_deadline"
        return head, 0.0
    try:
        scores = model.predict([(question, pair_text(c, RERANK_CHAR_LIMIT)) for c in head])
    except Exception as exc:  # never fail the request on the reranker
        log(f"[rerank] failed ({exc}); using BM25 order")
        trace["rerank_path"] = "bm25_fallback"
        return head, now_ms() - t0
    rerank_ms = now_ms() - t0
    RERANK_TIMES.append(rerank_ms)
    trace["rerank_pairs"] = len(head)
    trace["rerank_truncated"] = RERANK_CHAR_LIMIT
    trace["rerank_p50_ms"] = round(RERANK_STARTUP_MS or rerank_ms, 2)
    for c, s in zip(head, scores):
        c["rerank_score"] = float(s)
    trace["rerank_path"] = "rerank"
    return sorted(head, key=lambda c: c["rerank_score"], reverse=True), rerank_ms


# --------------------------------------------------------------------------- #
# step 4: draft
# --------------------------------------------------------------------------- #
DRAFT_PROMPT = """Answer the question using ONLY the passages below.

Rules:
- Plain prose, 2-4 sentences. No JSON, no bullet lists, no quotation marks.
- After every sentence put the passage number in square brackets, like [2].
- If a passage contains the specific fact asked for, state it; refuse only when no
  passage addresses the question at all.
- If the passages really do not answer the question, reply exactly "{refusal}" and
  then one line saying why.

Passages:
{passages}

Question: {question}
"""


def build_passages(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        head = f"[{i}] {c['paper']} §{c['section_path']} p{c['page_start']}"
        body = (render_table_passage(c) if c.get("is_table")
                else " ".join(c["text"].split())[:1000])
        parts.append(f"{head}\n{body}")
    return "\n\n".join(parts)


def claims_to_prose(text: str) -> str:
    """Join the model's bullet list into prose; citations stay inline."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    claims = [re.sub(r"^(?:[-*\u2022]|\d+[.)])\s*", "", ln) for ln in lines]
    return " ".join(c for c in claims if c)


STOPWORDS = {"the", "a", "an", "of", "to", "is", "are", "was", "were", "does", "did",
             "what", "which", "how", "much", "many", "and", "or", "in", "on", "for",
             "with", "that", "this", "it", "its", "be", "as", "at", "by", "from",
             "when", "true", "used", "use", "during", "report", "reports", "get",
             "gets", "have", "has", "可以"}  # noqa: RUF001 - stopword list only
REFUSAL_RETRY_INSTRUCTION = (
    "\n\nThe answer is present in the passages. Extract it verbatim with citations.")
JSON_RETRY_INSTRUCTION = (
    "\n\nYour previous reply was not the required JSON. Return only the JSON object "
    "described above, no other text.")


async def draft(question: str, chunks: list[dict], deadline_ns: int, trace: dict) -> dict:
    prompt = DRAFT_PROMPT.format(refusal=REFUSAL, passages=build_passages(chunks),
                                 question=question)
    res = await call_with_retries(DRAFT_MODEL, [{"role": "user", "content": prompt}],
                                  max_tokens=DRAFT_MAX_TOKENS, reasoning_effort="low",
                                  deadline_ns=deadline_ns, buckets=BUCKETS, trace=trace)
    res["stream_chunks"] = res.pop("chunks", [])
    if res["status"] != "ok":
        return res
    raw = res["text"].strip()
    res["draft_raw"] = raw
    res["text"] = expand_passage_refs(raw, chunks)
    if REFUSAL.lower() in res["text"].lower():
        overlap = best_question_overlap(question, chunks)
        trace["refusal_retry_overlap"] = overlap
        if overlap >= 3:
            trace["refusal_retry"] = True
            log(f"[refusal_retry] draft refused, passage overlap {overlap} -> one re-ask")
            res2 = await call_with_retries(
                DRAFT_MODEL,
                [{"role": "user", "content": prompt + REFUSAL_RETRY_INSTRUCTION}],
                max_tokens=DRAFT_MAX_TOKENS, reasoning_effort="low",
                deadline_ns=deadline_ns, buckets=BUCKETS, trace=trace)
            res2["stream_chunks"] = res2.pop("chunks", [])
            if res2["status"] == "ok":
                for k in ("tokens_in", "tokens_out", "first_token_ms", "total_ms",
                          "finish_reason", "usage", "ratelimit_headers"):
                    res[k] = res2.get(k, res.get(k))
                raw2 = res2["text"].strip()
                res["draft_raw"] = raw2
                res["text"] = expand_passage_refs(raw2, chunks)
                trace["refusal_retry_outcome"] = (
                    "refused_again" if REFUSAL.lower() in res["text"].lower() else "answered")
            else:
                trace["refusal_retry_outcome"] = f"error:{res2['status']}"
    return res


TOKENISE = re.compile(r"[a-z0-9]+")


def parse_table(text: str) -> tuple[str, list[str], list[list[str]], list[str]]:
    """Split a table chunk into caption, header, body rows and notes."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    caption = ""
    if lines and lines[0].strip().lower().startswith("table"):
        caption = lines[0].strip()
        lines = lines[1:]
    rows = [ln for ln in lines if "\t" in ln]
    notes = [ln.strip() for ln in lines if ln.lower().startswith("note:")]
    if not rows:
        return caption, [], [], notes
    header = [c.strip() for c in rows[0].split("\t")]
    body = [[c.strip() for c in r.split("\t")] for r in rows[1:]]
    if body and len(body) > 1:
        sub = body[0]
        cells = [c for c in sub if c]
        short = all(re.fullmatch(r"[A-Za-z][A-Za-z0-9 ()/.\-]{0,23}", c) for c in cells)
        if len(cells) >= 2 and short:
            width = max(len(header), len(sub))
            header = [" ".join(x for x in (
                header[j] if j < len(header) else "",
                sub[j] if j < len(sub) else "") if x).strip() for j in range(width)]
            body = body[1:]
    return caption, header, body, notes


def render_table_passage(chunk: dict) -> str:
    """One block per row with header -> value pairs; never a raw pipe grid."""
    caption, header, body, notes = parse_table(chunk["text"])
    if not header:
        return " ".join(chunk["text"].split())[:1000]
    blocks = []
    for row in body:
        label = row[0] if row and row[0] else (row[1] if len(row) > 1 else "")
        pairs = " | ".join(f"{header[j] or 'col%d' % j}={row[j]}"
                           for j in range(min(len(header), len(row))) if row[j])
        blocks.append(f'row "{label}": {pairs}')
    head = f"{caption}\n" if caption else ""
    extra = ("\n" + "\n".join(notes)) if notes else ""
    return head + "\n".join(blocks) + extra


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if w not in STOPWORDS and len(w) > 2}


def best_question_overlap(question: str, chunks: list[dict]) -> int:
    qw = content_words(question)
    return max((len(qw & content_words(c["text"])) for c in chunks), default=0)


def auto_cite(answer: str, chunks: list[dict]) -> tuple[str, int]:
    """Attach a passage to every uncited sentence, deterministically (no model)."""
    if not chunks:
        return answer, 0
    chunk_tokens = [content_words(c["text"]) for c in chunks]
    out, added = [], 0
    for sent in re.split(r"(?<=[.!?])\s+", answer):
        if CITATION_RE.search(sent):
            out.append(sent)
            continue
        toks = content_words(sent)
        best, best_score = None, 0.0
        for i, ct in enumerate(chunk_tokens):
            if not toks or not ct:
                continue
            score = len(toks & ct) / len(toks | ct)
            if score > best_score:
                best, best_score = i, score
        if best is None:
            out.append(sent)
            continue
        c = chunks[best]
        out.append(f"{sent.rstrip()} [{c['paper']} §{c['section_path']} p{c['page_start']}]")
        added += 1
    return " ".join(out), added


def expand_passage_refs(text: str, chunks: list[dict]) -> str:
    """[n] -> [paper §section p.page], using chunks.jsonl metadata."""
    def repl(m):
        n = int(m.group(1))
        if not (1 <= n <= len(chunks)):
            return m.group(0)
        c = chunks[n - 1]
        return f"[{c['paper']} §{c['section_path']} p{c['page_start']}]"
    return PASSAGE_CITE_RE.sub(repl, text)


def norm_section(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def resolve_citation(paper: str, section: str, page: int, chunks: list[dict]) -> dict | None:
    for c in chunks:
        if c["paper"] != paper:
            continue
        if not (c["page_start"] <= page <= c["page_end"]):
            continue
        a, b = norm_section(section), norm_section(c["section_path"])
        if not a or a in b or b in a:
            return c
        a_t, b_t = set(a.split()), set(b.split())
        if a_t and len(a_t & b_t) / len(a_t) >= 0.6:
            return c
    return None


LIST_MARKER = re.compile(r"^\s*\d{1,3}[.)]\s+", re.M)
BRACKET_CITE = re.compile(r"\[[^\]]*§[^\]]*\]")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


REFERENCE_RE = re.compile(
    r"\b\d{4}\.\d{4,5}(?:v\d+)?\b"                      # paper ids
    r"|(?:section|sec\.|\u00a7|table|figure|fig\.|eq\.|equation|appendix)\s*"
    r"\d+(?:\.\d+)*", re.IGNORECASE)


def _nums(text: str) -> list[str]:
    """Numbers claimed in prose - references to sections/tables/papers excluded."""
    return [t.rstrip(".") for t in NUMBER_RE.findall(REFERENCE_RE.sub(" ", text))]


COLUMN_QUALIFIERS = (
    (("english-to-german", "english to german", "en-de", "ende"), "en-de"),
    (("english-to-french", "english to french", "en-fr", "enfr"), "en-fr"),
    (("german-to-english", "german to english", "de-en"), "de-en"),
    (("chinese",), "chinese"),
    (("code", "github"), "code"),
)


def question_qualifier(question: str | None) -> str | None:
    if not question:
        return None
    q = " ".join(question.lower().split())
    for pats, token in COLUMN_QUALIFIERS:
        if any(p in q for p in pats):
            return token
    return None


def column_check(chunk: dict, numbers: list[str], qualifier: str | None) -> str:
    """Row-level check: a table number must sit in a row naming the queried column."""
    if qualifier is None:
        return "skipped"
    if not chunk.get("is_table"):
        return "not_table"
    caption, header, body, _ = parse_table(chunk["text"])
    if not body:
        return "skipped"
    qual_norm = qualifier.replace("-", " ")
    def labelled(row: list[str]) -> str:
        return " ".join(f"{header[j] if j < len(header) else ''} {v}"
                        for j, v in enumerate(row)).lower()

    hits = [row for row in body
            if any(n.replace(",", "") in " ".join(row).replace(",", "") for n in numbers)]
    if not hits:
        return "number_absent"
    for row in hits:
        text = labelled(row).replace("-", " ")
        if qual_norm in text or qualifier in labelled(row):
            return "ok"
    return "mismatch"


def verify(answer: str, chunks: list[dict], question: str | None = None) -> dict:
    """Citation survival only: no quotes, no column checks.

    A citation must resolve to a retrieved chunk (paper + page + section).
    Unresolved citations are dropped from the text; everything else ships.
    """
    result = {"answer": answer, "failed": False, "details": [], "citations": [],
              "quote_unverified": 0, "quote_missing": 0, "unsupported_claims": 0,
              "column_check": "not_applicable"}
    answer = LIST_MARKER.sub("", answer)

    drop: list[str] = []
    for m in CITATION_RE.finditer(answer):
        rec = {"citation": m.group(0).strip(), "paper": m.group(1),
               "section": m.group(2), "page": int(m.group(3)), "quote": "",
               "chunk_id": None, "status": "unresolved"}
        ch = resolve_citation(m.group(1), m.group(2), int(m.group(3)), chunks)
        if ch is None:
            result["details"].append(f"unresolved {m.group(0)[:60]}")
            drop.append(m.group(0))
        else:
            rec.update(chunk_id=ch["chunk_id"], status="resolved", quote_missing=True)
            result["quote_missing"] += 1
        result["citations"].append(rec)

    for span in drop:
        answer = answer.replace(span, "")
    result["answer"] = re.sub(r"[ \t]{2,}", " ", answer).strip()
    result["zero_resolved"] = not any(c["chunk_id"] for c in result["citations"])
    return result


# --------------------------------------------------------------------------- #
# ladder + orchestration
# --------------------------------------------------------------------------- #
def choose_rung(deadline_ns: int) -> int:
    """0 normal -> 1 fewer chunks -> 2 no rerank -> 3 extractive -> 4 refusal."""
    remaining = (deadline_ns - time.perf_counter_ns()) / 1e6
    tpm = BUCKETS.snapshot(DRAFT_MODEL)
    tpm = float("inf") if tpm is None else tpm
    if remaining < 1200 or tpm < 300:
        return 4
    if remaining < 1800 or tpm < 700:
        return 3
    if remaining < 2600 or tpm < 1500:
        return 2
    if remaining < 4200 or tpm < 2600 or BUCKETS.rpm(DRAFT_MODEL) > 25:
        return 1
    return 0


def extractive(question: str, chunks: list[dict]) -> str:
    spans = []
    for c in chunks[:3]:
        snippet = " ".join(c["text"].split())[:240]
        spans.append(f"{snippet} [{c['paper']} §{c['section_path']} p{c['page_start']}]")
    return " \n".join(spans)


def rephrase(question: str, hint: list[str]) -> str:
    stop = {"what", "which", "how", "does", "did", "the", "and", "for", "with", "that",
            "this", "are", "was", "were", "from", "into", "about", "tell", "give", "many"}
    words = [w for w in re.findall(r"[a-zA-Z0-9.]+", question.lower()) if w not in stop]
    extra = {"1706.03762": "transformer attention bleu params",
             "2410.00037": "moshi helium learning rate table hyper-parameters",
             "2507.07955": "h-net dynamic chunking params ratio loss" 
             }.get(hint[0] if hint else "", "")
    return " ".join(sorted(set(words))) + (" " + extra if extra else "")


def outage_message(status: str, body: str = "") -> str:
    """What a waiting person is told when the failure is ours, not the papers'.

    Never uses REFUSAL here. "Not stated in the provided papers" blames the library
    for an exhausted quota, which is the wrong answer the brief warns about.
    """
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", body or "")
    wait = ""
    if m:
        secs = float(m.group(1) or 0) * 60 + float(m.group(2))
        wait = f" Try again in about {int(round(secs))} seconds."
    if "tokens per day" in (body or "") or "requests per day" in (body or ""):
        return ("The assistant has used its daily model quota and cannot answer "
                "until it resets." + wait)
    if "tokens per minute" in (body or "") or "requests per minute" in (body or ""):
        return "The assistant is over its per-minute capacity right now." + wait
    if status == "timeout":
        return "The assistant ran out of time and gave up instead of guessing."
    if status == "busy":
        return ("The assistant is at capacity right now and did not answer. "
                "Try again shortly.")
    return f"The model service failed ({status})." + wait


async def run(question: str, setting: str, source: str) -> dict:
    t_start_ns = time.perf_counter_ns()
    deadline_ns = t_start_ns + int(REQUEST_DEADLINE_MS * 1e6)
    trace: dict = {
        "ts": time.time(), "question": question, "setting": setting,
        "source": source, "status": "ok", "answer_grade": None, "retries": 0,
        "route_ms": 0.0, "bm25_ms": 0.0, "rerank_ms": 0.0, "rerank_pairs": 0,
        "rerank_truncated": None, "rerank_path": "none",
        "rerank_p50_ms": RERANK_STARTUP_MS,
        "rerank_disabled": False, "queue_ms": 0.0,
        "draft_ttft_ms": None, "draft_total_ms": None, "verify_ms": 0.0,
        "ttft_user_visible_ms": None, "rung": 0, "retrieval_rounds": 0,
        "chunk_ids": [], "max_logit": None, "verify_failed": False,
        "tokens_remaining_at_start": None, "draft_tokens_in": None,
        "draft_tokens_out": None,
        "route_hint": [], "hint_source": "detect_hint",
        "table_intent": None, "multi_paper": None,
        "answer_text": "", "finish_reason": None,
        "answer_id": uuid.uuid4().hex[:16], "citations": [], "quote_unverified": 0,
        "quote_missing": 0, "verify_checked": False, "refusal_retry": False,
        "trace_error": None,
        "error": None,
    }
    chunks_all, index = corpus()
    reranker = get_reranker() if CHUNKS_PATH.exists() else None
    trace["tokens_remaining_at_start"] = BUCKETS.snapshot(DRAFT_MODEL)

    cfg = SETTINGS.get(setting, SETTINGS["S1"])
    trace["setting_note"] = cfg["note"]
    trace["rerank_setting"] = cfg["rerank"]
    rung = choose_rung(deadline_ns)
    trace["rung"] = rung
    if rung >= 4:
        trace.update(status="refused", answer_text=outage_message("busy"), rung=4,
                     verify_checked=True, verify_skipped="system_busy")
        return trace
    if rung == 3:
        cands, bm25_ms = bm25_candidates(question, chunks_all, index)
        trace["bm25_ms"] = round(bm25_ms, 2)
        ans = extractive(question, cands)
        trace.update(answer_text=ans, status="ok", rung=3, verify_checked=True,
                     verify_skipped="extractive",
                    chunk_ids=[c["chunk_id"] for c in cands[:3]],
                    ttft_user_visible_ms=round(now_ms() - t_start_ns / 1e6, 1))
        return trace

    # step 1: retrieval (no LLM router - measured slower and less accurate than
    # the string matcher, and it never won the race against BM25 anyway)
    hint = detect_hint(question)
    table_intent = wants_table(question)
    multi_paper = wants_compare(question)
    scored, bm25_ms = bm25_candidates(question, chunks_all, index)
    trace.update(bm25_ms=round(bm25_ms, 2), route_hint=hint,
                 table_intent=table_intent, multi_paper=multi_paper)
    ranked = apply_boosts(scored, hint, table_intent, True)

    keep = 3 if rung == 1 else (cfg["keep"] if rung == 0 else KEEP_CHUNKS)
    chunks: list[dict] = []
    rounds = 0
    round_log: list[dict] = []
    trace["round_queries"] = round_log
    search_query = question  # rephrasing changes RETRIEVAL only, never the question
    while True:
        rounds += 1
        t_round = now_ms()
        if cfg["rerank"] and rung <= 1 and reranker is not None:
            ordered, rerank_ms = rerank(search_query, ranked, reranker, deadline_ns, trace,
                                        force=True)
            trace["rerank_ms"] = round(rerank_ms, 2)
        else:
            ordered, rerank_ms = ranked, 0.0
            trace["rerank_path"] = "disabled_by_rung"
        chunks = ordered[:keep]
        if trace["max_logit"] is None and chunks and "rerank_score" in chunks[0]:
            trace["max_logit"] = round(chunks[0]["rerank_score"], 4)
        round_log.append({"round": rounds, "query": search_query,
                          "ms": round(now_ms() - t_round, 1),
                          "chunks": [c["chunk_id"] for c in chunks],
                          "reason": "initial" if rounds == 1 else "no hinted-paper chunk"})
        log(f"[round {rounds}] query={search_query[:60]!r} "
            f"{round_log[-1]['ms']}ms chunks={round_log[-1]['chunks'][:3]}")
        hinted = [c for c in chunks if c["paper"] in hint] if hint else [None]
        if hinted or rounds >= MAX_ROUNDS or not hint:
            break
        log(f"[round {rounds}] no hinted-paper chunk in top {keep} -> rephrase")
        search_query = rephrase(question, hint)
        scored2, bm25_ms2 = bm25_candidates(search_query, chunks_all, index)
        ranked = apply_boosts(scored2, hint, bool(table_intent), True)
    trace["retrieval_rounds"] = rounds
    trace["chunk_ids"] = [c["chunk_id"] for c in chunks]

    # step 4
    t_draft = now_ms()
    res = await draft(question, chunks, deadline_ns, trace)
    trace.update(draft_ttft_ms=res["first_token_ms"],
                 draft_total_ms=round(res["total_ms"], 2),
                 draft_tokens_in=res.get("tokens_in"),
                 draft_tokens_out=res.get("tokens_out"),
                 finish_reason=res.get("finish_reason"))
    trace["ttft_user_visible_ms"] = round(now_ms() - t_start_ns / 1e6, 1)
    if res["status"] != "ok":
        trace.update(status=res["status"], error=res.get("error_body"),
                     answer_text=outage_message(res["status"], res.get("error_body") or ""),
                     verify_checked=True, verify_skipped="upstream_error")
        return trace
    answer = res["text"].strip()
    trace["draft_answer_raw"] = answer
    if res.get("finish_reason") == "length" and not answer:
        trace.update(status="truncated", answer_text="",
                     verify_checked=True, verify_skipped="truncated_empty")
        return trace

    # step 5 + one repair round
    t_verify = now_ms()
    vr = verify(answer, chunks, question)
    answer, failed, details = vr["answer"], vr["failed"], vr["details"]
    _adopt_verify(trace, vr)
    trace["verify_ms"] = round(now_ms() - t_verify, 2)
    if failed and rounds < MAX_ROUNDS and res["status"] == "ok":
        log(f"[verify] failed ({details[:3]}); one more retrieval round")
        rounds += 1
        trace["retrieval_rounds"] = rounds
        t_repair = now_ms()
        q2 = rephrase(question, hint)
        scored2, _ = bm25_candidates(q2, chunks_all, index)
        ranked = apply_boosts(scored2, hint, bool(table_intent), True)
        if cfg["rerank"] and rung <= 1 and reranker is not None:
            ordered, rerank_ms2 = rerank(q2, ranked, reranker, deadline_ns, trace, force=True)
            trace["rerank_ms"] = round(trace["rerank_ms"] + rerank_ms2, 2)
        else:
            ordered = ranked
        chunks = ordered[:keep]
        trace["chunk_ids"] = [c["chunk_id"] for c in chunks]
        round_log.append({"round": rounds, "query": q2, "reason": "verify repair",
                          "chunks": [c["chunk_id"] for c in chunks],
                          "ms": round(now_ms() - t_repair, 1)})
        log(f"[round {rounds}] repair query={q2[:60]!r} "
            f"{round_log[-1]['ms']}ms chunks={round_log[-1]['chunks'][:3]}")
        res = await draft(question, chunks, deadline_ns, trace)
        if res["status"] == "ok":
            vr2 = verify(res["text"].strip(), chunks, question)
            trace["repair_round"] = True
            trace["draft_answer_raw_repair"] = res["text"].strip()
            if not vr2["failed"] or len(vr2["details"]) <= len(details):
                answer, failed, details = vr2["answer"], vr2["failed"], vr2["details"]
                _adopt_verify(trace, vr2)
                trace["adopted_verify"] = "repair"
    draft_refused = REFUSAL.lower() in (answer or "").lower()
    trace["verify_failed"] = bool(failed)
    trace["refusal_source"] = "draft" if draft_refused else None
    if draft_refused:
        trace["answer_text"] = answer
        log("[refusal] from draft")
        return trace
    if vr.get("zero_resolved", True):
        # never refuse an answer we already have: attach passages deterministically
        answer, added = auto_cite(answer, chunks)
        trace["auto_cited_count"] = added
        log(f"[auto_cite] {added} sentence(s) cited deterministically (no model)")
        if added:
            vr = verify(answer, chunks, question)
            _adopt_verify(trace, vr)
            trace["verify_failed"] = bool(vr["failed"])
    trace["answer_text"] = answer
    return trace


def _adopt_verify(trace: dict, vr: dict) -> None:
    """Single place that copies a verify result into the trace (repair rounds too)."""
    trace["citations"] = vr["citations"]
    trace["quote_unverified"] = vr["quote_unverified"]
    trace["quote_missing"] = vr["quote_missing"]
    trace["unsupported_claims"] = vr["unsupported_claims"]
    trace["column_check"] = vr.get("column_check")
    trace["verify_checked"] = True


def assert_trace_complete(trace: dict) -> None:
    """Fail loudly rather than silently backfilling a trace later."""
    if not trace.get("verify_checked") and not trace.get("verify_skipped"):
        raise RuntimeError(f"trace {trace.get('answer_id')} has no verify result recorded")
    if trace["citations"] is None:
        raise RuntimeError(f"trace {trace.get('answer_id')} lost its citations")
    if (trace.get("status") == "ok" and trace.get("answer_text") and not trace["citations"]
            and not trace.get("verify_skipped")
            and REFUSAL.lower() not in (trace.get("answer_text") or "").lower()):
        raise RuntimeError(f"trace {trace.get('answer_id')} answered without any citation record")


def _rewrite_row(answer_id: str, patch: dict) -> bool:
    """Rewrite the row for this answer_id in place; False if it is not there."""
    if not TRACE.exists():
        return False
    rows = [json.loads(line) for line in TRACE.open(encoding="utf-8") if line.strip()]
    hit = False
    for r in rows:
        if r.get("answer_id") == answer_id:
            r.update(patch)
            hit = True
    if hit:
        with TRACE.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hit


def write_trace(trace: dict) -> None:
    """Write the row first, then validate; a bad row is annotated, never dropped."""
    TRACE.parent.mkdir(parents=True, exist_ok=True)
    trace["trace_error"] = None
    safe = {k: v for k, v in trace.items() if "authorization" not in k.lower()}
    with TRACE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(safe, ensure_ascii=False) + "\n")
    try:
        assert_trace_complete(trace)
    except Exception as exc:  # request time never raises
        reason = f"{type(exc).__name__}: {exc}"
        log(f"[trace] {trace.get('answer_id')} invalid -> {reason}")
        _rewrite_row(trace.get("answer_id"), {"trace_error": reason})


def validate_traces(path: Path = TRACE) -> None:
    """Report-time validation: raises if any recorded row is invalid."""
    bad = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("trace_error"):
            bad.append((r.get("answer_id"), r["trace_error"]))
        else:
            try:
                assert_trace_complete(r)
            except Exception as exc:
                bad.append((r.get("answer_id"), str(exc)))
    if bad:
        raise RuntimeError(f"{len(bad)} invalid trace(s): {bad[:3]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="route -> retrieve -> draft -> verify")
    ap.add_argument("question")
    ap.add_argument("--setting", default="S1")
    ap.add_argument("--source", choices=["groq", "standin"], default="groq")
    args = ap.parse_args()

    if args.source == "standin":
        sys.exit("--source standin is not built yet (next step); use --source groq")
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set (load .env first)")

    decide_rerank_at_startup(corpus()[0])
    trace = asyncio.run(run(args.question, args.setting, args.source))
    trace["total_ms"] = round(trace.get("ttft_user_visible_ms") or 0, 1)
    write_trace(trace)
    print(json.dumps({k: v for k, v in trace.items() if k != "question"},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
