#!/usr/bin/env python3
"""Anchor every cited quote back to a rectangle on the cited PDF page.

For each citation recorded in traces/*.jsonl, open the cited paper's PDF, restrict
to the cited chunk's page range, and try in order:

  1. exact      - page.search_for(quote)
  2. normalised - match the normalised quote (lowercase, hyphens stripped,
                  whitespace collapsed) in the page's extracted text, map the match
                  back to characters, then search_for the matched prefix
  3. prefix8    - search_for(first 8 words of the quote)
  4. page       - page-level fallback: the whole page rect

Citations the model left without a quote are NOT treated as quoted: they are
anchored to the best-matching sentence of the cited chunk (longest token overlap
with the claim) and reported with stage "auto_span", so the trace never claims the
model quoted something it did not.

Writes citation_anchors.json and prints how many citations resolved at each stage,
plus the failures (paper / section / quote) so mapping bugs and quoting bugs can be
told apart.

    python scripts/anchors.py [--traces traces] [--papers papers] [--out citation_anchors.json]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pymupdf

STAGE_NAMES = {1: "exact", 2: "normalised", 3: "prefix8", 4: "page", 0: "unresolved"}

DASHES = {"\u2010": "-", "\u2011": "-", "\u2212": "-", "\u2013": "-", "\u2014": "-"}
SUPERSCRIPTS = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9', '⁻': '-', '⁺': '+'}
MATH_OPS = "+\\-=<>\u00b7\u00d7/,"


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


def load_chunks(path: Path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if line.strip():
                c = json.loads(line)
                out[c["chunk_id"]] = c
    return out


def collect_citations(traces: Path) -> list[dict]:
    """No backfilling: a trace without a citations record is a hard error."""
    cites = []
    for f in sorted(traces.glob("*.jsonl")):
        for line in f.open(encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            if "citations" not in rec:
                raise SystemExit(
                    f"{f.name}: trace {rec.get('answer_id')} has no 'citations' field; "
                    "fix the pipeline (no backfilling)")
            for c in rec["citations"] or []:
                cites.append({**c, "answer_id": rec.get("answer_id"),
                              "question": rec.get("question")})
    return cites


TOKEN = re.compile(r"[a-z0-9]+")


def best_sentence(chunk_text: str, claim: str) -> str:
    """Longest token overlap with the claim: what the model probably meant."""
    claim_tokens = set(TOKEN.findall(claim.lower()))
    if not claim_tokens:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", " ".join(chunk_text.split()))
    best, best_score = "", 0.0
    for s in sents:
        toks = set(TOKEN.findall(s.lower()))
        if not toks:
            continue
        score = len(toks & claim_tokens) / len(toks | claim_tokens)
        if score > best_score:
            best, best_score = s, score
    return best


def pdf_for(paper: str, papers: Path) -> Path | None:
    hits = sorted(papers.glob(f"{paper}v*.pdf"))
    return hits[0] if hits else None


def _fold(ch: str) -> str:
    low = ch.lower()
    low = DASHES.get(low, low)
    low = SUPERSCRIPTS.get(low, low)
    if low in "$ {}^":
        return ""
    return "\u00b7" if low in ("\u00b7", "\u00d7", "\u22c5") else low


def norm_with_map(page_text: str) -> tuple[str, list[int]]:
    """Normalised page text plus, for each normalised char, its original index."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = False
    for i, ch in enumerate(page_text):
        if ch.isspace():
            if not prev_space and out:
                out.append(" ")
                idx.append(i)
            prev_space = True
            continue
        prev_space = False
        folded = _fold(ch)
        for c in folded:
            out.append(c)
            idx.append(i)
    return "".join(out), idx


def try_stages(page, quote: str) -> tuple[int, list[list[float]]]:
    # 1. exact
    rects = page.search_for(quote)
    if rects:
        return 1, rects

    # 2. normalised match, mapped back to a searchable prefix
    text = page.get_text("text")
    norm, idx = norm_with_map(text)
    nq = norm_quote(quote)
    pos = norm.find(nq)
    if pos >= 0 and nq:
        start = idx[pos]
        end = idx[min(len(idx) - 1, pos + len(nq) - 1)]
        original = text[start:end + 1]
        words = original.split()
        for take in (len(words), max(4, len(words) // 2), 6, 4):
            probe = " ".join(words[:take]) if words else ""
            if len(probe) < 8:
                continue
            rects = page.search_for(probe)
            if rects:
                return 2, rects

    # 3. eight-word prefix
    words = quote.split()
    if len(words) >= 4:
        for take in (8, 6, 4):
            probe = " ".join(words[:take])
            if len(probe) < 8:
                continue
            rects = page.search_for(probe)
            if rects:
                return 3, rects

    # 4. page-level fallback
    return 4, [page.rect]


def main() -> None:
    ap = argparse.ArgumentParser(description="Anchor cited quotes to PDF rects")
    ap.add_argument("--traces", default="traces")
    ap.add_argument("--papers", default="papers")
    ap.add_argument("--chunks", default="chunks.jsonl")
    ap.add_argument("--out", default="citation_anchors.json")
    args = ap.parse_args()

    chunks = load_chunks(Path(args.chunks))
    cites = collect_citations(Path(args.traces))
    docs: dict[str, object] = {}
    anchors, failures = [], []
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 0: 0, "auto_span": 0}

    for c in cites:
        chunk = chunks.get(c.get("chunk_id") or "")
        entry = {"answer_id": c.get("answer_id"), "paper": c.get("paper"),
                 "section": c.get("section"), "page": c.get("page"),
                 "quote": c.get("quote"), "chunk_id": c.get("chunk_id"),
                 "claim": c.get("claim"), "status": c.get("status"),
                 "stage": 0, "rects": [], "page_used": None}
        if chunk is None:
            entry["stage"] = 0
            entry["reason"] = "no chunk"
            anchors.append(entry)
            failures.append(entry)
            counts[0] += 1
            continue
        auto_span = False
        if not c.get("quote"):
            # model cited without quoting: show the best-matching sentence instead
            span = best_sentence(chunk["text"], c.get("claim") or "")
            if not span:
                entry.update(stage=0, reason="no quote and no claim to match")
                anchors.append(entry)
                failures.append(entry)
                counts[0] += 1
                continue
            c = {**c, "quote": span}
            entry["auto_span_text"] = span
            auto_span = True

        path = pdf_for(c["paper"], Path(args.papers))
        if path is None:
            entry["stage"] = 0
            entry["reason"] = "pdf missing"
            anchors.append(entry)
            failures.append(entry)
            counts[0] += 1
            continue
        doc = docs.get(path.name) or pymupdf.open(path)
        docs[path.name] = doc

        best = (0, [], None)
        for pno in range(chunk["page_start"] - 1, min(chunk["page_end"], len(doc))):
            stage, rects = try_stages(doc[pno], c["quote"])
            if stage < 4:
                best = (stage, rects, pno + 1)
                break
            if best[0] == 0:
                best = (4, rects, pno + 1)

        stage, rects, page_no = best
        entry.update(stage=stage, page_used=page_no,
                     rects=[[round(v, 2) for v in r] for r in rects])
        if auto_span:
            entry["stage_name"] = "auto_span"
            entry["pdf_stage"] = stage
        anchors.append(entry)
        counts[stage] += 1
        if auto_span:
            counts["auto_span"] += 1
        if stage >= 3 and not auto_span:
            failures.append(entry)

    out = {
        "n_citations": len(cites),
        "summary": {**{STAGE_NAMES[k]: counts[k] for k in (1, 2, 3, 4, 0)},
                    "auto_span": counts["auto_span"]},
        "anchors": anchors,
        "failures": failures,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")

    print(f"citations: {len(cites)}")
    for k in (1, 2, 3, 4, 0):
        print(f"  stage {k} {STAGE_NAMES[k]:10s}: {counts[k]}")
    print(f"  auto_span (model gave no quote): {counts['auto_span']}")
    if failures:
        print("\nfailures (stage 3/4 or unresolved):")
        for f in failures:
            print(f"  [{STAGE_NAMES[f['stage']]}] {f['paper']} §{f['section']} p{f['page']} "
                  f"chunk={f['chunk_id']} :: {f['quote']!r}")
    else:
        print("\nno failures")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
