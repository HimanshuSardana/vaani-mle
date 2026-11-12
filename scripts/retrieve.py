#!/usr/bin/env python3
"""BM25 retrieval over chunks.jsonl, with optional cross-encoder reranking.

    python scripts/retrieve.py "how does dynamic chunking differ from BPE"
    python scripts/retrieve.py --top-k 8 --rerank --union-cap 8 "BLEU of big transformer"
    python scripts/retrieve.py --per-paper --rerank "parameter counts of the three papers"
    python scripts/retrieve.py --rerank --refuse-gate --gate-threshold -5 "capital of France"

Behaviour
- Indexed text = chunk text + section_path terms (path added twice, so section
  terms weigh a little more and "3.2 architecture"-style queries can hit).
- Paper hint: a query naming/aliasing a paper (moshi/helium, attention/
  transformer, dynamic chunking/h-net/hierarchical) boosts that paper's chunks
  by 1.5; --hard-hint restricts to it. A bare "transformer" only ever matches
  1706.03762.
- Table boost: numeric/measurement intent multiplies is_table scores by 1.5.
- --per-paper: when the hint covers several papers, or the hint is empty and the
  query compares papers, BM25 runs once per paper and the results merge, each
  paper contributing up to --top-k.
- --rerank: union (not replace) of BM25 top-k and rerank survivors, ordered by
  cross-encoder score, deduped, capped at --union-cap.
- --refuse-gate: DIAGNOSTIC ONLY. When the best cross-encoder logit over the
  BM25 candidate pool is below --gate-threshold it logs "would refuse" to stderr
  and sets stats.gate_would_refuse, but results are always returned. It never
  drops chunks and never reorders.

Output (stdout, JSON):
    {"results": [{chunk_id, paper, version, section_path, page_start, page_end,
                  is_table, score, text}, ...], "stats": {...}}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from pathlib import Path
from statistics import median

TOKEN_RE = re.compile(r"[a-z0-9]+")
# Bump when tokenize() or the corpus layout below changes: the cache key includes
# it, so an old pickled index is never reused against new text handling.
TOKENIZER_VERSION = "v1"
INDEX_CACHE = Path(".state/bm25_index.pkl")
RERANK_MODEL = "cross-encoder/ms-marco-minilm-l-6-v2"
RERANK_POOL = 25

PAPER_ALIASES: dict[str, tuple[str, ...]] = {
    "1706.03762": ("attention is all you need", "attention", "transformer",
                   "vaswani", "self-attention"),
    "2410.00037": ("moshi", "helium", "speech-text foundation model"),
    "2507.07955": ("dynamic chunking", "h-net", "hnet", "hierarchical",
                   "end-to-end hierarchical sequence modeling"),
}
PAPER_BOOST = 1.5
TABLE_BOOST = 1.5
TABLE_INTENT = ("how many", "how much", "bleu", "wer", "word error",
                "learning rate", "parameters", "params", "warm-up", "warm up",
                "batch size", "ratio", "%")
COMPARE_INTENT = ("compare", "comparison", " vs ", " versus ", "which papers",
                  "which of these", "all three", "three papers", "both papers",
                  "across the papers", "differ between papers", "the three",
                  "each paper", "each of the papers", "across papers")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def is_appendix_like(section_path: str) -> bool:
    """Appendix / lettered top-level sections (A-F) / acknowledgements."""
    top = section_path.split(" > ")[0].strip().lower()
    return (top.startswith("appendix")
            or re.match(r"^[a-f]\b", top) is not None
            or top.startswith("acknowledg"))


def detect_hint(query: str) -> list[str]:
    """Alias match on tokens, so punctuation ('H-Net)', 'transformer,') is safe."""
    toks = tokenize(query)
    hits = []
    for paper, aliases in PAPER_ALIASES.items():
        for alias in aliases:
            a = tokenize(alias)
            n = len(a)
            if any(toks[i:i + n] == a for i in range(len(toks) - n + 1)):
                hits.append(paper)
                break
    return hits


def wants_table(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in TABLE_INTENT)


def wants_compare(query: str) -> bool:
    q = " " + " ".join(query.lower().split()) + " "
    return any(k in q for k in COMPARE_INTENT)


def pair_text(chunk: dict, limit: int = 384) -> str:
    """What the cross-encoder actually reads: short head of the chunk + section.

    Chunks are long (median ~1k chars) but the model truncates at 512 tokens, so
    anything past the head is paid for and never seen (~128 ms/pair -> ~20 ms).
    """
    head = " ".join(chunk["text"].split())[:limit]
    return f"{head}\n{chunk.get('section_path', '')}"


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"chunks file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def corpus_tokens(chunks: list[dict]) -> list[list[str]]:
    """Indexed text: chunk text plus the section path twice, so headings weigh more."""
    corpus = []
    for c in chunks:
        sec = tokenize(c.get("section_path", ""))
        corpus.append(tokenize(c["text"]) + sec + sec)
    return corpus


def build_index(chunks: list[dict]):
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        sys.exit("rank_bm25 not installed. Run: pip install -r scripts/requirements.txt")
    return BM25Okapi(corpus_tokens(chunks))


def index_key(chunks_path: Path) -> str:
    """Identifies the exact chunks file and tokenizer the index was built from."""
    digest = hashlib.sha256(Path(chunks_path).read_bytes()).hexdigest()[:16]
    return f"{TOKENIZER_VERSION}:{len(load_chunks(Path(chunks_path)))}:{digest}"


def save_index(chunks_path: Path, bm25, cache: Path = INDEX_CACHE) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"key": index_key(chunks_path), "index": bm25}, f, protocol=4)
    os.replace(tmp, cache)  # atomic, so a killed process cannot leave a half file


def load_or_build_index(chunks_path: Path, chunks: list[dict] | None = None,
                        cache: Path = INDEX_CACHE, rebuild: bool = False):
    """Load the pickled BM25 index when it matches chunks.jsonl, else build and save."""
    chunks = chunks if chunks is not None else load_chunks(Path(chunks_path))
    key = index_key(Path(chunks_path))
    if not rebuild and cache.exists():
        try:
            with cache.open("rb") as f:
                blob = pickle.load(f)
            if blob.get("key") == key:
                return blob["index"]
            print(f"[index] stale cache for {key}, rebuilding", file=sys.stderr)
        except Exception as exc:  # corrupt, truncated, or a different rank_bm25 build
            print(f"[index] unusable cache ({exc}), rebuilding", file=sys.stderr)
    bm25 = build_index(chunks)
    save_index(Path(chunks_path), bm25, cache)
    print(f"[index] built and saved {key}", file=sys.stderr)
    return bm25


def load_reranker():
    """Lazy cross-encoder load; never fall back to BM25 order on failure."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        sys.exit(f"--rerank needs sentence-transformers ({exc}). "
                 "Install it with: pip install sentence-transformers")
    try:
        return CrossEncoder(RERANK_MODEL)
    except Exception as exc:  # missing weights, no network, bad cache, ...
        sys.exit(f"--rerank could not load '{RERANK_MODEL}': {exc}")


def stats(n_indexed: int, n_candidates: int, reranked: bool,
          query_terms: list[str], hint: list[str], table_boost: bool,
          score_floor: float, score_max: float = 0.0,
          score_median: float = 0.0, n_below_floor: int = 0,
          n_bm25: int = 0, n_rerank_kept: int = 0, union_cap: int = 0,
          per_paper_hits: dict | None = None, best_logit: float | None = None,
          refused: bool = False, gate_threshold: float | None = None,
          gate_would_refuse: bool = False) -> dict:
    return {"n_chunks_indexed": n_indexed,
            "n_candidates_after_filter": n_candidates,
            "reranked": reranked,
            "query_terms_used": query_terms,
            "hint": hint,
            "table_boost": table_boost,
            "score_floor": score_floor,
            "score_max": round(score_max, 4),
            "score_median": round(score_median, 4),
            "n_below_floor": n_below_floor,
            "n_bm25": n_bm25,
            "n_rerank_kept": n_rerank_kept,
            "union_cap": union_cap,
            "per_paper_hits": per_paper_hits or {},
            "best_logit": None if best_logit is None else round(best_logit, 4),
            "refused": refused,
            "gate_threshold": gate_threshold,
            "gate_would_refuse": gate_would_refuse}


def emit(results: list[dict], st: dict, refused: bool = False) -> None:
    fields = ["chunk_id", "paper", "version", "section_path", "page_start",
              "page_end", "is_table", "score", "text"]
    out: dict = {"refused": refused,
                 "results": [] if refused else [{k: r[k] for k in fields} for r in results]}
    out["stats"] = st
    print(json.dumps(out, ensure_ascii=False, indent=2))


def score_chunks(pairs, hint, table_boost):
    scored = []
    for c, s in pairs:
        score = float(s)
        if hint and c["paper"] in hint:
            score *= PAPER_BOOST
        if table_boost and c["is_table"]:
            score *= TABLE_BOOST
        scored.append((c, score))
    return scored


def main() -> None:
    ap = argparse.ArgumentParser(description="BM25 retrieval over chunks.jsonl")
    ap.add_argument("query", nargs="+", help="query text")
    ap.add_argument("--chunks", default="chunks.jsonl")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true",
                    help="rerank BM25 candidates with a cross-encoder")
    ap.add_argument("--rerank-pool", type=int, default=25,
                    help="BM25 candidates sent to the cross-encoder")
    ap.add_argument("--union-cap", type=int, default=8,
                    help="cap on union(bm25 top-k, rerank survivors)")
    ap.add_argument("--per-paper", action="store_true",
                    help="one BM25 round per paper when the query spans papers")
    ap.add_argument("--refuse-gate", action="store_true",
                    help="diagnostic only: log whether the best logit is below the threshold")
    ap.add_argument("--gate-threshold", type=float, default=0.0,
                    help="cross-encoder logit threshold for --refuse-gate")
    ap.add_argument("--exclude-appendix", action="store_true",
                    help="skip appendix / lettered / acknowledgements sections")
    ap.add_argument("--hard-hint", action="store_true",
                    help="restrict results to the paper the query names")
    ap.add_argument("--score-floor", type=float, default=0.0,
                    help="drop chunks scoring below this (default 0)")
    args = ap.parse_args()

    query = " ".join(args.query)
    terms = tokenize(query)
    hint = detect_hint(query)
    table_boost = wants_table(query)
    top_k = max(0, args.top_k)

    chunks = load_chunks(Path(args.chunks))
    base = stats(len(chunks), 0, args.rerank, terms, hint, table_boost,
                 args.score_floor, union_cap=args.union_cap,
                 gate_threshold=args.gate_threshold if args.refuse_gate else None)
    if not chunks:
        emit([], base)
        return

    pool = chunks
    if args.hard_hint and hint:
        pool = [c for c in pool if c["paper"] in hint]
    if args.exclude_appendix:
        pool = [c for c in pool if not is_appendix_like(c["section_path"])]
    if not pool:
        emit([], base)
        return

    # ---- candidate generation -------------------------------------------------
    per_paper_hits: dict[str, int] = {}
    rank_lists: dict[str, list[dict]] = {}
    papers = sorted({c["paper"] for c in pool})
    multi_paper_round = args.per_paper and (
        len(hint) > 1 or (not hint and wants_compare(query) and len(papers) > 1))

    if multi_paper_round:
        targets = [p for p in papers if p in hint] if len(hint) > 1 else papers
        candidates: list[tuple[dict, float]] = []
        seen: set[str] = set()
        for paper in targets:
            sub = [c for c in pool if c["paper"] == paper]
            if not sub:
                continue
            sc = score_chunks(zip(sub, build_index(sub).get_scores(terms)), hint, table_boost)
            sc.sort(key=lambda cs: cs[1], reverse=True)
            taken = sc[:top_k]
            per_paper_hits[paper] = len(taken)
            rank_lists[paper] = [c for c, _ in taken]
            for c, s in taken:
                if c["chunk_id"] not in seen:
                    seen.add(c["chunk_id"])
                    candidates.append((c, s))
    else:
        sc = score_chunks(zip(pool, build_index(pool).get_scores(terms)), hint, table_boost)
        sc.sort(key=lambda cs: cs[1], reverse=True)
        candidates = sc

    if not candidates:
        emit([], base)
        return

    bm25_top = candidates if multi_paper_round else candidates[:top_k]
    bm25_ids = {c["chunk_id"] for c, _ in bm25_top}

    # ---- optional rerank: union of bm25 top-k and rerank survivors ------------
    results: list[dict] = []
    n_rerank_kept = 0
    best_logit = None
    gate_would_refuse = False

    if args.refuse_gate and not args.rerank:
        sys.exit("--refuse-gate needs --rerank (the gate uses cross-encoder logits)")

    if args.rerank:
        model = load_reranker()
        head = candidates[:max(1, args.rerank_pool)]
        try:
            ce = model.predict([(query, pair_text(c)) for c, _ in head])
        except Exception as exc:
            sys.exit(f"--rerank prediction failed: {exc}")
        ranked = sorted((dict(c, score=float(s)) for (c, _), s in zip(head, ce)),
                        key=lambda r: r["score"], reverse=True)
        best_logit = ranked[0]["score"] if ranked else None

        # --refuse-gate is diagnostic only: record/log the decision, never drop
        # results in the production path.
        gate_would_refuse = (args.refuse_gate and best_logit is not None
                             and best_logit < args.gate_threshold)
        if gate_would_refuse:
            print(f"[gate] would refuse: max_logit {best_logit:.4f} < "
                  f"{args.gate_threshold} (results kept)", file=sys.stderr)

        survivors = [r for r in ranked if r["score"] >= args.score_floor]
        n_rerank_kept = len(survivors)
        # union: bm25 top-k OR rerank survivor, ordered by rerank score, capped
        union = [r for r in ranked
                 if r["chunk_id"] in bm25_ids or r["score"] >= args.score_floor]
        results = union[:max(0, args.union_cap)]
    else:
        results = [dict(c, score=s) for c, s in bm25_top]

    if multi_paper_round and targets:
        # Interleave each paper's own BM25 ranking, so every paper contributes
        # and a paper's rank-k chunk is not lost to another paper's rerank scores.
        by_id = {r["chunk_id"]: r for r in results}
        for c, sc_ in candidates:  # union members outside the cap still usable
            by_id.setdefault(c["chunk_id"], dict(c, score=sc_))
        interleaved: list[dict] = []
        for rank in range(top_k):
            for paper in targets:
                lst = rank_lists.get(paper) or []
                if rank < len(lst):
                    r = by_id.get(lst[rank]["chunk_id"])
                    if r is not None:
                        interleaved.append(r)
        results = interleaved[:max(1, args.union_cap)]

    shown = [r["score"] for r in results]
    st = stats(len(chunks), len(pool), args.rerank, terms, hint, table_boost,
               args.score_floor,
               score_max=max(shown, default=0.0),
               score_median=median(shown) if shown else 0.0,
               n_below_floor=len(ranked) - n_rerank_kept if args.rerank else
               sum(1 for _, s in candidates if s < args.score_floor),
               n_bm25=len(bm25_ids), n_rerank_kept=n_rerank_kept,
               union_cap=args.union_cap, per_paper_hits=per_paper_hits,
               best_logit=best_logit, refused=False,
               gate_threshold=args.gate_threshold if args.refuse_gate else None,
               gate_would_refuse=gate_would_refuse)
    emit(results, st)


if __name__ == "__main__":
    main()
