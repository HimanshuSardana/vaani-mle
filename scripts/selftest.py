#!/usr/bin/env python3
"""Offline self-test: import every module, build the index, load the reranker, run
verify() on fixtures. Zero API calls. Exits non-zero on any failure.

    python scripts/selftest.py
"""
from __future__ import annotations

import importlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

CHUNKS = ROOT / "chunks.jsonl"
FIXTURE = {
    "paper": "1706.03762", "chunk_id": "fx-1",
    "section_path": "6 Results > 6.1 Machine Translation", "page_start": 8, "page_end": 8,
    "is_table": False, "version": "v7",
    "text": "On the WMT 2014 English-to-German task it establishes a new "
            "state-of-the-art BLEU score of 28.4 .",
}

failures: list[str] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        print(f"  ok   {name}" + (f"  [{detail}]" if detail else ""))
    except Exception as exc:
        failures.append(f"{name}: {exc}")
        print(f"  FAIL {name}: {exc}")
        traceback.print_exc(limit=2)


def main() -> None:
    print("1. imports")
    mods = {}
    for name in ("client", "retrieve", "pipeline", "anchors", "api"):
        def _import(n=name):
            mods[n] = importlib.import_module(n)
            return "loaded"
        check(f"import {name}", _import)

    print("2. chunk corpus")
    chunks: list[dict] = []

    def _load():
        assert CHUNKS.exists(), "chunks.jsonl missing"
        with CHUNKS.open(encoding="utf-8") as f:
            chunks.extend(json.loads(line) for line in f if line.strip())
        assert len(chunks) > 100, f"only {len(chunks)} chunks"
        return f"{len(chunks)} chunks"
    check("load chunks.jsonl", _load)

    print("3. bm25 index")
    def _index():
        import retrieve
        bm25 = retrieve.build_index(chunks)
        scores = bm25.get_scores(retrieve.tokenize("bleu score english german"))
        assert max(scores) > 0, "no positive score"
        return f"top score {max(scores):.2f}"
    check("build BM25 index", _index)

    print("4. reranker")
    def _rerank():
        import pipeline
        model = pipeline.get_reranker()
        assert model is not None, "cross-encoder unavailable"
        import retrieve
        pairs = [("bleu", retrieve.pair_text(c, 128)) for c in chunks[:2]]
        s = model.predict(pairs)
        assert len(s) == 2
        return "2 pairs scored"
    check("load + predict cross-encoder", _rerank)

    print("5. verify() citation survival")
    def _verify():
        import pipeline
        ok = "It scores 28.4 BLEU [1706.03762 §6.1 Machine Translation p8]."
        r = pipeline.verify(ok, [FIXTURE])
        assert r["failed"] is False and r["citations"][0]["status"] == "resolved", r
        assert r["citations"][0]["chunk_id"] == "fx-1", r["citations"]
        bad = "It scores 28.4 BLEU [9999.99999 §Nowhere p3]."
        r2 = pipeline.verify(bad, [FIXTURE])
        assert r2["citations"][0]["status"] == "unresolved", r2["citations"]
        assert "9999.99999" not in r2["answer"], "unresolved citation not dropped"
        return "resolved + dropped"
    check("verify survival", _verify)

    print("6. prose -> citation expansion + auto-cite")
    def _parse():
        import pipeline
        out = pipeline.expand_passage_refs("It scores 28.4 [1]. It leads [1].", [FIXTURE])
        assert out.count("[1706.03762 §") == 2, out
        assert "[1]" not in out, "passage marker survived expansion"
        auto, added = pipeline.auto_cite("The model scores 28.4 BLEU.", [FIXTURE])
        assert added == 1 and "§" in auto, "auto_cite produced no citation"
        return "expanded + auto-cited"
    check("expand + auto_cite", _parse)

    print("6b. table rendering + column check")
    def _tables():
        import pipeline
        tbl = {"paper": "1706.03762", "chunk_id": "tb-1", "section_path": "6.1 MT",
               "page_start": 8, "page_end": 8, "is_table": True, "version": "v7",
               "text": ("Table 2: BLEU results\nModel\tBLEU\t\tTraining Cost (FLOPs)\t\n"
                        "\tEN-DE\tEN-FR\t\tEN-DE\tEN-FR\n"
                        "Transformer (big)\t28.4\t41.8\t\t2.3\\cdot 10^{19}\t1.4\\cdot 10^{20}\n")}
        rendered = pipeline.render_table_passage(tbl)
        assert "|" in rendered and "row \"Transformer (big)\"" in rendered, rendered[:120]
        assert "\t" not in rendered, "raw TSV leaked into the passage"
        assert pipeline.column_check(tbl, ["28.4"], "en-de") == "ok"
        # row-level check cannot separate two columns that share one row; grade.py's
        # forbidden_facts is what catches 41.8. Assert the honest behaviour:
        assert pipeline.column_check(tbl, ["41.8"], "en-de") == "ok"
        assert pipeline.column_check(tbl, ["28.4"], None) == "skipped"
        return "rendered"
    check("table render + column check", _tables)

    print("7. trace guard")
    def _guard():
        import pipeline
        try:
            pipeline.assert_trace_complete({"answer_id": "x", "citations": []})
        except RuntimeError:
            pass
        else:
            raise AssertionError("guard did not fire on a missing verify result")
        pipeline.assert_trace_complete({"answer_id": "y", "citations": [], "verify_checked": True,
                                        "status": "ok",
                                        "answer_text": "Not stated in the provided papers. why"})
        return "fires and passes"
    check("assert_trace_complete", _guard)

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("selftest passed (no API calls made)")


if __name__ == "__main__":
    main()
