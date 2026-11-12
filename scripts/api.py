#!/usr/bin/env python3
"""Minimal backend for quote anchors + PDF pages.

    uvicorn scripts.api:app --port 8810
    GET /citations/{answer_id}   -> {"answer_id":..., "anchors":[...]}
    GET /papers/<file>.pdf       -> the PDF itself (static)
    GET /health

Anchors come from citation_anchors.json (scripts/anchors.py). UI work comes later;
this only serves the data and the PDFs.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
ANCHORS = ROOT / "citation_anchors.json"
PAPERS = ROOT / "papers"

app = FastAPI(title="cite-anchors")


def _load() -> dict:
    if not ANCHORS.exists():
        raise HTTPException(503, "citation_anchors.json missing; run scripts/anchors.py")
    return json.loads(ANCHORS.read_text())


@app.get("/health")
def health() -> dict:
    data = _load() if ANCHORS.exists() else {}
    return {"status": "ok", "n_anchors": len(data.get("anchors") or []),
            "summary": data.get("summary")}


@app.get("/citations/{answer_id}")
def citations(answer_id: str) -> dict:
    data = _load()
    hits = [a for a in data["anchors"] if a.get("answer_id") == answer_id]
    if not hits:
        raise HTTPException(404, f"no citations for answer_id {answer_id}")
    return {"answer_id": answer_id, "anchors": hits}


@app.get("/papers/{name}")
def paper(name: str):
    path = (PAPERS / name).resolve()
    if not str(path).startswith(str(PAPERS.resolve())) or not path.exists():
        raise HTTPException(404, "no such paper")
    return FileResponse(path, media_type="application/pdf")
