#!/usr/bin/env python3
"""Ingest arXiv papers into section-based chunks (JSONL).

Sources
- Text and tables: ar5iv HTML (``https://ar5iv.labs.arxiv.org/html/{id}v{ver}``),
  cached under ``papers/html/``. Sections come from ``<section>``/``<h2>-<h6>``,
  tables from ``<table class="ltx_tabular">``.
- Page numbers: the PDF is used only to locate a chunk's page range.
- If an ar5iv fetch/parse fails for a paper, that paper alone falls back to the
  PDF (PyMuPDF) for text and tables; the failure is logged.

Chunking
- One chunk per section heading; a section longer than ``--max-chars`` is split
  at sentence boundaries. ``--max-chars 0`` disables splitting.
- Each table is one self-contained chunk (``is_table=True``), never split.
- References / bibliography / appendix-bibliography chunks are dropped.

Output: JSONL with chunk_id, paper, version, section_path, page_start,
page_end, text, is_table.

Usage:
    python scripts/ingest.py [--papers-dir papers] [--output chunks.jsonl]
                             [--max-chars 6000] [--refresh]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

try:
    import pymupdf  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF not installed. Run: pip install -r scripts/requirements.txt")

try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("beautifulsoup4 not installed. Run: pip install -r scripts/requirements.txt")

AR5IV = "https://ar5iv.labs.arxiv.org/html/{id}v{ver}"
FILENAME_RE = re.compile(r"^(?P<id>\d{4}\.\d{4,5})v(?P<ver>\d+)(?:-(?P<slug>.+))?\.pdf$")
DROPPED_SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|appendix\s+bibliography)\s*$", re.I)
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def parse_paper(path: Path) -> tuple[str, str]:
    m = FILENAME_RE.match(path.name)
    if not m:
        return path.stem, "1"
    return m.group("id"), m.group("ver")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# --------------------------------------------------------------------------- #
# ar5iv HTML
# --------------------------------------------------------------------------- #
def fetch_html(arxiv_id: str, ver: str, cache_dir: Path, refresh: bool) -> Path | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{arxiv_id}v{ver}.html"
    if dest.exists() and dest.stat().st_size > 0 and not refresh:
        return dest
    url = AR5IV.format(id=arxiv_id, ver=ver)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vaani-mle-ingest/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        if not data or b"<html" not in data[:2000].lower():
            raise ValueError("response is not HTML")
        dest.write_bytes(data)
        return dest
    except (urllib.error.URLError, ValueError, TimeoutError, OSError) as exc:
        log(f"[ar5iv] FAILED {url}: {exc}")
        return None


def html_text(node) -> str:
    """Readable text: math alttext instead of MathML, skips scripts/styles."""
    for bad in node.find_all(["script", "style", "annotation"]):
        bad.decompose()
    for m in node.find_all("math"):
        m.replace_with(m.get("alttext") or m.get_text(" ", strip=True))
    txt = node.get_text("\n", strip=True)
    return re.sub(r"[ \t]+", " ", txt)


def is_prose(text: str) -> bool:
    """A wide cell holding a description rather than a column value."""
    words = text.split()
    return len(words) >= 5 and len(text) >= 40


def render_grid(rows: list[list], where: str) -> str:
    """Render a table as TSV. Every row must match the header column count;
    a wide cell holding prose becomes a `note:` line after the table."""
    grid: dict[tuple[int, int], str] = {}
    occupied: set[tuple[int, int]] = set()
    notes: list[str] = []
    width = 0
    for r, row in enumerate(rows):
        c = 0
        for cell in row:
            while (r, c) in occupied:
                c += 1
            text = cell["text"]
            cs, rs = max(1, cell["colspan"]), max(1, cell["rowspan"])
            if cs > 1 and is_prose(text):
                notes.append(text)  # covered columns are described in prose
                text = ""
            for dr in range(rs):
                for dc in range(cs):
                    grid[(r + dr, c + dc)] = text if (dr == 0 and dc == 0) else ""
                    occupied.add((r + dr, c + dc))
            c += cs
            width = max(width, c)
    if not grid or width < 2:
        return ""
    height = max(r for r, _ in grid) + 1
    lines = ["\t".join(grid.get((r, c), "") for c in range(width)) for r in range(height)]
    lines = [ln for ln in lines if ln.strip("\t")]
    if not lines:
        return ""
    header_cols = len(lines[0].split("\t"))
    for i, ln in enumerate(lines):
        got = len(ln.split("\t"))
        if got != header_cols:
            raise SystemExit(
                f"TABLE ALIGNMENT ERROR in {where}: row {i} has {got} cells, "
                f"header has {header_cols}")
    return "\n".join(lines + [f"note: {n}" for n in notes])


def rows_from_html(tbl) -> list[list[dict]]:
    rows = []
    for tr in tbl.find_all("tr"):
        if tr.find_parent("table") is not tbl:
            continue  # nested table belongs to its own render
        row = []
        for cell in tr.find_all(["td", "th"], recursive=False):
            row.append({
                "text": " ".join(html_text(cell).split()),
                "colspan": int(cell.get("colspan") or 1),
                "rowspan": int(cell.get("rowspan") or 1),
            })
        if any(c["text"] for c in row):
            rows.append(row)
    return rows


def table_to_tsv(tbl, where: str) -> str:
    return render_grid(rows_from_html(tbl), where)


def is_data_table(tbl) -> bool:
    cls = " ".join(tbl.get("class") or [])
    if "ltx_equation" in cls or "ltx_eqn_table" in cls:
        return False  # display equations are rendered as tables
    if "ltx_tabular" in cls:
        return True
    return tbl.find("table", class_="ltx_tabular") is not None


def parse_html(path: Path) -> list[dict]:
    """Sections as blocks: {'section_path', 'text', 'tables': [markdown]}."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    root = soup.body or soup
    blocks: list[dict] = []
    stack: list[tuple[str, str]] = []  # (tag, title)
    cur = {"section_path": "Front matter", "parts": [], "tables": []}
    blocks.append(cur)

    def close_table(tbl) -> None:
        if not is_data_table(tbl):
            return
        where = f"{path}[{tbl.get('id') or 'table'}]"
        md = table_to_tsv(tbl, where)
        if not md:
            return
        cap = ""
        holder = tbl.find_parent(class_="ltx_table") or tbl
        for cand in [holder.find(class_="ltx_caption")] + holder.find_all(class_="ltx_caption"):
            if cand is not None:
                cap = " ".join(html_text(cand).split())
                break
        cur["tables"].append((cap + "\n" if cap else "") + md)

    for el in root.descendants:
        name = getattr(el, "name", None)
        if name in ("h2", "h3", "h4", "h5", "h6"):
            title = " ".join(html_text(el).split())
            if not title:
                continue
            level = int(name[1])
            while stack and int(stack[-1][0][1]) >= level:
                stack.pop()
            stack.append((name, title))
            cur = {"section_path": " > ".join(t for _, t in stack),
                   "parts": [], "tables": []}
            blocks.append(cur)
        elif name == "table":
            close_table(el)
        elif name in ("p", "li", "div") and el.find(["p", "li"]):
            continue  # container; its paragraphs are visited themselves
        elif name in ("p", "li"):
            txt = html_text(el)
            if txt:
                cur["parts"].append(txt)
    return blocks


# --------------------------------------------------------------------------- #
# PDF: text+table fallback and page lookup
# --------------------------------------------------------------------------- #
def page_index(pdf_path: Path) -> list[str]:
    """Normalized text per PDF page, used to locate chunks."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for p in doc:
        raw = p.get_text("text")
        pages.append(re.sub(r"[^a-z0-9]+", "", raw.lower()))
    doc.close()
    return pages


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def locate_pages(text: str, pages: list[str], hint: int) -> tuple[int, int]:
    """Find the PDF page range for a chunk's text via normalized snippets."""
    flat = norm(text)
    if not flat or not pages:
        return hint, hint
    probes = [flat[:80], flat[len(flat) // 2: len(flat) // 2 + 80], flat[-80:]]
    hits = []
    for pr in probes:
        if len(pr) < 20:
            continue
        for i, p in enumerate(pages):
            if pr in p:
                hits.append(i + 1)
                break
    if hits:
        return min(hits), max(hits)
    return hint, hint  # text not found in the PDF: keep the previous page


def pdf_blocks(pdf_path: Path) -> list[dict]:
    """Fallback: PDF sections + tables, same block shape as parse_html."""
    doc = pymupdf.open(pdf_path)
    blocks: list[dict] = []
    cur = {"section_path": "Front matter", "parts": [], "tables": []}
    blocks.append(cur)
    stack: list[tuple[int, str]] = []
    heading_re = re.compile(
        r"^(?:appendix\s+)?(?:[A-Z](?:\.\d+)*|\d+(?:\.\d+)*\.?)?\s*"
        r"(abstract|introduction|related work|background|preliminaries|"
        r"method(?:ology|od)?s?|experiments?|results?|discussion|conclusions?|"
        r"limitations?|acknowledg(e)?ments?|references|bibliography)$", re.I)
    for pno, page in enumerate(doc):
        for b in page.get_text("dict")["blocks"]:
            if b.get("type", 0) != 0:
                continue
            lines = b.get("lines", [])
            text = " ".join("".join(s["text"] for s in l["spans"]).strip() for l in lines)
            text = " ".join(text.split())
            if not text:
                continue
            size = max((s["size"] for l in lines for s in l["spans"]), default=10.0)
            bold = any("bold" in (s.get("font", "") or "").lower()
                       for l in lines for s in l["spans"])
            num = re.match(r"^(\d+(?:\.\d+)*)\s+(\S.{0,80})$", text)
            if ((heading_re.match(text) or (num and bold and not text.endswith(".")))
                    and size > 10.5):
                title = num.group(0) if num else text
                level = (num.group(1).count(".") + 1) if num else 1
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                cur = {"section_path": " > ".join(t for _, t in stack),
                       "parts": [], "tables": []}
                blocks.append(cur)
            else:
                cur["parts"].append(text)
        try:
            for t in page.find_tables().tables:
                cells = t.extract()
                where = f"{cur['section_path']}[pdf p{pno + 1}]"
                rows = [[{"text": " ".join((c or "").split()), "colspan": 1, "rowspan": 1}
                         for c in r] for r in cells]
                rows = [r for r in rows if any(c["text"] for c in r)]
                if len(rows) >= 2:
                    tsv = render_grid(rows, where)
                    if tsv:
                        cur["tables"].append(tsv)
        except SystemExit:
            raise
        except Exception:
            pass
    doc.close()
    return blocks


# --------------------------------------------------------------------------- #
# chunking
# --------------------------------------------------------------------------- #
def split_sentences(text: str, max_chars: int) -> list[str]:
    """Pack a section up to max_chars, splitting only at sentence ends."""
    if max_chars <= 0:
        return [text] if text.strip() else []
    sentences: list[str] = []
    for para in text.split("\n\n"):
        para = " ".join(para.split())
        if para:
            sentences.extend(s.strip() for s in SENTENCE_END_RE.split(para) if s.strip())
    out: list[str] = []
    buf = ""
    for sent in sentences:
        while len(sent) > max_chars:  # pathological single sentence
            cut = sent.rfind(" ", 0, max_chars)
            cut = max_chars if cut < max_chars // 2 else cut
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
            if not sent:
                break
        if not sent:
            continue
        if not buf:
            buf = sent
        elif len(buf) + 1 + len(sent) <= max_chars:
            buf += " " + sent
        else:
            out.append(buf)
            buf = sent
    if buf:
        out.append(buf)
    return out


def merge_short(chunks: list[dict], min_chars: int) -> tuple[list[dict], int, int]:
    """Fold chunks under `min_chars` into the next chunk of the same section;
    a short chunk with nothing following it in its section is dropped."""
    out: list[dict] = []
    merged = dropped = 0
    i = 0
    while i < len(chunks):
        cur = chunks[i]
        if cur["is_table"] or len(cur["text"]) >= min_chars:
            out.append(cur)
            i += 1
            continue
        nxt = chunks[i + 1] if i + 1 < len(chunks) else None
        if nxt is not None and not nxt["is_table"] and nxt["section_path"] == cur["section_path"]:
            nxt["text"] = cur["text"] + "\n\n" + nxt["text"]
            nxt["page_start"] = min(cur["page_start"], nxt["page_start"])
            nxt["page_end"] = max(cur["page_end"], nxt["page_end"])
            merged += 1
            i += 1  # consumed into nxt
            continue
        log(f"[drop] {cur['section_path']}: {len(cur['text'])} chars, "
            f"no following chunk: {cur['text'][:60]!r}")
        dropped += 1
        i += 1
    return out, merged, dropped


def build_chunks(paper: str, ver: str, blocks: list[dict], pages: list[str],
                 max_chars: int) -> list[dict]:
    out: list[dict] = []
    hint = 1
    for blk in blocks:
        path = blk["section_path"]
        if DROPPED_SECTION_RE.match(path.split(" > ")[-1]):
            continue
        text = "\n\n".join(blk["parts"]).strip()
        for piece in split_sentences(text, max_chars):
            ps, pe = locate_pages(piece, pages, hint)
            hint = pe
            out.append({"paper": paper, "version": f"v{ver}", "section_path": path,
                        "page_start": ps, "page_end": pe, "text": piece,
                        "is_table": False})
        for tbl in blk["tables"]:
            ps, pe = locate_pages(tbl, pages, hint)
            hint = pe
            out.append({"paper": paper, "version": f"v{ver}", "section_path": path,
                        "page_start": ps, "page_end": pe, "text": tbl,
                        "is_table": True})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest arXiv papers into chunks.jsonl")
    ap.add_argument("--papers-dir", default="papers")
    ap.add_argument("--html-dir", default="papers/html")
    ap.add_argument("--output", default="chunks.jsonl")
    ap.add_argument("--max-chars", type=int, default=6000,
                    help="split sections at sentence boundaries (0 disables)")
    ap.add_argument("--min-chars", type=int, default=120,
                    help="merge chunks below this size into the next one")
    ap.add_argument("--refresh", action="store_true", help="re-download ar5iv HTML")
    args = ap.parse_args()

    pdfs = sorted(Path(args.papers_dir).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {args.papers_dir}")

    all_chunks: list[dict] = []
    for pdf in pdfs:
        paper, ver = parse_paper(pdf)
        pages = page_index(pdf)

        html_path = fetch_html(paper, ver, Path(args.html_dir), args.refresh)
        blocks = None
        if html_path is not None:
            try:
                blocks = parse_html(html_path)
            except SystemExit:
                raise
            except Exception as exc:
                log(f"[ar5iv] PARSE FAILED {html_path.name}: {exc}")
        if blocks:
            source = "ar5iv"
        else:
            log(f"[ar5iv] falling back to PDF text/tables for {paper}v{ver} only")
            blocks = pdf_blocks(pdf)
            source = "pdf"

        chunks = build_chunks(paper, ver, blocks, pages, args.max_chars)
        chunks, merged, dropped = merge_short(chunks, args.min_chars)
        for c in chunks:
            all_chunks.append(c)
        n_tab = sum(1 for c in chunks if c["is_table"])
        print(f"{paper}v{ver} [{source}]: {len(chunks)} chunks ({n_tab} tables), "
              f"{merged} merged, {dropped} dropped")

    for i, c in enumerate(all_chunks, 1):
        c["chunk_id"] = f"{c['paper']}{c['version']}-{i:04d}"
    order = ["chunk_id", "paper", "version", "section_path",
             "page_start", "page_end", "text", "is_table"]
    with Path(args.output).open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps({k: c[k] for k in order}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_chunks)} chunks -> {args.output}")


if __name__ == "__main__":
    main()
