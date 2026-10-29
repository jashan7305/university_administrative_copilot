"""Phase 1 - sources: collect official NMIMS pages/PDFs, extract their text, build the source inventory and the
knowledge base.

Steps (each can be run alone with --step):
  collect    download every source in config.SOURCES into data/sources/<category>/ (polite 1.5 s delay;
             robots.txt checked by hand - the upload.nmims.edu login pages it disallows are not listed)
  extract    HTML -> trafilatura main text; PDF -> pypdf per page; pages without a text layer are rendered at
             220 dpi and OCR'd with RapidOCR, with OCR boxes regrouped into lines so table rows stay together
  build      data/final/source_inventory.csv and data/final/nmims_knowledge_base.csv (page-level chunks for
             PDFs, ~220-word windows for HTML; each chunk carries the ids of facts cited from that page)

Run:  python -m dataset.phase1_sources [--step collect|extract|build]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import time
import urllib.request
from collections import Counter
from datetime import date
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError

import numpy as np
import pandas as pd

from dataset.config import EXTRACTED, EXTRACTION_LOG, FETCH_LOG, FINAL, ROOT, SOURCES, SOURCES_DIR, load_facts

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36"
DELAY_S = 1.5
MIN_TEXT_WORDS = 15          # a PDF page with fewer extractable words is treated as scanned
FETCH_COLS = ["source_id", "url", "http_status", "content_type", "bytes", "sha256", "last_modified_header",
              "local_path", "category", "school", "campus", "fetched_on", "error"]


# ----------------------------------------------------------------------------- collect

def collect() -> None:
    rows = []
    for sid, src in SOURCES.items():
        out_dir = SOURCES_DIR / src.category
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(src.url.replace(" ", "%20"), headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=60) as r:
                status, ctype, body, last_mod = r.status, r.headers.get("Content-Type", ""), r.read(), r.headers.get("Last-Modified", "")
            ext = ".pdf" if body[:4] == b"%PDF" else ".html"
            name = f"{sid}_" + re.sub(r"[^A-Za-z0-9._-]+", "_", src.url.rstrip("/").split("/")[-1] or "index")[:90]
            path = out_dir / (name if name.lower().endswith(ext) else name + ext)
            path.write_bytes(body)
            rows.append([sid, src.url, status, ctype, len(body), hashlib.sha256(body).hexdigest(), last_mod,
                         str(path.relative_to(ROOT)), src.category, src.school, src.campus, date.today().isoformat(), ""])
            print(f"{sid} {status} {len(body):>9} {src.url}")
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            rows.append([sid, src.url, getattr(e, "code", ""), "", 0, "", "", "", src.category, src.school, src.campus,
                         date.today().isoformat(), str(e)])
            print(f"{sid} FAILED {src.url}: {e}")
        time.sleep(DELAY_S)
    FETCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(FETCH_LOG, "w", newline="") as f:
        csv.writer(f).writerows([FETCH_COLS] + rows)


# ----------------------------------------------------------------------------- extract

_ocr = None


def _ocr_page(page) -> str:
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR

        _ocr = RapidOCR()
    pix = page.get_pixmap(dpi=220)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
    result, _ = _ocr(img)
    if not result:
        return ""
    boxes = sorted(((min(p[1] for p in b), max(p[1] for p in b), min(p[0] for p in b), t) for b, t, _ in result),
                   key=lambda x: (x[0], x[2]))
    lines, cur, bottom = [], [], None
    for top, bot, left, text in boxes:
        if cur and top > bottom - 0.5 * (bot - top):   # starts below the current line -> new line
            lines.append(cur)
            cur = []
        cur.append((left, text))
        bottom = bot if len(cur) == 1 else max(bottom, bot)
    if cur:
        lines.append(cur)
    return "\n".join("  ".join(t for _, t in sorted(line)) for line in lines)


def _extract_pdf(path) -> tuple[str, int, int, str]:
    import pymupdf
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    created = str((reader.metadata or {}).get("/CreationDate", "") or "")
    doc = pymupdf.open(str(path))
    parts, ocr_pages = [], 0
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if len(text.split()) < MIN_TEXT_WORDS:
            text = _ocr_page(doc[i])
            ocr_pages += 1
        parts.append(f"=== page {i + 1} ===\n{text.strip()}")
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})", created)
    return "\n\n".join(parts), len(reader.pages), ocr_pages, (f"{m[1]}-{m[2]}-{m[3]}" if m else "")


def extract() -> None:
    import trafilatura

    EXTRACTED.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in csv.DictReader(open(FETCH_LOG)):
        if not r["local_path"]:
            continue
        path = ROOT / r["local_path"]
        if path.suffix == ".pdf":
            text, pages, ocr_pages, created = _extract_pdf(path)
            method = "pdf_text" if ocr_pages == 0 else ("pdf_ocr" if ocr_pages == pages else "pdf_text+ocr")
        else:
            html = path.read_bytes().decode("utf-8", "ignore")
            text = trafilatura.extract(html, include_tables=True, include_links=False, favor_recall=True) or ""
            pages, ocr_pages, created, method = 1, 0, "", "html_trafilatura"
        (EXTRACTED / f"{r['source_id']}.txt").write_text(text)
        rows.append([r["source_id"], r["url"], method, pages, ocr_pages, len(text.split()), created])
        print(f"{r['source_id']} {method:16s} pages={pages:<4} ocr={ocr_pages:<4} words={len(text.split())}")
    with open(EXTRACTION_LOG, "w", newline="") as f:
        csv.writer(f).writerows([["source_id", "url", "method", "pages", "ocr_pages", "words", "pdf_creation_date"]] + rows)


# ----------------------------------------------------------------------------- build inventory + KB

def build() -> None:
    fetch = {r["source_id"]: r for r in csv.DictReader(open(FETCH_LOG))}
    ext = {r["source_id"]: r for r in csv.DictReader(open(EXTRACTION_LOG))}
    facts = load_facts()

    def pub_date(sid: str) -> str:
        if ext.get(sid, {}).get("pdf_creation_date"):
            return ext[sid]["pdf_creation_date"]
        try:
            lm = fetch[sid].get("last_modified_header", "")
            return parsedate_to_datetime(lm).date().isoformat() if lm else ""
        except (TypeError, ValueError):
            return ""

    inv = []
    for sid, s in SOURCES.items():
        f, status = fetch[sid], fetch[sid]["http_status"]
        inv.append({
            "source_id": sid, "title": s.title, "url": s.url, "domain": s.url.split("/")[2], "source_type": s.source_type,
            "category": s.category, "school": s.school, "campus": s.campus, "academic_year": s.academic_year,
            "publication_date": pub_date(sid), "last_verified": f["fetched_on"], "current_or_historical": s.currency,
            "relevance": s.relevance if status == "200" else "NONE",
            "notes": s.notes if status == "200" else f"HTTP {status}: not retrievable. {s.notes}".strip(),
            "http_status": status, "words_extracted": ext.get(sid, {}).get("words", "0"),
            "extraction_method": ext.get(sid, {}).get("method", ""),
            "facts_cited": sum(1 for x in facts.values() if x["source_id"] == sid),
        })
    FINAL.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inv).to_csv(FINAL / "source_inventory.csv", index=False)

    kb = []
    for row in inv:
        sid, path = row["source_id"], EXTRACTED / f"{row['source_id']}.txt"
        if row["http_status"] != "200" or not path.exists():
            continue
        text = path.read_text()
        if "=== page" in text:
            pieces = [(int(m.group(1)), body) for m, body in
                      zip(re.finditer(r"=== page (\d+) ===", text), re.split(r"=== page \d+ ===", text)[1:])]
        else:
            words = text.split()
            pieces = [(1, " ".join(words[i:i + 220])) for i in range(0, len(words), 200)]
        for page, body in pieces:
            body = re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t]+", " ", body)).strip()
            if len(body.split()) < 30:
                continue
            here = [x for x in facts.values() if x["source_id"] == sid and x["page"] == page]
            sub = Counter(x["sub_intent"] for x in here).most_common(1)
            dept = Counter(x["department"] for x in here).most_common(1)
            kb.append({
                "document_id": f"{sid}-P{page:03d}-{len(kb):05d}", "title": row["title"],
                "category": here[0]["intent"] if here else row["category"].upper(),
                "sub_category": sub[0][0] if sub else "", "department": dept[0][0] if dept else "",
                "school": row["school"], "campus": row["campus"], "academic_year": row["academic_year"],
                "document_type": row["source_type"], "content": body, "page": page, "source_id": sid,
                "source_url": row["url"], "source_title": row["title"], "source_date": row["publication_date"],
                "last_verified": row["last_verified"], "current_or_historical": row["current_or_historical"],
                "fact_ids": "|".join(sorted(x["fact_id"] for x in here)),
            })
    pd.DataFrame(kb).to_csv(FINAL / "nmims_knowledge_base.csv", index=False)
    print(f"inventory: {len(inv)} sources; knowledge base: {len(kb)} chunks")


STEPS = {"collect": collect, "extract": extract, "build": build}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step", choices=STEPS, help="run a single step (default: all)")
    args = ap.parse_args()
    for name in ([args.step] if args.step else STEPS):
        STEPS[name]()
