from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import re
import fitz
import pytesseract
from PIL import Image

@dataclass
class PageText:
    document_id: str
    filename: str
    page_number: int
    text: str
    used_ocr: bool

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    start_char: int
    end_char: int


def document_id_for(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:16]


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


import shutil
from concurrent.futures import ThreadPoolExecutor

_HAS_TESSERACT: bool | None = None

def is_tesseract_available() -> bool:
    global _HAS_TESSERACT
    if _HAS_TESSERACT is None:
        try:
            cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
            _HAS_TESSERACT = shutil.which(cmd) is not None or Path(cmd).is_file()
        except Exception:
            _HAS_TESSERACT = False
    return _HAS_TESSERACT


def extract_pdf(path: Path, ocr_threshold: int = 0, ocr_dpi: int = 220) -> list[PageText]:
    path = Path(path)
    doc_id = document_id_for(path)
    pages: list[PageText] = []
    doc = fitz.open(path)
    has_tesseract = is_tesseract_available() if ocr_threshold > 0 else False
    try:
        for i, page in enumerate(doc):
            native = clean_text(page.get_text("text"))
            used_ocr = (len(native) < ocr_threshold) and has_tesseract if ocr_threshold > 0 else False
            text = native
            if used_ocr:
                try:
                    pix = page.get_pixmap(dpi=ocr_dpi, alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_result = clean_text(pytesseract.image_to_string(img))
                    if ocr_result:
                        text = ocr_result
                    else:
                        used_ocr = False
                except Exception:
                    used_ocr = False
            pages.append(PageText(doc_id, path.name, i + 1, text, used_ocr))
    finally:
        doc.close()
    return pages


def extract_pdfs_parallel(paths: list[Path], ocr_threshold: int = 0, ocr_dpi: int = 220, max_workers: int = 4) -> dict[Path, list[PageText]]:
    results = {}
    with ThreadPoolExecutor(max_workers=min(len(paths), max_workers or 4)) as executor:
        futures = {executor.submit(extract_pdf, p, ocr_threshold, ocr_dpi): p for p in paths}
        for future in futures:
            p = futures[future]
            try:
                results[p] = future.result()
            except Exception as e:
                results[p] = []
    return results


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]


def chunk_page(page: PageText, size: int = 1000, overlap: int = 200) -> list[Chunk]:
    text = page.text
    if not text:
        return []
    step = max(1, size - overlap)
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        target_end = min(len(text), start + size)
        if target_end < len(text):
            window = text[start:target_end]
            candidates = [m.end() for m in re.finditer(r"[.!?](?:\s|$)", window)]
            if candidates and candidates[-1] > size * 0.55:
                target_end = start + candidates[-1]
        chunk_text = text[start:target_end].strip()
        if chunk_text:
            cid = hashlib.sha256(f"{page.document_id}:{page.page_number}:{start}".encode()).hexdigest()[:20]
            chunks.append(Chunk(cid, page.document_id, page.filename, page.page_number, chunk_text, start, target_end))
        if target_end >= len(text):
            break
        start = max(start + step, target_end - overlap)
    return chunks


def ingest_pdf(path: Path, ocr_threshold=50, ocr_dpi=220, size=1000, overlap=200):
    pages = extract_pdf(path, ocr_threshold, ocr_dpi)
    chunks = [c for p in pages for c in chunk_page(p, size, overlap)]
    return pages, chunks
