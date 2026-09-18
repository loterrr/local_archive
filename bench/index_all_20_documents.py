import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import Settings
from src.ingest import ingest_pdf
from src.retrieval import HybridIndex

DOCS_DIR = ROOT / "data" / "documents"
INDEX_DIR = ROOT / "data" / "indexes" / "current"

def main():
    settings = Settings()
    pdf_files = sorted(list(DOCS_DIR.glob("*.pdf")))
    print(f"[INFO] Found {len(pdf_files)} PDF documents in {DOCS_DIR}")

    all_chunks = []
    doc_stats = []

    t_start = time.perf_counter()
    for idx, pdf_path in enumerate(pdf_files, 1):
        t0 = time.perf_counter()
        pages, chunks = ingest_pdf(
            pdf_path,
            ocr_threshold=50,
            ocr_dpi=220,
            size=settings.chunk_size,
            overlap=settings.chunk_overlap
        )
        elapsed = time.perf_counter() - t0
        all_chunks.extend(chunks)
        doc_stats.append((pdf_path.name, len(pages), len(chunks), elapsed))
        print(f"  [{idx:02d}/{len(pdf_files)}] {pdf_path.name}: {len(pages)} pages -> {len(chunks)} chunks ({elapsed:.2f}s)")

    print(f"\n[INFO] Extracted {len(all_chunks)} total chunks across {len(pdf_files)} documents.")
    print(f"[INFO] Building HybridIndex with model: {settings.embedding_model}...")

    t_idx_start = time.perf_counter()
    index = HybridIndex(settings.embedding_model, settings.cache_size, settings.cache_ttl)
    
    def on_progress(curr, total):
        if curr % 200 == 0 or curr == total:
            print(f"  Embedding progress: {curr}/{total} ({curr/total*100:.1f}%)")

    index.build(all_chunks, batch_size=32, progress_callback=on_progress)
    index.save(INDEX_DIR)
    
    t_total = time.perf_counter() - t_start
    print(f"\n[DONE] Full 20-document index built and saved to {INDEX_DIR} in {t_total:.2f}s!")
    print(f"Total Chunks: {len(index.chunks)}")

if __name__ == "__main__":
    main()
