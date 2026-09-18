# Project Cleanup Plan (Dry Run)

**Project:** LocalRAG  
**Technology Stack:** Python 3.11, Streamlit, FAISS, BM25, Sentence-Transformers, Cross-Encoder, Ollama  
**Date:** 2026-09-16  

---

## 1. Candidate Classifications

### 🗑️ SAFE TO DELETE (Obsolete & Superseded Scripts)
The following scripts were development prototypes or initial 5-paper variants that have been completely superseded by the unified 20-paper benchmark suite:

| Path | Category | Reason |
| :--- | :--- | :--- |
| `bench/generate_50_queries.py` | `OBSOLETE` | Superseded by `bench/generate_20_doc_50_queries.py` (which spans all 20 PDF documents). |
| `bench/generate_excel_comparison.py` | `OBSOLETE` | Superseded by `bench/run_deep_research_eval.py` (generates the full 4-sheet formatted workbook). |
| `bench/make_eval_dataset.py` | `OBSOLETE` | Early development prototype script for generating test JSON queries. |
| `bench/run_benchmark.py` | `OBSOLETE` | Early 10-query benchmark runner superseded by `run_deep_research_eval.py`. |
| `bench/evaluation.example.json` | `OBSOLETE` | Stale 2-query example JSON file from initial repo scaffold. |
| `data/evaluation_result.json` | `OBSOLETE` | Stale 10-query initial evaluation output. |

---

### 🧹 TEMPORARY & CACHE ARTIFACTS (Safe to Clean)
| Path | Category | Reason |
| :--- | :--- | :--- |
| `src/__pycache__/` | `GENERATED` | Python bytecode cache. |
| `bench/__pycache__/` | `GENERATED` | Python bytecode cache. |
| `tests/__pycache__/` | `GENERATED` | Python bytecode cache. |

---

### 🛡️ KEEP (Protected Core Production & Benchmark Suite)
| Directory / File | Category | Purpose |
| :--- | :--- | :--- |
| `src/*` | `REQUIRED` | Core modules: `ingest.py`, `retrieval.py`, `reranker.py`, `llm.py`, `evaluation.py`, `ragas_eval.py`, `audit.py`, `config.py`. |
| `app/streamlit_app.py` | `REQUIRED` | Primary web UI & Live Evaluation Dashboard. |
| `data/documents/*.pdf` (20 files) | `REQUIRED` | 20 peer-reviewed AI/NLP research papers. |
| `data/indexes/current/*` | `REQUIRED` | Active FAISS FlatIP & BM25 index with 2,131 chunks. |
| `data/evaluation_deep_research_50.json` | `REQUIRED` | Active 50-query benchmark dataset across all 20 papers. |
| `data/retrieval_evaluation_comparison.xlsx` | `REQUIRED` | Formatted 4-sheet evaluation comparison workbook. |
| `data/cache_*.json` | `REQUIRED` | Pre-computed latency, reranker comparison, and ranx IR metrics. |
| `bench/index_all_20_documents.py` | `REQUIRED` | Active automated indexing pipeline for the 20 documents. |
| `bench/generate_20_doc_50_queries.py` | `REQUIRED` | Active 50-query generator across all 20 documents. |
| `bench/run_deep_research_eval.py` | `REQUIRED` | Active deep research evaluation runner & Excel generator. |
| `bench/profile_reranker_latency.py` | `REQUIRED` | Active CPU latency profiler & candidate sweep analyzer. |
| `bench/run_shallow_vs_deep_eval.py` | `REQUIRED` | Active comparative benchmark for Shallow L-2 vs Deep L-6. |
| `bench/run_ragas_and_ranx_benchmark.py`| `REQUIRED` | Active ranx IR benchmark & RAGAS runner. |
| `SYSTEM_DEFENSE_GUIDE.md` | `REQUIRED` | Project defense manual, verbal script, and 14 Q&As. |
| `tests/test_core.py` | `REQUIRED` | Unit tests for chunking, evaluation math, and ranking. |
| `requirements.txt`, `.env*`, `README.md` | `REQUIRED` | Environment configuration and documentation. |
| `exports/`, `models/` | `KEEP` | Runtime output and model directories. |

---

## 2. Estimated Impact
- **Files to remove:** 6 obsolete files + bytecode caches.
- **Space saved:** ~100 KB code redundancy + bytecode.
- **Zero breaking risk:** All references in `app/streamlit_app.py`, `src/`, and `bench/` point exclusively to the active suite.
