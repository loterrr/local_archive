import json
import time
import statistics
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
from src.retrieval import HybridIndex
from src.reranker import CrossEncoderReranker, SHALLOW_RERANKER_MODEL, DEEP_RERANKER_MODEL

INDEX_DIR = Path("data/indexes/current")
OUTPUT_PATH = Path("data/cache_latency_profile.json")

SAMPLE_QUERIES = [
    "What are the key differences between dense and sparse retrieval in RAG?",
    "How does Reciprocal Rank Fusion calculate chunk rank?",
    "What is the role of the Cross-Encoder in multi-stage search pipelines?",
    "How does Self-RAG evaluate retrieval reflection tokens?",
    "What are the limitations of pure vector similarity search when distractor documents exist?",
]

CANDIDATE_SWEEP_KS = [5, 10, 15, 20, 25, 30, 40, 50]

def profile():
    print("[INFO] Starting CPU Latency Profiler...")
    hybrid = HybridIndex.load(INDEX_DIR, "sentence-transformers/all-MiniLM-L6-v2")
    chunks = hybrid.chunks
    print(f"[INFO] Loaded index with {len(chunks)} chunks.")

    reranker_l2 = CrossEncoderReranker(model_name=SHALLOW_RERANKER_MODEL)
    reranker_l6 = CrossEncoderReranker(model_name=DEEP_RERANKER_MODEL)

    # Warmup models
    print("[INFO] Warming up models...")
    _ = reranker_l2.rerank(SAMPLE_QUERIES[0], [(c, 0.5, 1, 1) for c in chunks[:5]])
    _ = reranker_l6.rerank(SAMPLE_QUERIES[0], [(c, 0.5, 1, 1) for c in chunks[:5]])

    sweep_results = []
    
    for cand_k in CANDIDATE_SWEEP_KS:
        print(f"Profiling candidate_k = {cand_k}...")
        faiss_times = []
        bm25_times = []
        rrf_times = []
        l2_times = []
        l6_times = []

        for q in SAMPLE_QUERIES:
            # Profile FAISS dense
            t0 = time.perf_counter()
            q_emb = hybrid.embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True)
            _, dense_ids = hybrid.index.search(np.asarray(q_emb, dtype=np.float32), cand_k)
            t_faiss = (time.perf_counter() - t0) * 1000
            faiss_times.append(t_faiss)

            # Profile BM25 sparse
            t0 = time.perf_counter()
            tokens = [w for w in q.lower().split() if w]
            scores = hybrid.bm25.get_scores(tokens)
            sparse_ids = np.argsort(scores)[::-1][:cand_k]
            t_bm25 = (time.perf_counter() - t0) * 1000
            bm25_times.append(t_bm25)

            # Profile RRF Fusion
            t0 = time.perf_counter()
            dense_rank = {int(idx): r for r, idx in enumerate(dense_ids[0], start=1) if idx >= 0}
            sparse_rank = {int(idx): r for r, idx in enumerate(sparse_ids, start=1)}
            fused = {}
            for idx, r in dense_rank.items():
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + r)
            for idx, r in sparse_rank.items():
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + r)
            best = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:cand_k]
            candidates = [(hybrid.chunks[idx], float(sc), dense_rank.get(idx), sparse_rank.get(idx)) for idx, sc in best]
            t_rrf = (time.perf_counter() - t0) * 1000
            rrf_times.append(t_rrf)

            # Profile L-2 Shallow Rerank
            t0 = time.perf_counter()
            _ = reranker_l2.rerank(q, candidates, top_k=min(5, cand_k))
            t_l2 = (time.perf_counter() - t0) * 1000
            l2_times.append(t_l2)

            # Profile L-6 Deep Rerank
            t0 = time.perf_counter()
            _ = reranker_l6.rerank(q, candidates, top_k=min(5, cand_k))
            t_l6 = (time.perf_counter() - t0) * 1000
            l6_times.append(t_l6)

        avg_faiss = statistics.mean(faiss_times)
        avg_bm25 = statistics.mean(bm25_times)
        avg_rrf = statistics.mean(rrf_times)
        avg_hybrid_total = avg_faiss + avg_bm25 + avg_rrf
        avg_l2 = statistics.mean(l2_times)
        avg_l6 = statistics.mean(l6_times)

        total_l2_pipeline = avg_hybrid_total + avg_l2
        total_l6_pipeline = avg_hybrid_total + avg_l6
        speedup = avg_l6 / avg_l2 if avg_l2 > 0 else 1.0

        sweep_results.append({
            "candidate_k": cand_k,
            "faiss_dense_ms": round(avg_faiss, 2),
            "bm25_sparse_ms": round(avg_bm25, 2),
            "rrf_fusion_ms": round(avg_rrf, 2),
            "hybrid_retrieval_ms": round(avg_hybrid_total, 2),
            "rerank_shallow_l2_ms": round(avg_l2, 2),
            "rerank_deep_l6_ms": round(avg_l6, 2),
            "total_pipeline_l2_ms": round(total_l2_pipeline, 2),
            "total_pipeline_l6_ms": round(total_l6_pipeline, 2),
            "reranker_speedup": round(speedup, 2),
            "pipeline_speedup": round(total_l6_pipeline / total_l2_pipeline, 2),
        })

    profile_summary = {
        "device": "CPU (Local Inference)",
        "chunk_count": len(chunks),
        "models": {
            "shallow_l2": {
                "name": SHALLOW_RERANKER_MODEL,
                "layers": 2,
                "params": "8.5 Million",
                "disk_size_mb": 34.2,
                "recommended_candidate_k": 20,
            },
            "deep_l6": {
                "name": DEEP_RERANKER_MODEL,
                "layers": 6,
                "params": "22.7 Million",
                "disk_size_mb": 87.5,
                "recommended_candidate_k": 20,
            }
        },
        "sweep_data": sweep_results,
        "default_k20_comparison": next((item for item in sweep_results if item["candidate_k"] == 20), sweep_results[3]),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(profile_summary, f, indent=2)

    print(f"[DONE] Profiling complete! Saved results to {OUTPUT_PATH}")

if __name__ == "__main__":
    profile()
