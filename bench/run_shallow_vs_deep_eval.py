import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import Settings
from src.evaluation import recall_at_k, precision_at_k, reciprocal_rank_at_k
from src.retrieval import HybridIndex
from src.reranker import CrossEncoderReranker, SHALLOW_RERANKER_MODEL, DEEP_RERANKER_MODEL

def evaluate_rank_list(rankings: dict[str, Sequence[str]], queries: list[dict], k: int = 5) -> dict:
    rows = []
    for q in queries:
        qid = q["query_id"]
        rel = set(q["relevant_chunk_ids"])
        ranked = list(rankings.get(qid, []))
        first = next((i for i, cid in enumerate(ranked[:k], 1) if cid in rel), None)
        rows.append({
            "query_id": qid,
            "category": q.get("category", "General"),
            "query": q["query"],
            "first_relevant_rank": first,
            "recall_at_k": recall_at_k(ranked, rel, k),
            "mrr_at_k": reciprocal_rank_at_k(ranked, rel, k),
            "precision_at_k": precision_at_k(ranked, rel, k),
        })
    n = len(rows)
    return {
        "k": k,
        "num_queries": n,
        "recall_at_k": round(sum(r["recall_at_k"] for r in rows) / n, 4) if n else 0.0,
        "mrr_at_k": round(sum(r["mrr_at_k"] for r in rows) / n, 4) if n else 0.0,
        "precision_at_k": round(sum(r["precision_at_k"] for r in rows) / n, 4) if n else 0.0,
        "details": rows,
    }

def run():
    print("[INFO] Starting Shallow (L-2) vs Deep (L-6) Comparative Benchmark...")
    settings = Settings()
    index_dir = ROOT / "data" / "indexes" / "current"
    dataset_file = ROOT / "data" / "evaluation_deep_research_50.json"
    output_cache = ROOT / "data" / "cache_reranker_comparison.json"

    index = HybridIndex.load(index_dir, settings.embedding_model)
    queries = json.loads(dataset_file.read_text(encoding="utf-8"))["queries"]
    
    reranker_l2 = CrossEncoderReranker(SHALLOW_RERANKER_MODEL)
    reranker_l6 = CrossEncoderReranker(DEEP_RERANKER_MODEL)

    k_candidates = 20
    k_final = 10

    l2_rankings = {}
    l6_rankings = {}
    hybrid_rankings = {}
    l2_timings = []
    l6_timings = []
    hybrid_timings = []

    print(f"[INFO] Evaluating {len(queries)} queries with candidate_k={k_candidates}...")
    for idx, q in enumerate(queries, 1):
        # 1. Hybrid Search
        t0 = time.perf_counter()
        candidates = index.search(
            q["query"],
            dense_k=settings.dense_k,
            final_k=k_final,
            rrf_k=settings.rrf_k,
            candidate_k=k_candidates
        )
        t_hy = (time.perf_counter() - t0) * 1000
        hybrid_timings.append(t_hy)
        hybrid_rankings[q["query_id"]] = [c.chunk_id for c, *_ in candidates]

        # 2. Shallow L-2 Rerank
        t0 = time.perf_counter()
        rr_l2 = reranker_l2.rerank(q["query"], candidates, top_k=k_final, batch_size=settings.reranker_batch_size)
        t_l2 = (time.perf_counter() - t0) * 1000
        l2_timings.append(t_l2)
        l2_rankings[q["query_id"]] = [x.chunk.chunk_id for x in rr_l2]

        # 3. Deep L-6 Rerank
        t0 = time.perf_counter()
        rr_l6 = reranker_l6.rerank(q["query"], candidates, top_k=k_final, batch_size=settings.reranker_batch_size)
        t_l6 = (time.perf_counter() - t0) * 1000
        l6_timings.append(t_l6)
        l6_rankings[q["query_id"]] = [x.chunk.chunk_id for x in rr_l6]

        if idx % 10 == 0 or idx == len(queries):
            print(f"  Processed {idx}/{len(queries)}: Hybrid={t_hy:.1f}ms | L-2={t_l2:.1f}ms | L-6={t_l6:.1f}ms")

    # Multi-K evaluation
    k_vals = [1, 5, 10, 20]
    multi_k = {}
    for k in k_vals:
        multi_k[k] = {
            "hybrid": evaluate_rank_list(hybrid_rankings, queries, k=k),
            "shallow_l2": evaluate_rank_list(l2_rankings, queries, k=k),
            "deep_l6": evaluate_rank_list(l6_rankings, queries, k=k),
        }

    avg_hy_lat = round(sum(hybrid_timings) / len(hybrid_timings), 2)
    avg_l2_lat = round(sum(l2_timings) / len(l2_timings), 2)
    avg_l6_lat = round(sum(l6_timings) / len(l6_timings), 2)

    result = {
        "candidate_k": k_candidates,
        "final_k": k_final,
        "query_count": len(queries),
        "latencies": {
            "hybrid_retrieval_avg_ms": avg_hy_lat,
            "shallow_l2_rerank_avg_ms": avg_l2_lat,
            "deep_l6_rerank_avg_ms": avg_l6_lat,
            "shallow_l2_total_pipeline_ms": round(avg_hy_lat + avg_l2_lat, 2),
            "deep_l6_total_pipeline_ms": round(avg_hy_lat + avg_l6_lat, 2),
            "reranker_speedup": round(avg_l6_lat / avg_l2_lat, 2) if avg_l2_lat > 0 else 1.0,
            "total_speedup": round((avg_hy_lat + avg_l6_lat) / (avg_hy_lat + avg_l2_lat), 2),
        },
        "multi_k_summary": {
            k: {
                "hybrid_mrr": multi_k[k]["hybrid"]["mrr_at_k"],
                "hybrid_recall": multi_k[k]["hybrid"]["recall_at_k"],
                "shallow_l2_mrr": multi_k[k]["shallow_l2"]["mrr_at_k"],
                "shallow_l2_recall": multi_k[k]["shallow_l2"]["recall_at_k"],
                "deep_l6_mrr": multi_k[k]["deep_l6"]["mrr_at_k"],
                "deep_l6_recall": multi_k[k]["deep_l6"]["recall_at_k"],
            }
            for k in k_vals
        },
        "multi_k_detailed": multi_k,
        "telemetry": {
            "shallow_l2": reranker_l2.metrics(),
            "deep_l6": reranker_l6.metrics(),
        }
    }

    output_cache.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[DONE] Comparative evaluation complete! Results saved to {output_cache}")

if __name__ == "__main__":
    run()
