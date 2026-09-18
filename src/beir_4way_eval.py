"""
Comprehensive 4-Way BEIR Retrieval Benchmark:
1. Sparse (BM25Okapi)
2. Dense Bi-Encoder (all-MiniLM-L6-v2)
3. Hybrid RRF (Dense + BM25 Fusion)
4. Enhanced (Hybrid + Cross-Encoder Reranking ms-marco-MiniLM-L-2-v2)
"""

import json
import re
import time
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer, CrossEncoder

DATA_DIR = Path("data/beir/scifact")

def run_4way_benchmark(top_k_rerank: int = 30):
    print("Loading SciFact dataset...")
    corpus, queries, qrels = GenericDataLoader(data_folder=str(DATA_DIR)).load(split="test")
    doc_ids = list(corpus.keys())
    q_ids = list(queries.keys())
    print(f"Loaded {len(corpus):,} docs, {len(queries):,} test queries, {len(qrels):,} qrels.")

    # 1. BM25 Sparse Indexing & Retrieval
    print("\n--- 1. Evaluating Sparse BM25 Retrieval ---")
    doc_texts = [f"{corpus[did].get('title', '')} {corpus[did].get('text', '')}" for did in doc_ids]
    tokenized_corpus = [re.findall(r"\w+", t.lower()) for t in doc_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_results = {}
    for qid in q_ids:
        q_tokens = re.findall(r"\w+", queries[qid].lower())
        scores = bm25.get_scores(q_tokens)
        top_indices = np.argsort(-scores)[:100]
        bm25_results[qid] = {doc_ids[idx]: float(scores[idx]) for idx in top_indices if scores[idx] > 0}

    # 2. Dense Retrieval (all-MiniLM-L6-v2)
    print("\n--- 2. Evaluating Dense Bi-Encoder Retrieval ---")
    bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb_cache = DATA_DIR / "corpus_emb.npy"
    if emb_cache.exists():
        corpus_emb = np.load(emb_cache)
    else:
        corpus_emb = bi_encoder.encode(doc_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        np.save(emb_cache, corpus_emb)

    q_texts = [queries[qid] for qid in q_ids]
    q_embs = bi_encoder.encode(q_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    
    dense_results = {}
    for i, qid in enumerate(q_ids):
        cos_scores = np.dot(corpus_emb, q_embs[i])
        top_indices = np.argsort(-cos_scores)[:100]
        dense_results[qid] = {doc_ids[idx]: float(cos_scores[idx]) for idx in top_indices}

    # 3. Hybrid Retrieval (Dense + BM25 RRF Fusion)
    print("\n--- 3. Evaluating Hybrid RRF (Dense + BM25) ---")
    rrf_k = 60
    hybrid_results = {}
    for qid in q_ids:
        fused = {}
        # Dense ranks
        d_items = sorted(dense_results[qid].items(), key=lambda x: x[1], reverse=True)
        for rank, (did, _) in enumerate(d_items, 1):
            fused[did] = fused.get(did, 0.0) + (1.0 / (rrf_k + rank))
        # BM25 ranks
        b_items = sorted(bm25_results.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
        for rank, (did, _) in enumerate(b_items, 1):
            fused[did] = fused.get(did, 0.0) + (1.0 / (rrf_k + rank))
        
        sorted_hybrid = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:100]
        hybrid_results[qid] = {did: score for did, score in sorted_hybrid}

    # 4. Enhanced Retrieval (Hybrid + Cross-Encoder Reranking - Fast Batched)
    print(f"\n--- 4. Evaluating Enhanced (Hybrid + Cross-Encoder ms-marco-MiniLM-L-2-v2, top_{top_k_rerank}) ---")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2", max_length=256)
    
    all_pairs = []
    query_cand_map = [] # (qid, list_of_dids, original_hybrid_ranks)
    
    for qid in q_ids:
        query_text = queries[qid]
        all_hybrid_items = sorted(hybrid_results[qid].items(), key=lambda x: x[1], reverse=True)
        cand_items = all_hybrid_items[:top_k_rerank]
        remaining_items = all_hybrid_items[top_k_rerank:100]
        
        dids = [did for did, _ in cand_items]
        for did in dids:
            all_pairs.append((query_text, f"{corpus[did].get('title', '')}. {corpus[did].get('text', '')}"))
        query_cand_map.append((qid, dids, remaining_items))

    print(f"Reranking {len(all_pairs):,} query-candidate pairs with max_length=256, batch_size=256...")
    all_scores = cross_encoder.predict(all_pairs, batch_size=256, show_progress_bar=True)
    
    enhanced_results = {}
    ptr = 0
    for qid, dids, remaining_items in query_cand_map:
        count = len(dids)
        q_scores = all_scores[ptr : ptr + count]
        ptr += count
        ranked = sorted(zip(dids, q_scores), key=lambda x: x[1], reverse=True)
        
        # Enhanced result combines reranked top items + tail items
        res_dict = {}
        # Min rerank score for scaling tail items below top-k
        min_rerank_score = float(min(q_scores)) if len(q_scores) > 0 else 0.0
        for did, s in ranked:
            res_dict[did] = float(s)
        for rank_offset, (did, _) in enumerate(remaining_items, start=1):
            res_dict[did] = min_rerank_score - rank_offset * 0.01
            
        enhanced_results[qid] = res_dict

    # Metrics Evaluation via BEIR EvaluateRetrieval
    evaluator = EvaluateRetrieval()
    k_vals = [1, 3, 5, 10, 100]
    
    pipelines = {
        "BM25 (Sparse)": bm25_results,
        "Dense (all-MiniLM-L6-v2)": dense_results,
        "Hybrid (BM25 + Dense RRF)": hybrid_results,
        "Enhanced (Hybrid + Cross-Encoder)": enhanced_results
    }
    
    summary = {}
    for name, res in pipelines.items():
        ndcg, map_m, recall, prec = evaluator.evaluate(qrels, res, k_vals)
        mrr = evaluator.evaluate_custom(qrels, res, k_vals, metric="mrr")
        summary[name] = {
            "NDCG@10": round(ndcg.get("NDCG@10", 0.0), 4),
            "MRR@10": round(mrr.get("MRR@10", 0.0), 4),
            "Recall@10": round(recall.get("Recall@10", 0.0), 4),
            "Recall@100": round(recall.get("Recall@100", 0.0), 4),
            "MAP@10": round(map_m.get("MAP@10", 0.0), 4)
        }

    out_file = Path("data/beir/beir_4way_results.json")
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[+] Results saved to {out_file}")
    return summary

if __name__ == "__main__":
    res = run_4way_benchmark()
    print("\n" + "="*70)
    print("4-WAY RETRIEVAL COMPARISON ON SCIFACT (300 Claims, 5,183 Abstracts):")
    print("="*70)
    print(json.dumps(res, indent=2))
