from __future__ import annotations
import os
import sys
from pathlib import Path
import json
import logging
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.models import SentenceBERT
from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def run_beir_benchmark(
    dataset_name: str = "scifact",
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
    data_dir: str = "data/beir",
    batch_size: int = 64
) -> dict:
    """
    Downloads and benchmarks a BEIR dataset with:
    1. Dense Retrieval (all-MiniLM-L6-v2)
    2. Cross-Encoder Reranking (ms-marco-MiniLM-L-2-v2)
    Computes official NDCG@k, MRR@k, MAP@k, and Recall@k.
    """
    out_dir = Path(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Download & extract dataset if not present
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    dataset_folder = out_dir / dataset_name
    if not dataset_folder.exists():
        logging.info(f"Downloading BEIR dataset: {dataset_name} from {url}...")
        zip_file = util.download_and_unzip(url, str(out_dir))
    
    # 2. Load Corpus, Queries, and Qrels (ground truth)
    logging.info(f"Loading {dataset_name} data...")
    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_folder)).load(split="test")
    logging.info(f"Loaded {len(corpus):,} documents, {len(queries):,} queries, {len(qrels):,} judgments.")

    # 3. Step 1: Dense Retrieval Evaluation (Bi-Encoder)
    logging.info(f"Running Dense Retrieval with model: {dense_model_name}")
    model = DRES(SentenceBERT(dense_model_name), batch_size=batch_size)
    retriever = EvaluateRetrieval(model, score_function="cos_sim")
    dense_results = retriever.retrieve(corpus, queries)

    logging.info("Evaluating Dense Retrieval...")
    dense_ndcg, dense_map, dense_recall, dense_precision = retriever.evaluate(qrels, dense_results, retriever.k_values)
    dense_mrr = retriever.evaluate_custom(qrels, dense_results, retriever.k_values, metric="mrr")

    # 4. Step 2: Cross-Encoder Reranker Evaluation
    logging.info(f"Running Cross-Encoder Reranking with model: {reranker_model_name}")
    reranker = Rerank(CrossEncoder(reranker_model_name), batch_size=batch_size)
    rerank_results = reranker.rerank(corpus, queries, dense_results, top_k=50)

    logging.info("Evaluating Reranked Results...")
    rr_ndcg, rr_map, rr_recall, rr_precision = retriever.evaluate(qrels, rerank_results, retriever.k_values)
    rr_mrr = retriever.evaluate_custom(qrels, rerank_results, retriever.k_values, metric="mrr")

    summary = {
        "dataset": dataset_name,
        "corpus_size": len(corpus),
        "queries_count": len(queries),
        "models": {
            "dense_bi_encoder": dense_model_name,
            "cross_encoder_reranker": reranker_model_name
        },
        "metrics": {
            "dense": {
                "NDCG@10": round(dense_ndcg.get("NDCG@10", 0.0), 4),
                "MRR@10": round(dense_mrr.get("MRR@10", 0.0), 4),
                "Recall@10": round(dense_recall.get("Recall@10", 0.0), 4),
                "Recall@100": round(dense_recall.get("Recall@100", 0.0), 4),
                "MAP@10": round(dense_map.get("MAP@10", 0.0), 4)
            },
            "reranked": {
                "NDCG@10": round(rr_ndcg.get("NDCG@10", 0.0), 4),
                "MRR@10": round(rr_mrr.get("MRR@10", 0.0), 4),
                "Recall@10": round(rr_recall.get("Recall@10", 0.0), 4),
                "Recall@100": round(rr_recall.get("Recall@100", 0.0), 4),
                "MAP@10": round(rr_map.get("MAP@10", 0.0), 4)
            },
            "delta": {
                "NDCG@10": round(rr_ndcg.get("NDCG@10", 0.0) - dense_ndcg.get("NDCG@10", 0.0), 4),
                "MRR@10": round(rr_mrr.get("MRR@10", 0.0) - dense_mrr.get("MRR@10", 0.0), 4),
                "Recall@10": round(rr_recall.get("Recall@10", 0.0) - dense_recall.get("Recall@10", 0.0), 4)
            }
        }
    }

    report_path = out_dir / f"beir_{dataset_name}_results.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info(f"Results saved to {report_path}")
    return summary


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    print(f"=== Running BEIR Evaluation on: {dataset} ===")
    results = run_beir_benchmark(dataset_name=dataset)
    print("\n" + "="*50)
    print("BEIR BENCHMARK SUMMARY RESULTS:")
    print("="*50)
    print(json.dumps(results["metrics"], indent=2))
