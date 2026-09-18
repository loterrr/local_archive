from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
import json
from pathlib import Path
import time

@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    query: str
    relevant_chunk_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict, fallback_id: str):
        qid = str(raw.get("query_id", fallback_id))
        query = str(raw.get("query", "")).strip()
        relevant = raw.get("relevant_chunk_ids", [])
        if not isinstance(relevant, list):
            raise ValueError(f"{qid}: relevant_chunk_ids must be a list")
        ids = tuple(str(x) for x in relevant if str(x).strip())
        if not query or not ids:
            raise ValueError(f"{qid}: query and relevant_chunk_ids are required")
        return cls(qid, query, ids)

def load_eval_queries(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        raw_items = [json.loads(x) for x in text.splitlines() if x.strip()]
    else:
        raw = json.loads(text)
        raw_items = raw["queries"] if isinstance(raw, dict) and "queries" in raw else raw
    if not isinstance(raw_items, list):
        raise ValueError("Expected a list or {queries: [...]} ")
    return [EvalQuery.from_dict(x, str(i + 1)) for i, x in enumerate(raw_items)]

def save_eval_template(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"queries": [{"query_id": "q1", "query": "What is a data model?", "relevant_chunk_ids": ["REPLACE_WITH_CHUNK_ID"]}]}, indent=2), encoding="utf-8")

def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    relevant = set(relevant_ids)
    return float(bool(set(ranked_ids[:k]) & relevant)) if relevant else 0.0

def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    relevant = set(relevant_ids)
    top = ranked_ids[:k]
    return sum(1 for x in top if x in relevant) / k if k else 0.0

def reciprocal_rank_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int = 5) -> float:
    relevant = set(relevant_ids)
    for rank, cid in enumerate(ranked_ids[:k], 1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0

def evaluate_rankings(rankings: dict[str, Sequence[str]], queries: Sequence[EvalQuery], k: int = 5) -> dict:
    rows=[]
    for q in queries:
        relevant=set(q.relevant_chunk_ids)
        ranked=list(rankings.get(q.query_id, []))
        first=next((i for i,cid in enumerate(ranked[:k],1) if cid in relevant), None)
        rows.append({"query_id":q.query_id,"query":q.query,"first_relevant_rank":first,
                     "recall_at_k":recall_at_k(ranked,relevant,k),"mrr_at_k":reciprocal_rank_at_k(ranked,relevant,k),
                     "precision_at_k":precision_at_k(ranked,relevant,k)})
    n=len(rows)
    return {"k":k,"num_queries":n,
            "recall_at_k":round(sum(r["recall_at_k"] for r in rows)/n,4) if n else 0,
            "mrr_at_k":round(sum(r["mrr_at_k"] for r in rows)/n,4) if n else 0,
            "precision_at_k":round(sum(r["precision_at_k"] for r in rows)/n,4) if n else 0,
            "details":rows}

def compare_rankings(before: dict[str, Sequence[str]], after: dict[str, Sequence[str]], queries: Sequence[EvalQuery], k: int = 5) -> dict:
    movement=[]
    for q in queries:
        rel=set(q.relevant_chunk_ids)
        br=next((i for i,x in enumerate(before.get(q.query_id,[])[:k],1) if x in rel), None)
        ar=next((i for i,x in enumerate(after.get(q.query_id,[])[:k],1) if x in rel), None)
        movement.append({"query_id":q.query_id,"before_rank":br,"after_rank":ar,
                         "improved": br is not None and ar is not None and ar < br,
                         "recovered": br is None and ar is not None})
    return {"improved_queries":sum(x["improved"] for x in movement),
            "recovered_queries":sum(x["recovered"] for x in movement),"details":movement}

def evaluate_pipeline(index, queries, reranker=None, k=5, dense_k=32, rrf_k=60, reranker_candidates=50, batch_size=32):
    hybrid_rankings={}; reranked_rankings={}; timings=[]
    for q in queries:
        t=time.perf_counter()
        candidates=index.search(q.query,dense_k=dense_k,final_k=k,rrf_k=rrf_k,candidate_k=max(k,reranker_candidates if reranker else k))
        hybrid_rankings[q.query_id]=[c.chunk_id for c,*_ in candidates]
        if reranker:
            rr=reranker.rerank(q.query,candidates,top_k=k,batch_size=batch_size)
            reranked_rankings[q.query_id]=[x.chunk.chunk_id for x in rr]
        timings.append((time.perf_counter()-t)*1000)
    result={"config":{"k":k,"dense_k":dense_k,"rrf_k":rrf_k,"reranker_candidates":reranker_candidates},
            "hybrid":evaluate_rankings(hybrid_rankings,queries,k),
            "timing":{"queries":len(queries),"avg_query_ms":round(sum(timings)/len(timings),2) if timings else 0,
                      "total_ms":round(sum(timings),2)}}
    if reranker:
        result["reranked"]=evaluate_rankings(reranked_rankings,queries,k)
        result["reranker_effect"]=compare_rankings(hybrid_rankings,reranked_rankings,queries,k)
        result["reranker_metrics"]=reranker.metrics()
    return result

def results_to_json(results: dict) -> str:
    return json.dumps(results, indent=2, ensure_ascii=False)
