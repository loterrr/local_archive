from pathlib import Path
from src.ingest import PageText, chunk_page

def test_chunk_overlap():
    text = ("Sentence one. Sentence two. Sentence three. " * 80).strip()
    p = PageText("doc", "a.pdf", 1, text, False)
    chunks = chunk_page(p, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.text for c in chunks)


def test_repo_files():
    root = Path(__file__).resolve().parents[1]
    assert (root / "app" / "streamlit_app.py").exists()


from src.evaluation import recall_at_k, reciprocal_rank_at_k, evaluate_rankings


def test_recall_and_mrr_at_5():
    ranked = ["a", "b", "c", "d", "e", "f"]
    assert recall_at_k(ranked, ["c"], 5) == 1.0
    assert reciprocal_rank_at_k(ranked, ["c"], 5) == 1 / 3
    assert recall_at_k(ranked, ["f"], 5) == 0.0
    assert reciprocal_rank_at_k(ranked, ["f"], 5) == 0.0


def test_evaluate_rankings_aggregation():
    queries = [
        type("Q", (), {"query_id": "q1", "query": "one", "relevant_chunk_ids": ("b",)})(),
        type("Q", (), {"query_id": "q2", "query": "two", "relevant_chunk_ids": ("z",)})(),
    ]
    result = evaluate_rankings({"q1": ["a", "b"], "q2": ["x", "y"]}, queries, 5)
    assert result["recall_at_k"] == 0.5
    assert result["mrr_at_k"] == 0.25

from src.evaluation import precision_at_k, compare_rankings

def test_precision_at_5():
    assert precision_at_k(["a", "b", "c", "d", "e"], ["b", "e"], 5) == 0.4

def test_reranker_effect_recovery():
    qs=[type("Q",(),{"query_id":"q1","query":"x","relevant_chunk_ids":("z",)})()]
    out=compare_rankings({"q1":["a","b"]},{"q1":["z","a"]},qs,5)
    assert out["recovered_queries"] == 1
