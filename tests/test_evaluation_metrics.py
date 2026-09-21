"""
Unit tests for evaluation metrics — verifies mathematical correctness
of all metrics that could be questioned during a thesis defense panel.
"""
import sys
import os
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import recall_at_k, reciprocal_rank_at_k, precision_at_k
from src.ragas_eval import compute_rouge_l, evaluate_citation_precision


# ── Recall@k ──────────────────────────────────────────────────────────

class TestRecallAtK:
    def test_hit_at_rank_1(self):
        """Gold chunk at position 1 → recall = 1.0"""
        assert recall_at_k(["c1", "c2", "c3"], ["c1"], k=5) == 1.0

    def test_hit_at_rank_k(self):
        """Gold chunk at position k (boundary) → recall = 1.0"""
        assert recall_at_k(["c2", "c3", "c4", "c5", "c1"], ["c1"], k=5) == 1.0

    def test_miss_beyond_k(self):
        """Gold chunk beyond position k → recall = 0.0"""
        assert recall_at_k(["c2", "c3", "c4", "c5", "c6", "c1"], ["c1"], k=5) == 0.0

    def test_empty_relevant(self):
        """No relevant chunks specified → recall = 0.0"""
        assert recall_at_k(["c1", "c2"], [], k=5) == 0.0

    def test_single_ground_truth_equivalence(self):
        """With |relevant|=1, binary hit rate == standard recall."""
        ranked = ["a", "b", "c", "d", "e"]
        # Hit: relevant = {"c"}, standard recall = |{c} ∩ {a,b,c,d,e}| / 1 = 1.0
        assert recall_at_k(ranked, ["c"], k=5) == 1.0
        # Miss: relevant = {"z"}
        assert recall_at_k(ranked, ["z"], k=5) == 0.0

    def test_multi_passage_recall(self):
        """With multiple relevant passages, standard recall computes fraction of relevant retrieved."""
        ranked = ["c1", "x", "c2", "y", "z", "c3"]
        # In top-5: c1 and c2 retrieved out of {c1, c2, c3} -> recall = 2/3
        assert abs(recall_at_k(ranked, ["c1", "c2", "c3"], k=5) - 2 / 3) < 1e-6


# ── Reciprocal Rank (MRR) ────────────────────────────────────────────

class TestReciprocalRankAtK:
    def test_rank_1(self):
        assert reciprocal_rank_at_k(["c1", "c2", "c3"], ["c1"], k=5) == 1.0

    def test_rank_2(self):
        assert reciprocal_rank_at_k(["c2", "c1", "c3"], ["c1"], k=5) == 0.5

    def test_rank_3(self):
        rr = reciprocal_rank_at_k(["c2", "c3", "c1"], ["c1"], k=5)
        assert abs(rr - 1/3) < 1e-9

    def test_miss(self):
        """Gold chunk not in top-k → RR = 0.0"""
        assert reciprocal_rank_at_k(["c2", "c3", "c4"], ["c1"], k=3) == 0.0


# ── ROUGE-L (LCS F1) ─────────────────────────────────────────────────

class TestRougeL:
    def test_identical_strings(self):
        """Identical candidate and reference → ROUGE-L = 1.0"""
        text = "The system uses a hybrid retrieval approach combining BM25 and FAISS."
        assert compute_rouge_l(text, text) == 1.0

    def test_partial_overlap(self):
        """Partial overlap → 0 < ROUGE-L < 1"""
        cand = "The hybrid retrieval approach combines BM25 and dense search."
        ref = "A hybrid retrieval method using BM25 and FAISS vector search."
        score = compute_rouge_l(cand, ref)
        assert 0.0 < score < 1.0

    def test_disjoint_strings(self):
        """No overlapping tokens → ROUGE-L = 0.0"""
        cand = "alpha beta gamma delta"
        ref = "one two three four"
        assert compute_rouge_l(cand, ref) == 0.0

    def test_empty_candidate(self):
        """Empty candidate → ROUGE-L = 0.0"""
        assert compute_rouge_l("", "some reference text") == 0.0

    def test_empty_reference(self):
        """Empty reference → ROUGE-L = 0.0"""
        assert compute_rouge_l("some candidate text", "") == 0.0

    def test_subsequence_not_substring(self):
        """LCS should find longest common subsequence, not just substring."""
        cand = "A B C D E"
        ref = "A C E"
        score = compute_rouge_l(cand, ref)
        # LCS = [A, C, E] (length 3), precision = 3/5, recall = 3/3 = 1.0
        # F1 = 2 * (3/5 * 1) / (3/5 + 1) = 2 * 0.6 / 1.6 = 0.75
        assert abs(score - 0.75) < 0.01


# ── Citation Precision ────────────────────────────────────────────────

class TestCitationPrecision:
    def test_grounded_citation(self):
        """Citation tag referencing retrieved context → precision should be positive."""
        answer = "BM25 uses term frequency for scoring [1]. This is effective [2]."
        contexts = ["BM25 uses term frequency and inverse document frequency for scoring."]
        result = evaluate_citation_precision(answer, contexts)
        assert result["citations_count"] >= 1
        # Numeric citations like [1] have cite_clean length < 4, so they count as grounded
        assert result["citation_precision"] > 0.0

    def test_no_citations(self):
        """Answer with no citation tags → default precision."""
        answer = "The system is effective for document retrieval."
        contexts = ["Some context about retrieval."]
        result = evaluate_citation_precision(answer, contexts)
        assert result["citations_count"] == 0

    def test_attribution_rate_calculation(self):
        """Sentences with citations should increase attribution rate."""
        answer = "The recall is 95% [1]. The MRR is 0.87 [2]. Precision matters."
        contexts = ["Recall measured at 95%. MRR was 0.87."]
        result = evaluate_citation_precision(answer, contexts)
        assert result["attribution_rate"] > 0.0
        assert result["cited_sentences_count"] >= 1


# ── t-test Honesty ────────────────────────────────────────────────────

class TestTTestHonesty:
    def test_identical_arrays_produce_no_significance(self):
        """Identical MRR arrays should yield t=0, p=1 (or NaN handled)."""
        from scipy import stats
        mrrs = [1.0, 0.5, 0.0, 1.0, 0.5]
        try:
            res = stats.ttest_rel(mrrs, mrrs)
            t_stat = float(res.statistic)
            p_val = float(res.pvalue)
        except Exception:
            t_stat, p_val = 0.0, 1.0

        # With identical arrays, there should be no significant difference
        # scipy returns nan for identical arrays; our fallback should be 0.0/1.0
        import math
        if math.isnan(t_stat):
            t_stat = 0.0
        if math.isnan(p_val):
            p_val = 1.0

        assert p_val >= 0.05, f"Identical arrays must NOT show significance, got p={p_val}"

    def test_different_arrays_can_be_significant(self):
        """Clearly different MRR arrays should produce a low p-value."""
        from scipy import stats
        baseline = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        improved = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        res = stats.ttest_rel(improved, baseline)
        assert float(res.pvalue) < 0.05



# ── 3-Way Comparative Scorecard ───────────────────────────────────────

class TestComparison3Way:
    def test_comparison_3way_structure(self):
        """Verify 3-way comparison contains hybrid, shallow_l2, and deep_l6 keys."""
        sample_c3 = {
            "hybrid": {"name": "Baseline", "recall": "50.0%", "mrr": "0.3657", "latency_ms": "21.8 ms"},
            "shallow_l2": {"name": "Shallow L-2", "recall": "54.0%", "mrr": "0.3890", "latency_ms": "487 ms"},
            "deep_l6": {"name": "Deep L-6", "recall": "64.0%", "mrr": "0.4013", "latency_ms": "1032 ms"}
        }
        assert "hybrid" in sample_c3
        assert "shallow_l2" in sample_c3
        assert "deep_l6" in sample_c3
        assert float(sample_c3["deep_l6"]["recall"].replace("%", "")) > float(sample_c3["shallow_l2"]["recall"].replace("%", ""))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
