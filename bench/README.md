# LocalRAG Benchmarking

Use stable `chunk_id` values from `data/indexes/current/chunks.json` to define relevance labels.

Example:

```bash
python -m bench.run_benchmark --index data/indexes/current --eval bench/evaluation.example.json
```

The result contains:

- Recall@K
- MRR@K
- Precision@K
- average query latency
- reranker latency and candidate count
- number of queries whose relevant result moved upward
- number of queries recovered by reranking

Use `--no-reranker` to generate a hybrid-only baseline for an A/B comparison.
