"""
SciFact Claim Verifier & RAG Synthesizer
Uses the BEIR SciFact corpus (5,183 peer-reviewed biomedical abstracts),
retrieves relevant evidence using dense embeddings + cross-encoder reranker,
and uses local Ollama (phi3.5/qwen2.5) to classify and explain the claim:
  - SUPPORT
  - CONTRADICT
  - NOT ENOUGH INFO
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from .download_models import ensure_model_ready

DATA_DIR = Path("data/beir/scifact")
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
QRELS_PATH = DATA_DIR / "qrels" / "test.tsv"

class SciFactVerifier:
    def __init__(self, dense_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
                 ollama_model: str = "phi3.5:latest"):
        self.ollama_model = ollama_model
        print(f"[*] Loading SciFact corpus from {CORPUS_PATH}...")
        self.corpus = {}
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.corpus[item["_id"]] = {
                    "title": item.get("title", ""),
                    "text": item.get("text", "")
                }
        self.doc_ids = list(self.corpus.keys())
        print(f"[+] Loaded {len(self.corpus):,} abstracts.")

        print(f"[*] Loading dense retriever: {dense_model}...")
        dense_path, dense_local = ensure_model_ready(dense_model)
        if dense_local:
            try:
                self.bi_encoder = SentenceTransformer(dense_path, local_files_only=True)
            except TypeError:
                self.bi_encoder = SentenceTransformer(dense_path, model_kwargs={"local_files_only": True})
        else:
            self.bi_encoder = SentenceTransformer(dense_path)
        
        print(f"[*] Loading cross-encoder reranker: {rerank_model}...")
        rerank_path, rerank_local = ensure_model_ready(rerank_model)
        if rerank_local:
            try:
                self.cross_encoder = CrossEncoder(rerank_path, local_files_only=True)
            except TypeError:
                self.cross_encoder = CrossEncoder(rerank_path, automodel_args={"local_files_only": True})
        else:
            self.cross_encoder = CrossEncoder(rerank_path)

        
        # Precompute or load corpus embeddings if cached
        emb_cache = DATA_DIR / "corpus_emb.npy"
        if emb_cache.exists():
            print(f"[*] Loading precomputed embeddings from {emb_cache}...")
            self.corpus_embeddings = np.load(emb_cache)
        else:
            print(f"[*] Encoding {len(self.doc_ids):,} abstracts (this happens once)...")
            texts = [f"{self.corpus[did]['title']} {self.corpus[did]['text']}" for did in self.doc_ids]
            self.corpus_embeddings = self.bi_encoder.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
            np.save(emb_cache, self.corpus_embeddings)
            print(f"[+] Cached corpus embeddings to {emb_cache}")

    def retrieve(self, claim: str, top_k_dense: int = 30, top_k_rerank: int = 5):
        """Retrieve and rerank top abstracts for a claim."""
        q_emb = self.bi_encoder.encode([claim], normalize_embeddings=True)
        # Cosine similarities
        scores = np.dot(self.corpus_embeddings, q_emb.T).squeeze()
        top_indices = np.argsort(-scores)[:top_k_dense]
        
        candidates = []
        pairs = []
        for idx in top_indices:
            did = self.doc_ids[idx]
            doc = self.corpus[did]
            full_text = f"{doc['title']}. {doc['text']}"
            candidates.append((did, doc, float(scores[idx])))
            pairs.append((claim, full_text))
            
        # Cross-encoder reranking
        rerank_scores = self.cross_encoder.predict(pairs)
        reranked_indices = np.argsort(-rerank_scores)[:top_k_rerank]
        
        results = []
        for r_idx in reranked_indices:
            did, doc, bi_score = candidates[r_idx]
            results.append({
                "doc_id": did,
                "title": doc["title"],
                "text": doc["text"],
                "rerank_score": float(rerank_scores[r_idx]),
                "bi_score": bi_score
            })
        return results

    def verify_claim(self, claim: str) -> dict:
        """Retrieve evidence and use LLM to verify claim."""
        evidence_docs = self.retrieve(claim, top_k_dense=30, top_k_rerank=3)
        
        context_str = ""
        for i, ev in enumerate(evidence_docs, 1):
            context_str += f"[Evidence {i}] Title: {ev['title']}\nAbstract: {ev['text']}\n\n"

        prompt = f"""You are an expert scientific peer reviewer.
Analyze the following scientific CLAIM based strictly on the provided EVIDENCE ABSTRACTS.

CLAIM:
"{claim}"

EVIDENCE ABSTRACTS:
{context_str}

Determine whether the evidence SUPPORTS, CONTRADICTS, or is NOT ENOUGH INFO to verify the claim.
Format your response as:
VERDICT: [SUPPORT | CONTRADICT | NOT ENOUGH INFO]
RATIONALE: (Explain why in 2-3 concise sentences citing Evidence 1, 2, or 3)
KEY_FINDING: (1 sentence summary of the scientific consensus from the abstracts)
"""
        try:
            resp = requests.post(
                "http://127.0.0.1:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 300}
                },
                timeout=45
            )
            llm_text = resp.json().get("response", "").strip() if resp.status_code == 200 else "Error contacting LLM"
        except Exception as e:
            llm_text = f"LLM error: {e}"

        return {
            "claim": claim,
            "evidence": evidence_docs,
            "verification": llm_text
        }

if __name__ == "__main__":
    verifier = SciFactVerifier()
    
    sample_claims = [
        "1 in 5 million in UK have abnormal PrP positivity.",
        "40mg/day dosage of folic acid and 2mg/day dosage of vitamin B12 does not affect chronic kidney disease (CKD) progression.",
        "0-dimensional biomaterials lack inductive properties."
    ]
    
    query = sys.argv[1] if len(sys.argv) > 1 else sample_claims[0]
    print(f"\n==================================================")
    print(f"VERIFYING CLAIM: {query}")
    print(f"==================================================")
    result = verifier.verify_claim(query)
    
    print("\n--- TOP RETRIEVED EVIDENCE ---")
    for i, ev in enumerate(result["evidence"], 1):
        print(f"\n[#{i}] Doc ID: {ev['doc_id']} (Score: {ev['rerank_score']:.4f})")
        print(f"Title: {ev['title']}")
        print(f"Snippet: {ev['text'][:200]}...")
        
    print("\n--- LLM VERDICT & SYNTHESIS ---")
    print(result["verification"])
