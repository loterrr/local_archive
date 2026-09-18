from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from collections import OrderedDict
from threading import RLock
import hashlib
import json
import pickle
import re
import time
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from .ingest import Chunk


class QueryCache:
    def __init__(self, max_size: int = 128, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    def _key(self, query: str, dense_k: int, candidate_k: int, rrf_k: int) -> str:
        raw = f"{query.strip().lower()}|{dense_k}|{candidate_k}|{rrf_k}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def get(self, query: str, dense_k: int, candidate_k: int, rrf_k: int):
        key = self._key(query, dense_k, candidate_k, rrf_k)
        with self._lock:
            item = self._cache.get(key)
            if not item:
                self.misses += 1
                return None
            created, value = item
            if time.time() - created >= self.ttl_seconds:
                self._cache.pop(key, None)
                self.misses += 1
                return None
            self._cache.move_to_end(key)
            self.hits += 1
            return value

    def put(self, query: str, dense_k: int, candidate_k: int, rrf_k: int, value) -> None:
        key = self._key(query, dense_k, candidate_k, rrf_k)
        with self._lock:
            self._cache[key] = (time.time(), value)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


class HybridIndex:
    def __init__(self, model_name: str, cache_size: int = 128, cache_ttl: int = 300):
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray | None = None
        self.index = None
        self.bm25 = None
        self.cache = QueryCache(cache_size, cache_ttl)
        self.stats = {"searches": 0, "cache_hits": 0, "total_latency_ms": 0.0}

    def build(self, chunks: list[Chunk], batch_size: int = 64, progress_callback=None):
        # Content-based deduplication makes repeated uploads / duplicate pages harmless.
        unique = {}
        for c in chunks:
            unique[c.chunk_id] = c
        self.chunks = list(unique.values())
        texts = [c.text for c in self.chunks]
        if not texts:
            raise ValueError("No chunks to index")

        all_embeddings = []
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            emb = self.embedder.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
            all_embeddings.append(emb)
            if progress_callback:
                progress_callback(min(i + len(batch), total), total)

        embeddings = np.vstack(all_embeddings)
        self.embeddings = embeddings
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(np.asarray(embeddings, dtype=np.float32))
        tokenized = [re.findall(r"\w+", t.lower()) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.cache.clear()
        return embeddings

    def add_chunks(self, new_chunks: list[Chunk], batch_size: int = 64, progress_callback=None):
        """Incrementally index new chunks without re-embedding existing chunks."""
        if self.index is None or not self.chunks:
            return self.build(new_chunks, batch_size=batch_size, progress_callback=progress_callback)

        existing_ids = {c.chunk_id for c in self.chunks}
        # Deduplicate and filter out already indexed chunks
        to_add = []
        seen = set()
        for c in new_chunks:
            if c.chunk_id not in existing_ids and c.chunk_id not in seen:
                seen.add(c.chunk_id)
                to_add.append(c)

        if not to_add:
            return self.embeddings

        texts = [c.text for c in to_add]
        all_embeddings = []
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            emb = self.embedder.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=batch_size
            )
            all_embeddings.append(emb)
            if progress_callback:
                progress_callback(min(i + len(batch), total), total)

        new_embeddings = np.vstack(all_embeddings)
        self.index.add(np.asarray(new_embeddings, dtype=np.float32))
        self.chunks.extend(to_add)

        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        else:
            self.embeddings = new_embeddings

        # Rapidly rebuild BM25 inverted lexical index (<30ms)
        tokenized = [re.findall(r"\w+", c.text.lower()) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.cache.clear()
        return self.embeddings

    def remove_document(self, filename: str):
        """Remove all chunks belonging to a filename without re-embedding the entire corpus."""
        remaining_indices = [i for i, c in enumerate(self.chunks) if c.filename != filename]
        if len(remaining_indices) == len(self.chunks):
            return

        if not remaining_indices:
            self.chunks = []
            self.index = None
            self.bm25 = None
            self.embeddings = None
            self.cache.clear()
            return

        self.chunks = [self.chunks[i] for i in remaining_indices]
        if self.embeddings is not None and len(self.embeddings) > len(remaining_indices):
            self.embeddings = self.embeddings[remaining_indices]
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(np.asarray(self.embeddings, dtype=np.float32))
        else:
            # Rebuild index if embeddings array wasn't cached
            texts = [c.text for c in self.chunks]
            self.embeddings = self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(np.asarray(self.embeddings, dtype=np.float32))

        tokenized = [re.findall(r"\w+", c.text.lower()) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        self.cache.clear()

    def search(self, query: str, dense_k=32, final_k=8, rrf_k=60, candidate_k: int | None = None):
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Index not built")
        if not query or not query.strip():
            return []
        started = time.perf_counter()
        retrieval_k = min(max(final_k, candidate_k or final_k), len(self.chunks))
        cached = self.cache.get(query, dense_k, retrieval_k, rrf_k)
        if cached is not None:
            self.stats["searches"] += 1
            self.stats["cache_hits"] += 1
            self.stats["total_latency_ms"] += (time.perf_counter() - started) * 1000
            return cached

        q = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        actual_k = min(dense_k, len(self.chunks))
        _, dense_ids = self.index.search(np.asarray(q, dtype=np.float32), actual_k)
        dense_rank = {int(idx): rank for rank, idx in enumerate(dense_ids[0], start=1) if idx >= 0}

        tokens = re.findall(r"\w+", query.lower())
        scores = self.bm25.get_scores(tokens)
        sparse_ids = np.argsort(scores)[::-1][:actual_k]
        sparse_rank = {int(idx): rank for rank, idx in enumerate(sparse_ids, start=1)}

        fused = {}
        for idx, rank in dense_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)
        for idx, rank in sparse_rank.items():
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank)

        best = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:retrieval_k]
        result = [(self.chunks[idx], float(score), dense_rank.get(idx), sparse_rank.get(idx)) for idx, score in best]
        self.cache.put(query, dense_k, retrieval_k, rrf_k, result)
        elapsed = (time.perf_counter() - started) * 1000
        self.stats["searches"] += 1
        self.stats["total_latency_ms"] += elapsed
        return result

    def metrics(self) -> dict:
        avg = self.stats["total_latency_ms"] / self.stats["searches"] if self.stats["searches"] else 0.0
        return {**self.stats, "avg_latency_ms": round(avg, 2), "cache": self.cache.stats()}

    def save(self, directory: Path | str):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(directory / "faiss.index"))
        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False, indent=2)
        if self.bm25 is not None:
            with open(directory / "bm25.pkl", "wb") as f:
                pickle.dump(self.bm25, f)
        if self.embeddings is not None:
            np.save(directory / "embeddings.npy", self.embeddings)
        manifest = {
            "embedding_model": self.model_name,
            "chunk_count": len(self.chunks),
            "content_hash": hashlib.sha256("".join(c.chunk_id for c in self.chunks).encode()).hexdigest(),
            "created_at": time.time(),
        }
        (directory / "meta.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path | str, model_name: str, cache_size: int = 128, cache_ttl: int = 300):
        directory = Path(directory)
        obj = cls(model_name, cache_size, cache_ttl)
        if (directory / "faiss.index").exists():
            obj.index = faiss.read_index(str(directory / "faiss.index"))
        if (directory / "chunks.json").exists():
            raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
            obj.chunks = [Chunk(**x) for x in raw]
        if (directory / "bm25.pkl").exists():
            obj.bm25 = pickle.loads((directory / "bm25.pkl").read_bytes())
        if (directory / "embeddings.npy").exists():
            try:
                obj.embeddings = np.load(directory / "embeddings.npy")
            except Exception:
                pass
        return obj
