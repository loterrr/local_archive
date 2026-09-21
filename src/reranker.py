from __future__ import annotations
from dataclasses import dataclass
from threading import RLock
import time
import numpy as np
from sentence_transformers import CrossEncoder
from .download_models import ensure_model_ready

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"
SHALLOW_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-2-v2"
DEEP_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

@dataclass(frozen=True)
class RerankedContext:
    chunk: object
    retrieval_score: float
    reranker_score: float
    dense_rank: int | None
    sparse_rank: int | None

class CrossEncoderReranker:
    """Lazy-loaded local MS MARCO Cross-Encoder with telemetry and multi-model support."""
    _model_cache: dict[str, CrossEncoder] = {}

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._lock = RLock()
        self.predict_calls = 0
        self.total_pairs = 0
        self.total_latency_ms = 0.0
        self.last_latency_ms = 0.0

    def set_model(self, model_name: str):
        if self.model_name != model_name:
            self.model_name = model_name

    def _get_model(self) -> CrossEncoder:
        if self.model_name not in self._model_cache:
            with self._lock:
                if self.model_name not in self._model_cache:
                    resolved_path, is_local = ensure_model_ready(self.model_name)
                    if is_local:
                        try:
                            self._model_cache[self.model_name] = CrossEncoder(
                                resolved_path,
                                device=self.device,
                                local_files_only=True,
                            )
                        except TypeError:
                            self._model_cache[self.model_name] = CrossEncoder(
                                resolved_path,
                                device=self.device,
                                automodel_args={"local_files_only": True},
                            )
                    else:
                        self._model_cache[self.model_name] = CrossEncoder(resolved_path, device=self.device)
        return self._model_cache[self.model_name]


    def rerank(self, query: str, candidates, top_k: int = 10, batch_size: int = 32):
        if not candidates:
            return []
        model = self._get_model()
        started = time.perf_counter()
        pairs = [
            (query, f"[{item[0].filename} Page {item[0].page_number}]: {item[0].text}")
            for item in candidates
        ]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)[:top_k]
        self.predict_calls += 1
        self.total_pairs += len(pairs)
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        self.total_latency_ms += self.last_latency_ms
        return [
            RerankedContext(chunk=item[0], retrieval_score=float(item[1]), reranker_score=float(score),
                            dense_rank=item[2], sparse_rank=item[3])
            for item, score in ranked
        ]

    def metrics(self) -> dict:
        avg_lat = (self.total_latency_ms / self.predict_calls) if self.predict_calls else 0.0
        throughput = (self.total_pairs / (self.total_latency_ms / 1000.0)) if self.total_latency_ms > 0 else 0.0
        return {
            "model": self.model_name,
            "layers": 2 if "L-2" in self.model_name or "L2" in self.model_name else 6,
            "params_est": "~8.5M" if "L-2" in self.model_name or "L2" in self.model_name else "~22.7M",
            "predict_calls": self.predict_calls,
            "total_pairs": self.total_pairs,
            "last_latency_ms": round(self.last_latency_ms, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "pairs_per_sec": round(throughput, 1),
        }
