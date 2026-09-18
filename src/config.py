from dataclasses import dataclass
import os

def env_bool(key, default):
    return os.getenv(key, str(default)).lower() in {"1","true","yes","on"}

@dataclass(frozen=True)
class Settings:
    llm_base_url: str = os.getenv("LOCALRAG_LLM_BASE_URL", "http://127.0.0.1:11434")
    llm_model: str = os.getenv("LOCALRAG_LLM_MODEL", "qwen2.5-3b-instruct-q4_k_m.gguf")
    embedding_model: str = os.getenv("LOCALRAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("LOCALRAG_CHUNK_SIZE", "450"))
    chunk_overlap: int = int(os.getenv("LOCALRAG_CHUNK_OVERLAP", "80"))
    dense_k: int = int(os.getenv("LOCALRAG_DENSE_K", "20"))
    final_k: int = int(os.getenv("LOCALRAG_FINAL_K", "5"))
    reranker_enabled: bool = env_bool("LOCALRAG_RERANKER_ENABLED", True)
    reranker_model: str = os.getenv("LOCALRAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2")
    reranker_candidates: int = int(os.getenv("LOCALRAG_RERANKER_CANDIDATES", "20"))
    reranker_batch_size: int = int(os.getenv("LOCALRAG_RERANKER_BATCH_SIZE", "32"))
    rrf_k: int = int(os.getenv("LOCALRAG_RRF_K", "60"))
    ocr_threshold: int = int(os.getenv("LOCALRAG_OCR_THRESHOLD", "50"))
    ocr_dpi: int = int(os.getenv("LOCALRAG_OCR_DPI", "220"))
    max_new_tokens: int = int(os.getenv("LOCALRAG_MAX_NEW_TOKENS", "700"))
    temperature: float = float(os.getenv("LOCALRAG_TEMPERATURE", "0.1"))
    top_p: float = float(os.getenv("LOCALRAG_TOP_P", "0.9"))
    repetition_penalty: float = float(os.getenv("LOCALRAG_REPETITION_PENALTY", "1.15"))
    cache_size: int = int(os.getenv("LOCALRAG_CACHE_SIZE", "128"))
    cache_ttl: int = int(os.getenv("LOCALRAG_CACHE_TTL", "300"))
