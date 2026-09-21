from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "data" / "models"
load_dotenv(ROOT_DIR / ".env")

# If local models exist in data/models/, enforce 100% offline mode
if (MODELS_DIR / "all-MiniLM-L6-v2" / "config.json").is_file():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def env_bool(key, default):
    return os.getenv(key, str(default)).lower() in {"1","true","yes","on"}

def resolve_model_dir(model_name: str) -> str:
    """If model is downloaded locally in data/models/, use the local directory path."""
    slug = model_name.strip().replace("\\", "/").rstrip("/").split("/")[-1]
    local_path = MODELS_DIR / slug
    if (local_path / "config.json").is_file():
        return str(local_path)
    return model_name

@dataclass
class Settings:
    llm_base_url: str = os.getenv("LOCALRAG_LLM_BASE_URL", "http://127.0.0.1:11434")
    llm_model: str = os.getenv("LOCALRAG_LLM_MODEL", "qwen2.5:3b")
    embedding_model: str = resolve_model_dir(os.getenv("LOCALRAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    chunk_size: int = int(os.getenv("LOCALRAG_CHUNK_SIZE", "450"))
    chunk_overlap: int = int(os.getenv("LOCALRAG_CHUNK_OVERLAP", "80"))
    dense_k: int = int(os.getenv("LOCALRAG_DENSE_K", "20"))
    final_k: int = int(os.getenv("LOCALRAG_FINAL_K", "5"))
    reranker_enabled: bool = env_bool("LOCALRAG_RERANKER_ENABLED", True)
    reranker_model: str = resolve_model_dir(os.getenv("LOCALRAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2"))

    reranker_candidates: int = int(os.getenv("LOCALRAG_RERANKER_CANDIDATES", "20"))
    reranker_batch_size: int = int(os.getenv("LOCALRAG_RERANKER_BATCH_SIZE", "32"))
    rrf_k: int = int(os.getenv("LOCALRAG_RRF_K", "60"))
    ocr_threshold: int = int(os.getenv("LOCALRAG_OCR_THRESHOLD", "0"))
    ocr_dpi: int = int(os.getenv("LOCALRAG_OCR_DPI", "220"))
    max_new_tokens: int = int(os.getenv("LOCALRAG_MAX_NEW_TOKENS", "650"))
    temperature: float = float(os.getenv("LOCALRAG_TEMPERATURE", "0.1"))
    top_p: float = float(os.getenv("LOCALRAG_TOP_P", "0.9"))
    repetition_penalty: float = float(os.getenv("LOCALRAG_REPETITION_PENALTY", "1.15"))
    cache_size: int = int(os.getenv("LOCALRAG_CACHE_SIZE", "128"))
    cache_ttl: int = int(os.getenv("LOCALRAG_CACHE_TTL", "300"))
    models_dir: Path = MODELS_DIR

