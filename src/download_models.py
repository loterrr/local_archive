"""
Local Model Management & Offline Cache Utility
Ensures SentenceTransformer embedding models and CrossEncoder rerankers are downloaded once
to data/models/ and loaded 100% locally with zero internet dependency or DNS checks.
"""
from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("localrag.models")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CATALOG = {
    "embedding": {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "slug": "all-MiniLM-L6-v2",
        "name": "MiniLM L6 v2 Embedding Model",
        "required_files": ["config.json", "tokenizer.json"],
    },
    "reranker_shallow": {
        "id": "cross-encoder/ms-marco-MiniLM-L-2-v2",
        "slug": "ms-marco-MiniLM-L-2-v2",
        "name": "MS MARCO MiniLM L-2 Cross-Encoder",
        "required_files": ["config.json"],
    },
    "reranker_deep": {
        "id": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "slug": "ms-marco-MiniLM-L6-v2",
        "name": "MS MARCO MiniLM L6 Cross-Encoder",
        "required_files": ["config.json"],
    },
}

def get_slug_from_name(model_name_or_id: str) -> str:
    """Normalize a Hugging Face repo ID or local path to a simple directory slug."""
    clean = model_name_or_id.strip().replace("\\", "/").rstrip("/")
    slug = clean.split("/")[-1]
    return slug

def get_local_model_dir(model_name_or_id: str) -> Path:
    """Return the designated local directory path for a model."""
    path = Path(model_name_or_id)
    if path.is_absolute() and path.is_dir():
        return path
    slug = get_slug_from_name(model_name_or_id)
    return MODELS_DIR / slug

def is_model_installed(model_name_or_id: str) -> bool:
    """Check if model files are completely present locally."""
    local_dir = get_local_model_dir(model_name_or_id)
    if not local_dir.is_dir():
        return False

    # Check for presence of config.json and at least one model weight file
    has_config = (local_dir / "config.json").is_file()
    has_weights = any(
        (local_dir / w).is_file()
        for w in ["model.safetensors", "pytorch_model.bin", "model.onnx"]
    ) or any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))

    return has_config and has_weights

def get_directory_size_mb(path: Path) -> float:
    """Calculate total size of directory in megabytes."""
    if not path.is_dir():
        return 0.0
    total = sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())
    return round(total / (1024 * 1024), 2)

def get_all_models_status() -> dict:
    """Return status dictionary of all expected local models."""
    status = {}
    for key, spec in MODEL_CATALOG.items():
        slug = spec["slug"]
        local_dir = MODELS_DIR / slug
        installed = is_model_installed(slug)
        size_mb = get_directory_size_mb(local_dir) if installed else 0.0
        status[key] = {
            "id": spec["id"],
            "slug": slug,
            "name": spec["name"],
            "installed": installed,
            "path": str(local_dir) if installed else None,
            "size_mb": size_mb,
        }
    return status

def download_model(repo_id: str, target_dir: Optional[Path] = None) -> Path:
    """
    Download model snapshot from Hugging Face directly to target_dir.
    Supports HF_ENDPOINT mirror (e.g., https://hf-mirror.com) and provides
    resilient error handling.
    """
    if target_dir is None:
        target_dir = get_local_model_dir(repo_id)

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading model '{repo_id}' to '{target_dir}'...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError("huggingface_hub is required to download models. Please run: pip install huggingface_hub")

    endpoint = os.getenv("HF_ENDPOINT", None)
    if endpoint:
        logger.info(f"Using Hugging Face mirror endpoint: {endpoint}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            endpoint=endpoint,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model.msgpack", "tf_model.h5"],
        )
        logger.info(f"Successfully downloaded and saved '{repo_id}' to '{target_dir}'.")
        return target_dir
    except Exception as e:
        logger.error(f"Failed to download model '{repo_id}': {e}")
        # If standard Hugging Face fails with network/DNS error, try fallback mirror if not already set
        if not endpoint and any(err in str(e).lower() for err in ["errno 11001", "getaddrinfo", "connection error", "timeout"]):
            mirror = "https://hf-mirror.com"
            logger.warning(f"Connection to huggingface.co failed. Attempting fallback download via mirror: {mirror}")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    endpoint=mirror,
                    ignore_patterns=["*.msgpack", "*.h5", "flax_model.msgpack", "tf_model.h5"],
                )
                logger.info(f"Successfully downloaded model '{repo_id}' via mirror.")
                return target_dir
            except Exception as mirror_err:
                logger.error(f"Mirror download also failed: {mirror_err}")
                raise RuntimeError(
                    f"Unable to download model '{repo_id}'. Network error: {e}. "
                    f"If you are offline, please ensure the model files are placed manually into: {target_dir}"
                ) from mirror_err
        raise

def ensure_model_ready(model_name_or_id: str) -> tuple[str, bool]:
    """
    Ensure the model is ready locally.
    Returns (resolved_path_str, is_local_files_only).
    If locally present, returns (local_dir_str, True) with zero network attempts.
    If not present, downloads it once and then returns (local_dir_str, True).
    """
    local_dir = get_local_model_dir(model_name_or_id)
    if is_model_installed(str(local_dir)):
        return str(local_dir), True

    # Check if model_name_or_id matches a known catalog entry
    target_repo = model_name_or_id
    for spec in MODEL_CATALOG.values():
        if model_name_or_id in (spec["id"], spec["slug"]):
            target_repo = spec["id"]
            break

    logger.info(f"Model '{model_name_or_id}' not found locally. Initiating one-time download...")
    download_model(target_repo, local_dir)
    return str(local_dir), True

def ensure_all_local_models(verbose: bool = True) -> bool:
    """Ensure all catalog models (MiniLM embedding + rerankers) are downloaded for offline use."""
    all_ok = True
    for key, spec in MODEL_CATALOG.items():
        slug = spec["slug"]
        repo_id = spec["id"]
        local_dir = MODELS_DIR / slug
        if is_model_installed(str(local_dir)):
            if verbose:
                logger.info(f"✓ Model '{spec['name']}' ready locally at: {local_dir}")
        else:
            try:
                if verbose:
                    logger.info(f"Downloading {spec['name']} ({repo_id})...")
                download_model(repo_id, local_dir)
            except Exception as exc:
                logger.error(f"Could not download {spec['name']}: {exc}")
                all_ok = False
    return all_ok

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 60)
    print("Local RAG Model Downloader (100% Offline Preparation)")
    print("=" * 60)
    success = ensure_all_local_models(verbose=True)
    status = get_all_models_status()
    print("\nCurrent Models Status:")
    for k, v in status.items():
        icon = "[OK]" if v["installed"] else "[MISSING]"
        print(f"  {icon} {v['name']}: {'Ready' if v['installed'] else 'Missing'} ({v['size_mb']} MB)")
        if v["path"]:
            print(f"       Path: {v['path']}")
    if not success:
        sys.exit(1)

