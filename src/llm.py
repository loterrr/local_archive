from __future__ import annotations
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
import requests
import re
from typing import Optional, Generator, Dict, Any, List
from threading import RLock

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
_LLAMA_LOCK = RLock()

SYSTEM_PROMPT = """You are Nexus: The Archive, an articulate, conversational academic research partner grounded strictly in peer-reviewed scientific literature.

CONVERSATIONAL & STYLE GUIDELINES:
1. Speak naturally, fluidly, and engagingly like an expert research colleague. Give well-synthesized answers using clear paragraphs and markdown structure.
2. NEVER format responses as robotic repeated templates (e.g. NEVER write "- [1] filename p.X discusses... \n - Summary: ..."). Avoid mechanical formulaic phrases.
3. Integrate source evidence seamlessly in conversational prose, placing bracket citations at the end of sentences (e.g., "...using sliding window attention [1]." or "...as demonstrated by Wampler et al. [1, p. 1]").
4. When asked to summarize or explain documents, directly explain the actual subject matter, thesis, methodology, and key contributions of the paper itself. NEVER divert to the theory of "summarization" or "memory compression" algorithms unless specifically asked about those algorithms.
5. Base all factual claims strictly on the provided LITERATURE EVIDENCE. If evidence is lacking, state so plainly without fabricating facts. Always finish your final sentence completely."""

GREETING_PATTERNS = {
    "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
    "howdy", "yo", "sup", "hola", "bonjour", "help", "who are you", "what can you do",
    "what is this", "what is the archive", "thank you", "thanks", "bye", "goodbye"
}

# Pre-defined curated compact GGUF models for device-constrained & CPU/GPU environments
KNOWN_GGUF_CATALOG = {
    "qwen2.5-3b": {
        "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_gb": 2.1,
        "description": "Recommended for 4GB-6GB GPUs & 8GB+ RAM CPUs. High accuracy, 45+ tok/s on GPU."
    },
    "qwen2.5-1.5b": {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size_gb": 1.0,
        "description": "Ultra-light CPU model. Minimal RAM (<1.2GB), 25+ tok/s on CPU."
    },
    "phi3.5-mini": {
        "repo_id": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "filename": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "size_gb": 2.2,
        "description": "Compact reasoning architecture, 128k context support."
    }
}

def is_conversational_query(text: str) -> bool:
    clean = re.sub(r"[^\w\s]", "", text.strip().lower())
    if not clean:
        return True
    if clean in GREETING_PATTERNS:
        return True
    words = clean.split()
    if len(words) <= 3 and any(w in {"hello", "hi", "hey", "greetings", "howdy", "thanks", "thank"} for w in words):
        return True
    if any(phrase in clean for phrase in ["who are you", "what can you do", "what is this"]):
        return True
    return False

def get_conversational_response(query: str) -> str:
    clean = re.sub(r"[^\w\s]", "", query.strip().lower())
    if any(w in clean for w in ["thank", "thanks"]):
        return "You're very welcome! Let me know if you need anything else from the indexed literature."
    if any(w in clean for w in ["bye", "goodbye"]):
        return "Goodbye! The Archive remains offline and ready whenever you need it."
    
    return """Hello! Welcome to **Archive**, your offline academic research assistant.

I am grounded in your active research manuscripts. What would you like to explore or verify today?"""

def format_rag_context(contexts, doc_map: Optional[Dict[str, int]] = None) -> str:
    blocks = []
    for i, item in enumerate(contexts, 1):
        if isinstance(item, tuple) or isinstance(item, list):
            c = item[0]
        else:
            c = item
        blocks.append(f"[{i}] {c.filename} (Page {c.page_number}):\n{c.text}")
    return "\n\n".join(blocks)


def get_hardware_profile() -> Dict[str, Any]:
    """Detects CPU/GPU capabilities and provides optimal inference configuration."""
    cpu_count = os.cpu_count() or 4
    optimal_threads = max(1, min(8, cpu_count - 1))
    
    gpu_available = False
    gpu_name = "CPU Only"
    gpu_vram_gb = 0.0
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    except Exception:
        pass

    return {
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "cpu_cores": cpu_count,
        "optimal_threads": optimal_threads,
        "recommended_gpu_layers": -1 if gpu_available else 0,
        "engine_mode": "GPU Accelerated" if gpu_available else "CPU (SIMD/AVX2)"
    }


def find_local_gguf_models() -> List[Path]:
    """Scans data/models directory for available GGUF files."""
    if not MODELS_DIR.exists():
        return []
    return sorted(list(MODELS_DIR.glob("*.gguf")), key=lambda p: p.stat().st_size, reverse=True)


# Global singleton cache for loaded llama_cpp Llama instances
_LLAMA_CPP_CACHE: Dict[str, Any] = {}

class LlamaCppEngine:
    """Direct in-process GGUF LLM inference with hardware auto-tuning."""
    
    def __init__(self, model_path: Optional[str] = None, n_ctx: int = 4096):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.hardware = get_hardware_profile()
        self._llm = None
        
    def _resolve_model_path(self, auto_download: bool = False) -> Optional[Path]:
        if self.model_path:
            p = Path(self.model_path)
            if p.is_file():
                return p
            p_rel = MODELS_DIR / p.name
            if p_rel.is_file():
                return p_rel
                
        local_models = find_local_gguf_models()
        if local_models:
            return local_models[0]
            
        if auto_download:
            try:
                print("Provisioning Qwen2.5-3B-Instruct-Q4_K_M.gguf for local offline inference...")
                return download_gguf_model("qwen2.5-3b")
            except Exception as e:
                print(f"Auto-download failed: {e}")
        return None

    def is_available(self) -> bool:
        try:
            import llama_cpp
            return self._resolve_model_path(auto_download=False) is not None
        except ImportError:
            return False

    def get_llm_instance(self):
        resolved = self._resolve_model_path(auto_download=True)
        if not resolved:
            raise FileNotFoundError("No local GGUF model file found in data/models/.")
            
        cache_key = f"{resolved.resolve()}:{self.n_ctx}"
        if cache_key in _LLAMA_CPP_CACHE:
            return _LLAMA_CPP_CACHE[cache_key]
            
        import llama_cpp
        n_gpu_layers = self.hardware["recommended_gpu_layers"]
        n_threads = self.hardware["optimal_threads"]
        
        # Load with memory mapping and optimal thread budget
        llm = llama_cpp.Llama(
            model_path=str(resolved.resolve()),
            n_ctx=self.n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_batch=512,
            verbose=False
        )
        _LLAMA_CPP_CACHE[cache_key] = llm
        return llm

    def generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 700,
        repetition_penalty: float = 1.15,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        llm = self.get_llm_instance()
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for turn in history[-4:]:
                if turn.get("role") in {"user", "assistant"} and turn.get("content"):
                    messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": prompt})
        with _LLAMA_LOCK:
            response = llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repeat_penalty=repetition_penalty
            )
        return response["choices"][0]["message"]["content"].strip()

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 700,
        repetition_penalty: float = 1.15
    ) -> Generator[str, None, None]:
        llm = self.get_llm_instance()
        stream = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repeat_penalty=repetition_penalty,
            stream=True
        )
        for chunk in stream:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content")
            if content:
                yield content


class OllamaEngine:
    """Ollama REST API wrapper with auto-launch capability."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "qwen2.5:3b", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
            return r.ok
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 300,
        repetition_penalty: float = 1.15,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for turn in history[-4:]:
                if turn.get("role") in {"user", "assistant"} and turn.get("content"):
                    messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                "repeat_penalty": repetition_penalty,
                "num_thread": 4
            }
        }
        
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            if r.ok:
                data = r.json()
                msg = data.get("message", {}).get("content", "").strip()
                if msg:
                    return msg
        except Exception:
            pass

        # Fallback to OpenAI-compatible endpoint
        payload_v1 = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False
        }
        r = requests.post(f"{self.base_url}/v1/chat/completions", json=payload_v1, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 300,
        repetition_penalty: float = 1.15
    ) -> Generator[str, None, None]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
                "repeat_penalty": repetition_penalty,
                "num_thread": 4
            }
        }
        r = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=self.timeout)
        if r.ok:
            for line in r.iter_lines():
                if line:
                    chunk_data = json.loads(line.decode("utf-8"))
                    token = chunk_data.get("message", {}).get("content", "")
                    if token:
                        yield token
        else:
            yield self.generate(prompt, system_prompt, temperature, top_p, max_tokens, repetition_penalty)


class LocalLLM:
    """
    Unified LLM Interface.
    Prioritizes in-process llama-cpp-python for zero-daemon operation,
    with seamless automatic fallback to local Ollama.
    """
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "qwen2.5-3b-instruct-q4_k_m.gguf",
        model_path: Optional[str] = None,
        timeout: int = 120
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_path = model_path
        self.timeout = timeout
        
        # Resolve appropriate Ollama model tag if model is a gguf file
        ollama_model_tag = "qwen2.5:3b" if "qwen" in model.lower() else ("phi3.5:latest" if "phi" in model.lower() else model)
        
        self.llama_cpp_engine = LlamaCppEngine(model_path=model_path)
        self.ollama_engine = OllamaEngine(base_url=base_url, model=ollama_model_tag, timeout=timeout)

    def get_active_backend(self) -> str:
        """Determines which backend engine will be used."""
        # 1. If explicit GGUF requested or GGUF exists and llama-cpp is installed
        if self.model.endswith(".gguf") or (self.model_path and Path(self.model_path).exists()):
            if self.llama_cpp_engine.is_available():
                return "llama-cpp-python"
                
        # 2. Check local GGUF models in data/models/
        if self.llama_cpp_engine.is_available():
            return "llama-cpp-python"
            
        # 3. Check Ollama
        if self.ollama_engine.is_available():
            return "ollama"
            
        # 4. Try ensuring Ollama is running
        if ensure_ollama_running() and self.ollama_engine.is_available():
            return "ollama"
            
        return "none"

    def generate(
        self,
        question: str,
        contexts: Any,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 650,
        repetition_penalty: float = 1.15,
        history: Optional[List[Dict[str, str]]] = None,
        doc_map: Optional[Dict[str, int]] = None,
        is_all_docs: bool = False,
        **kwargs
    ) -> str:
        if not contexts:
            return "The uploaded documents do not contain enough information to answer this question."

        context_str = format_rag_context(contexts, doc_map=doc_map)
        budget = kwargs.get("max_new_tokens", max_tokens)
        if is_all_docs:
            prompt = (
                f"INDEXED RESEARCH MANUSCRIPTS IN THE ARCHIVE ({len(contexts)} Files):\n{context_str}\n\n"
                f"RESEARCH QUERY: {question}\n\n"
                f"Act as a professional, articulate academic research partner. Provide an engaging, beautifully structured executive summary covering EACH of the {len(contexts)} manuscripts indexed above.\n\n"
                f"Instructions:\n"
                f"1. Start with a warm, natural introductory sentence acknowledging the active collection.\n"
                f"2. Dedicate a clean section to EACH manuscript using its actual title or core topic as a bold header (e.g. '**1. [Paper Title]**').\n"
                f"3. For each document, write a cohesive 2-3 sentence paragraph summarizing its core thesis, research problem, and key methodologies/findings, citing the manuscript as [1], [2], etc.\n"
                f"4. Do NOT output repetitive mechanical templates like '- [1] filename discusses... \\n - Summary: ...'. Speak naturally like a real chatbot.\n"
                f"5. Finish with a brief concluding cross-paper takeaway highlighting the common themes across the collection."
            )
            tokens_to_use = max(budget, 950)
        else:
            prompt = (
                f"LITERATURE EVIDENCE:\n{context_str}\n\n"
                f"RESEARCH QUERY: {question}\n\n"
                f"Synthesize a clear, cohesive, conversational academic response directly answering the research query using the literature evidence above. "
                f"Cite supporting evidence using bracket numbers like [1] or [2] at the end of sentences:"
            )
            tokens_to_use = budget
        
        return self.generate_raw(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=temperature,
            top_p=top_p,
            max_tokens=tokens_to_use,
            repetition_penalty=repetition_penalty,
            history=history
        )

    def generate_raw(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_tokens: int = 400,
        repetition_penalty: float = 1.15,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        sys_p = system_prompt or SYSTEM_PROMPT
        backend = self.get_active_backend()

        # Primary: llama-cpp-python
        if backend == "llama-cpp-python":
            try:
                return self.llama_cpp_engine.generate(
                    prompt=prompt,
                    system_prompt=sys_p,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty,
                    history=history
                )
            except Exception as e:
                # Fallback to Ollama if llama-cpp encounter runtime issue
                if self.ollama_engine.is_available():
                    return self.ollama_engine.generate(
                        prompt=prompt,
                        system_prompt=sys_p,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        repetition_penalty=repetition_penalty,
                        history=history
                    )
                raise RuntimeError(f"llama-cpp-python generation failed: {e}")

        # Secondary: Ollama
        if backend == "ollama":
            return self.ollama_engine.generate(
                prompt=prompt,
                system_prompt=sys_p,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                history=history
            )

        raise RuntimeError("No local LLM backend is currently available. Please place a GGUF model in 'data/models/' or start Ollama.")

    def stream_generate(
        self,
        question: str,
        contexts: Any,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 300,
        repetition_penalty: float = 1.15
    ) -> Generator[str, None, None]:
        if not contexts:
            yield "The uploaded documents do not contain enough information to answer this question."
            return

        context_str = format_rag_context(contexts)
        prompt = f"LITERATURE EVIDENCE:\n{context_str}\n\nRESEARCH QUERY: {question}\n\nProvide a synthesized, verifiable answer citing the source manuscripts [filename.pdf p.X] or [1], [2]:"
        
        backend = self.get_active_backend()
        if backend == "llama-cpp-python":
            try:
                for chunk in self.llama_cpp_engine.stream_generate(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty
                ):
                    yield chunk
                return
            except Exception:
                pass

        if backend == "ollama" or self.ollama_engine.is_available():
            for chunk in self.ollama_engine.stream_generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty
            ):
                yield chunk
            return

        yield self.generate(question, contexts, temperature, top_p, max_tokens, repetition_penalty)

    def health(self) -> bool:
        """Returns True if either llama-cpp-python with a model or Ollama is online."""
        if self.llama_cpp_engine.is_available():
            return True
        if self.ollama_engine.is_available():
            return True
        return ensure_ollama_running()

    def get_info(self) -> Dict[str, Any]:
        """Provides status metadata regarding backend, hardware, and active model."""
        hw = get_hardware_profile()
        backend = self.get_active_backend()
        local_ggufs = [p.name for p in find_local_gguf_models()]
        
        return {
            "active_backend": backend,
            "hardware": hw,
            "ollama_available": self.ollama_engine.is_available(),
            "local_gguf_count": len(local_ggufs),
            "local_gguf_models": local_ggufs,
            "ollama_model": self.model
        }


def ensure_ollama_running() -> bool:
    """Attempts to check if Ollama is running, and if not, launches ollama serve."""
    try:
        r = requests.get("http://127.0.0.1:11434/", timeout=1)
        if r.ok:
            return True
    except requests.RequestException:
        pass

    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        local_app = os.environ.get("LOCALAPPDATA", "")
        if local_app:
            candidate = Path(local_app) / "Programs" / "Ollama" / "ollama.exe"
            if candidate.is_file():
                ollama_bin = str(candidate)

    if ollama_bin:
        try:
            subprocess.Popen([ollama_bin, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(8):
                time.sleep(0.5)
                try:
                    if requests.get("http://127.0.0.1:11434/", timeout=1).ok:
                        return True
                except requests.RequestException:
                    pass
        except Exception:
            pass
    return False


def get_available_models(base_url: str = "http://127.0.0.1:11434") -> list[str]:
    """
    Returns unified list of available models:
    Local GGUF models in data/models/ + models registered in Ollama.
    """
    models = []
    
    # 1. Local GGUF models in data/models/
    for p in find_local_gguf_models():
        models.append(f"gguf:{p.name}")
        
    # 2. Ollama models
    base = base_url.rstrip("/")
    try:
        r = requests.get(f"{base}/api/tags", timeout=2)
        if r.ok:
            data = r.json()
            for m in data.get("models", []):
                name = m.get("name")
                if name and name not in models:
                    models.append(name)
    except Exception:
        pass
        
    try:
        r = requests.get(f"{base}/v1/models", timeout=2)
        if r.ok:
            data = r.json()
            for m in data.get("data", []):
                m_id = m.get("id")
                if m_id and m_id not in models:
                    models.append(m_id)
    except Exception:
        pass
        
    return models


def download_gguf_model(alias_or_repo: str = "qwen2.5-3b", progress_callback=None) -> Path:
    """
    Downloads a quantized GGUF model from HuggingFace directly to data/models/.
    """
    from huggingface_hub import hf_hub_download
    
    if alias_or_repo.lower() in KNOWN_GGUF_CATALOG:
        entry = KNOWN_GGUF_CATALOG[alias_or_repo.lower()]
        repo_id = entry["repo_id"]
        filename = entry["filename"]
    else:
        # User specified custom repo/filename in format "repo_id:filename"
        if ":" in alias_or_repo:
            repo_id, filename = alias_or_repo.split(":", 1)
        else:
            repo_id = alias_or_repo
            filename = "model.gguf"

    target_file = MODELS_DIR / filename
    if target_file.exists():
        return target_file

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(MODELS_DIR),
        local_dir_use_symlinks=False
    )
    return Path(downloaded_path)


def pull_model(base_url: str, model_name: str, progress_callback=None):
    """Pulls a model from Ollama with progress updates."""
    base = base_url.rstrip("/")
    url = f"{base}/api/pull"
    resp = requests.post(url, json={"name": model_name, "stream": True}, stream=True, timeout=1200)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            try:
                data = json.loads(line.decode("utf-8"))
                if progress_callback:
                    status = data.get("status", "")
                    completed = data.get("completed", 0)
                    total = data.get("total", 0)
                    progress_callback(status, completed, total)
            except Exception:
                pass
