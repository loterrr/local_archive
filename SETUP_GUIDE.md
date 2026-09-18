# LocalRAG: Complete New Device Setup Guide

This guide walks you step-by-step through setting up and running **LocalRAG** on another machine (Windows, macOS, or Linux).

---

## 💻 1. System Requirements

| Specification | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Windows 10/11, macOS 12+, Ubuntu 20.04+ | Windows 11, macOS (Apple Silicon M-series), or Ubuntu 22.04+ |
| **CPU** | 4 Cores | 6+ Cores |
| **RAM** | 8 GB | 16 GB (recommended for running 3.8B LLM smoothly) |
| **Disk Space** | 8 GB free | 15 GB free (for models, index, and cache) |
| **GPU** | Not required (100% CPU-optimized) | Optional (NVIDIA CUDA or Apple Metal for faster LLM tokens) |

---

## 📦 2. Prerequisites to Install on the New Device

Before setting up LocalRAG, install these three free tools:

### 1. Python 3.10 or 3.11 (Recommended: Python 3.11)
* **Windows**: Download from [python.org](https://www.python.org/downloads/).  
  ⚠️ **Important**: During installation, **check the box: "Add python.exe to PATH"**.
* **macOS**: Install via Homebrew: `brew install python@3.11`
* **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip`

### 2. Ollama (For Local Offline AI Generation)
* Download and install from: [https://ollama.com/download](https://ollama.com/download)
* After installing, make sure Ollama is running in your taskbar or terminal.

### 3. Tesseract OCR *(Optional - only needed if you ingest scanned image PDFs)*
* **Windows**: Download installer from [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
* **macOS**: `brew install tesseract`
* **Linux**: `sudo apt install -y tesseract-ocr`

---

## 🚀 3. Step-by-Step Installation

### Step 1: Copy the Project Files
Copy the `localrag` folder to the new device (via USB, zip, or Git).

> 💡 **What to copy / what to skip:**
> * **DO copy**: `app/`, `src/`, `data/`, `bench/`, `tests/`, `requirements.txt`, `.env.example`.
>   * *Keeping `data/indexes/current/` is recommended:* It contains the pre-indexed 4,995 chunks from the 20 scientific papers, allowing you to use the system immediately without waiting to re-index!
> * **DO NOT copy**: `.venv/` (Python virtual environments must be created fresh on the new machine).

---

### Step 2: Open Terminal / Command Prompt
Navigate into the project directory:
* **Windows (PowerShell or CMD)**:
  ```powershell
  cd C:\path\to\localrag
  ```
* **macOS / Linux**:
  ```bash
  cd /path/to/localrag
  ```

---

### Step 3: Create and Activate the Python Virtual Environment

#### On Windows:
```powershell
# If PowerShell blocks script execution, run this first:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Create virtual environment:
python -m venv .venv

# Activate virtual environment:
.venv\Scripts\activate
```
*(You will see `(.venv)` appear at the start of your terminal prompt).*

#### On macOS / Linux:
```bash
# Create virtual environment:
python3 -m venv .venv

# Activate virtual environment:
source .venv/bin/activate
```

---

### Step 4: Install Python Dependencies

Run the following command inside your activated `(.venv)`:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*This installs FastAPI, Uvicorn, FAISS, PyMuPDF, Sentence-Transformers, Rank-BM25, Pandas, SciPy, OpenPyXL, and Python-Docx.*

---

### Step 5: Configure the Environment Variables (`.env`)

Create your `.env` configuration file from the template:

* **Windows**:
  ```cmd
  copy .env.example .env
  ```
* **macOS / Linux**:
  ```bash
  cp .env.example .env
  ```

The default `.env` is already configured for optimal offline CPU performance:
```ini
LOCALRAG_LLM_BASE_URL=http://127.0.0.1:11434
LOCALRAG_LLM_MODEL=phi3.5:latest
LOCALRAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LOCALRAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-2-v2
LOCALRAG_RERANKER_ENABLED=true
LOCALRAG_RERANKER_CANDIDATES=50
LOCALRAG_RERANKER_BATCH_SIZE=32
LOCALRAG_DENSE_K=32
LOCALRAG_FINAL_K=8
LOCALRAG_RRF_K=60
LOCALRAG_CACHE_SIZE=128
LOCALRAG_CACHE_TTL=300
```

---

### Step 6: Pull and Pre-Warm the Ollama Model

Make sure Ollama is running, then pull the target model (`phi3.5:latest`):
```bash
ollama pull phi3.5:latest
```

Verify that Ollama responds:
```bash
ollama run phi3.5:latest "Hello, are you ready?"
```
*(Type `/bye` to exit the Ollama interactive shell once it responds).*

---

### Step 7: (Optional) Pre-Download HuggingFace Models

On the first search or benchmark run, `sentence-transformers` automatically downloads two small models:
1. `sentence-transformers/all-MiniLM-L6-v2` (~80 MB)
2. `cross-encoder/ms-marco-MiniLM-L-2-v2` (~45 MB)

If your new device is **connected to the internet**, this happens automatically during the first search.

> 🔒 **For Completely Offline / Air-Gapped Machines:**
> If the new device will NEVER touch the internet, copy the HuggingFace cache folder from your current device:
> * **Windows Source**: `C:\Users\<YourUser>\.cache\huggingface\`
> * **Destination**: Place it into the same `~/.cache/huggingface/` folder on the new machine.

---

## 🖥️ 4. Launching the System

Start the FastAPI application server:

```bash
python app/server.py
```

You should see output similar to:
```
============================================================
NEXUS: THE ARCHIVE - FastAPI Server Starting
============================================================
Host: 127.0.0.1 | Port: 3000
Serving frontend from: .../app/static
Documents path: .../data/documents
Index path: .../data/indexes/current
INFO:     Uvicorn running on http://127.0.0.1:3000 (Press CTRL+C to quit)
```

Now open your web browser (Chrome, Edge, Firefox, or Safari) and go to:
👉 **`http://localhost:3000`**

---

## 🧪 5. Verification & Smoke Testing

### 1. Check UI Status Badges
At the top of the webpage, confirm:
* **Ollama Status:** Green dot showing `phi3.5:latest (Connected)`.
* **Corpus Status:** `4,995 chunks (20 documents)`.
* **Reranker:** `Active (ms-marco-MiniLM-L-2-v2)`.

### 2. Test a Query in "The Journal" Tab
Ask a question in the chat box, for example:
> *"How does DPR use in-batch negative passages during dual-encoder loss computation?"*

Verify:
1. The AI answers with page citations.
2. Click **"Retrieval Diagnostics"** to see retrieved chunk metadata, dense ranks, sparse ranks, and reranker scores.

### 3. Run Automated Unit Tests
In a separate terminal with `(.venv)` active, run:
```bash
pytest tests/
```
All 6 tests should pass in ~1.2 seconds.

### 4. Run the Real-Time Benchmark
* In the web UI, click the **"Empirical Benchmark"** tab.
* Keep default settings (`Top 10 Output`, `Upstream Pool: 50`).
* Click **"Run Benchmark"**.
* In ~20 seconds, you will see the full live evaluation across 50 scientific queries, with Recall, MRR, and $p$-value statistics!

---

## 🛠️ 6. Troubleshooting & Common Issues

### Issue 1: `Port 3000 is already in use`
* **Cause**: Another service is using port 3000.
* **Fix (Windows)**:
  ```powershell
  # Find process using port 3000:
  netstat -ano | findstr :3000
  # Kill process by PID:
  taskkill /PID <PID_NUMBER> /F
  ```
* **Alternative**: Edit `app/server.py` line 898 and change `port=3000` to `port=8080`.

### Issue 2: `Ollama: Disconnected` or `Failed to connect to http://127.0.0.1:11434`
* **Cause**: The Ollama background service is not running.
* **Fix**:
  * Open the Ollama application from your Start Menu / Applications folder.
  * Or start it from terminal: `ollama serve`

### Issue 3: `Index not found. Please build or load index.`
* **Cause**: `data/indexes/current/` was not copied over.
* **Fix**:
  1. Make sure the 20 PDFs are in `data/documents/`.
  2. In the web interface, go to the **"Documents"** tab.
  3. Click **"Rebuild Corpus Index"**.
  4. The system will extract, chunk, and embed all 20 documents in ~2 minutes.

### Issue 4: `Execution of scripts is disabled on this system` (PowerShell)
* **Cause**: Windows default PowerShell execution policy.
* **Fix**: Run:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

---

## 📑 7. Quick Reference Command Summary

```bash
# 1. Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Start Server
python app/server.py

# 3. Run Tests
pytest tests/

# 4. Generate Word Defense Guide (.docx)
python bench/convert_defense_guide_to_docx.py
```
