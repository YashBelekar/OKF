# OKF AI Knowledge Assistant

A fully offline document-intelligence app. Upload documents; it extracts text (with OCR for scans), builds an
[OKF](https://github.com/GoogleCloudPlatform/open-knowledge-format) knowledge base (Markdown + YAML frontmatter,
provenance, links, `index.md`, `log.md`), indexes everything, and answers questions with a local LLM through
Ollama, citing document and page.

```
upload -> extract/OCR -> chunk -> SQLite (+FTS5) -> embeddings
                              \-> LLM concept/entity extraction -> OKF files
question -> hybrid retrieval (BM25 + cosine, RRF) -> grounded prompt -> Ollama -> answer + citations
```

## Setup (Windows PowerShell)

```powershell
cd D:\OKF_AI_Assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

1. Install [Ollama](https://ollama.com/download/windows), then `ollama pull qwen3:1.7b`.
2. For OCR, install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) (the app finds it on PATH or at
   `C:\Program Files\Tesseract-OCR`; override with the `TESSERACT_CMD` environment variable).
3. While still online, run once so the embedding model is cached for offline use:
   `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

## Run

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open http://localhost:8501. The sidebar shows whether Ollama, the model and the embedder are ready.

## Test

```powershell
python -m pytest tests -v
python -m evaluation.evaluate evaluation/sample_questions.json   # edit the questions for your documents
```

Tests use a fake LLM and a deterministic embedder, so they need neither Ollama nor model downloads.

## Layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI (upload, ask, knowledge base, documents) |
| `pipeline.py` | `KnowledgeAssistant`: ingestion, retrieval and answering in one place |
| `ingestion/` | loader, text extraction for all formats, OCR, chunking |
| `knowledge_engine/` | OKF generator, LLM concept/entity extractor, OKF validator |
| `retrieval/` | FTS5 keyword search, embeddings, semantic search, hybrid (RRF) |
| `ai_engine/` | Ollama client, prompt builder, answer generator |
| `storage/database.py` | SQLite schema: documents, chunks, embeddings, FTS5 |
| `config/settings.py` | all paths and model settings (`OKF_HOME`, `OLLAMA_MODEL` env overrides) |
| `raw_documents/`, `knowledge_base/`, `data/` | originals, OKF files, database (all local) |

## Behaviour worth knowing

- **Incremental**: duplicate files (same SHA-256) are skipped; only new chunks are embedded.
- **Grounded**: if nothing relevant is retrieved, the LLM is not called and the app says the knowledge base does
  not contain enough information. Sources are marked cited/retrieved based on the `[Evidence N]` markers.
- **Degrades gracefully**: without Ollama, ingestion and search still work; without sentence-transformers a basic
  hashing embedder is used (the UI warns you); without PyMuPDF, PDFs fall back to `pypdf` (no OCR).
- **Speed**: concept extraction runs one LLM call per sampled chunk (max 40 per document) and is slow on CPU.
  Untick it in the upload tab to ingest quickly and still get full search.
- **Backups**: copy the project folder (excluding `.venv`). Moving to another PC: reinstall, `ollama pull` the model, copy the folder.

## Differences from the original guide (deliberate fixes)

- Chunks, embeddings and the keyword index live in SQLite/FTS5 instead of loose JSON files; the guide's settings
  said `.json` while the store wrote JSONL.
- Semantic search ranked with `np.argsort(...)[:-1][:limit]`, which returns the *least* similar chunks; fixed.
- Embeddings were rebuilt for the whole corpus on every upload; now incremental.
- The validator only understood absolute links while the generator wrote relative ones; both are now consistent.
- Duplicate concepts were silently dropped; now new sources are merged into the existing page's provenance.
- Qwen3 emits `<think>` blocks and `think: false` is sent; any leftover thinking text is stripped.
- Activation path is `.venv\Scripts\Activate.ps1` (the guide mixed `.venv` and `venv`).
- Ollama GPU note: the guide states GTX 1650 needs driver 570+; verify against Ollama's current GPU docs. CPU
  inference works regardless.


---

## ⚠️ Final Note: Hardware Requirements for "Smart + Fast" Responses

This project is designed to run **fully offline** on modest hardware. The default configuration (`qwen3:1.7b` + `all-MiniLM-L6-v2`, CPU-first) is chosen so that the application works correctly on low-RAM machines without a powerful GPU. On such hardware, ingestion, OCR, chunking, SQLite/FTS5 indexing, OKF generation, hybrid retrieval (BM25 + cosine + RRF), and grounded answering with citations all function as described above.

However, the following limitations are **hardware-bound, not software-bound**, and must be stated honestly.

### What this build achieves on modest hardware

| Capability | Status on low-VRAM / CPU-first setups |
| :--- | :--- |
| Fully offline operation | ✅ Yes (after one-time model download) |
| Document upload, OCR, chunking, SQLite/FTS5 indexing | ✅ Yes |
| OKF knowledge base generation (Markdown + YAML, provenance, links, `index.md`, `log.md`) | ✅ Yes |
| Hybrid retrieval with citations (document + page) | ✅ Yes (retrieval completes in milliseconds) |
| Grounded answer generation via Ollama | ✅ Yes, but **slow** |
| Multi-paragraph answers with bullet points | ⚠️ Possible with `qwen3:1.7b`, but quality is limited |
| **3–4 second response with ChatGPT-level synthesis** | ❌ **Not achievable without sufficient VRAM** |

### Why the "3–4 second, ChatGPT-level" goal requires more VRAM

The delay is **not** caused by the indexing or retrieval algorithm. Hybrid retrieval (BM25 + cosine similarity + Reciprocal Rank Fusion) completes in **tens of milliseconds**. The delay comes entirely from **autoregressive token generation** inside the LLM, where every output token requires a full pass through the model's weights.

The speed of that pass is limited by **memory bandwidth**. System RAM delivers roughly 50–100 GB/s; GPU VRAM delivers 300–900 GB/s. This is why the same model runs 5–10× faster in VRAM than on CPU. It is a physical limit, not a configuration issue.

To generate a comprehensive 200–300 word answer (≈300–450 tokens) in **3–4 seconds**, the model must sustain **75–150 tokens/second**. That requires the model to reside **entirely in VRAM** with headroom for the KV cache. The table below summarizes the requirement:

| Goal | Model | Minimum VRAM | Expected Speed |
| :--- | :--- | :--- | :--- |
| **This build (default)** | `qwen3:1.7b` (Q4) | 2 GB (or CPU-only) | 5–20 s per answer |
| Usable document Q&A | 3B–4B (Q4) | **4 GB** | ~6–10 s per answer |
| **"Smart + fast" target** | **7B–8B (Q4)** | **8 GB** | **3–5 s per answer** |
| Comfortable "smart + fast" | 8B (Q4) | 12 GB | 2–4 s per answer |
| ChatGPT-level synthesis | 14B+ (Q4) | 16–24 GB | 3–6 s per answer |

**Conclusion:** This application will run correctly on low-RAM machines and will answer questions grounded in your documents, with citations, fully offline. But to achieve the **"3–4 second, ChatGPT-level, comprehensive response"** goal, the model must run a **7B–8B parameter model entirely in VRAM**, which requires a GPU with **at least 8 GB of VRAM** (12 GB recommended). No indexing algorithm, prompt engineering, or software optimization can substitute for that VRAM — the bottleneck is the physical rate at which the model's weights can be read from memory during generation.

If the hardware is later upgraded to a machine with 8–12 GB VRAM, change one line in `config/settings.py` (or set the `OLLAMA_MODEL` environment variable) to `qwen3:8b` (or another 7B–8B model), run `ollama pull qwen3:8b` once while online, and the same pipeline will deliver the fast, comprehensive experience — with no other code changes required.
