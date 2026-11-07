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
