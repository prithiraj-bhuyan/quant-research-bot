# Quant Finance Research Portal

A RAG-powered research assistant for quantitative finance papers. Fetches papers from arXiv, builds hybrid search indices (FAISS dense + BM25 sparse), reranks with a cross-encoder, and generates citation-backed answers via any OpenAI-compatible LLM.

---

## Features

- **arXiv Paper Ingestion** — Fetch and bucket papers by category (`q-fin.TR`, `q-fin.PM`, `q-fin.ST`)
- **Section-Aware Chunking** — Detects paper sections (Abstract, Methodology, Results, etc.) and never splits across boundaries. Preserves math/formula blocks.
- **Hybrid Retrieval** — FAISS (dense) + BM25 (sparse) merged via Reciprocal Rank Fusion
- **Cross-Encoder Reranking** — `ms-marco-MiniLM-L-6-v2` re-scores candidates for precision
- **Structured Citations** — Every answer cites `[Paper Title | §Section | Page N | Category]`
- **RAGAs-Style Evaluation** — Context Precision, Answer Relevancy, Faithfulness across 20 test queries
- **Chat UI** — Minimal browser interface with conversation history

---

## Project Structure

```
Alpha Research Portal/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI endpoints
│   ├── rag.py             # RAG pipeline (retrieval → rerank → generate)
│   ├── embeddings.py      # PDF extraction, chunking, FAISS + BM25 indexing
│   ├── evaluation.py      # RAGAs-style evaluation metrics
│   └── scraper.py         # arXiv paper fetcher
├── chat.html              # Chat UI (served at /)
├── quant_library/         # Downloaded PDFs (auto-created)
├── index/                 # FAISS + BM25 indices (auto-created)
├── .env                   # Environment variables (you create this)
├── pyproject.toml
└── README.md
```

---

## Prerequisites

- **Python 3.10 – 3.14** (3.12 recommended)
- **Poetry** for dependency management
- An LLM API key (Groq free tier, OpenAI, or local Ollama)

---

## Setup

### 1. Clone the repository

```bash
cd "Alpha Research Portal"
```

### 2. Install dependencies

```bash
poetry env use python3.12
poetry install
```

If adding packages manually:

```bash
poetry add fastapi "uvicorn[standard]" arxiv requests PyMuPDF sentence-transformers rank-bm25 openai python-dotenv
poetry add faiss-cpu --python ">=3.10,<3.15"
```

### 3. Create the `.env` file

Create a `.env` file in the **project root** (same level as `app/` and `chat.html`):

```bash
touch .env
```

Add the following, replacing placeholders with your actual values:

```env
# ── LLM Configuration ──────────────────────────────────────
# Option A: Groq (free, recommended)
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=<your-groq-api-key>
LLM_MODEL=llama-3.3-70b-versatile
```

```

---

## Running the App

### Start the server

```bash
cd app
uvicorn main:app --reload --port 8000
```

### Open the UI

Go to **[http://localhost:8000](http://localhost:8000)** in your browser.

Swagger API docs are at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

---

## Usage Workflow

### Step 1: Fetch papers from arXiv

Use the sidebar in the chat UI, or call the API directly:

```bash
curl -X POST http://localhost:8000/getPapers \
  -H "Content-Type: application/json" \
  -d '{"max_papers": 20}'
```

Papers are downloaded to `quant_library/` and bucketed by category.

### Step 2: Build the search index

Click **"Build / Update Index"** in the sidebar, or:

```bash
curl -X POST http://localhost:8000/buildEmbeddings \
  -H "Content-Type: application/json" \
  -d '{"chunk_size": 500, "overlap": 80}'
```

This extracts text from PDFs, chunks them (section-aware), generates embeddings, and builds both FAISS and BM25 indices. Poll `/buildStatus` to check progress.

### Step 3: Ask questions

Type a question in the chat UI. The pipeline:

1. **Hybrid retrieval** — searches FAISS (dense) and BM25 (sparse), merges via RRF
2. **Reranking** — cross-encoder re-scores top candidates
3. **Generation** — LLM produces a citation-backed answer
4. **Citations** — displayed as cards below the answer

### Step 4: Run evaluation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"mode": "embedding", "num_queries": 20}'
```

Then view the report:

```bash
curl http://localhost:8000/evalReport
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/getPapers` | Fetch papers from arXiv |
| `GET` | `/papers` | List downloaded papers by category |
| `POST` | `/buildEmbeddings` | Incremental index build (background) |
| `POST` | `/rebuildEmbeddings` | Full rebuild from scratch (background) |
| `GET` | `/buildStatus` | Poll build progress |
| `GET` | `/indexStatus` | Index metadata (model, size, params) |
| `POST` | `/chat` | RAG-powered Q&A with citations |
| `POST` | `/search` | Hybrid search (raw results) |
| `POST` | `/retrieve` | Retrieval + reranking, no LLM |
| `POST` | `/evaluate` | Run RAGAs-style evaluation (background) |
| `GET` | `/evalReport` | Get latest evaluation report |
| `POST` | `/removePapers` | Remove papers from index |

---

## CLI Tools

You can also use the modules directly from the command line:

```bash
# Build index
python embeddings.py build

# Incremental update
python embeddings.py update

# Search
python embeddings.py search "portfolio optimization"

# Check index status
python embeddings.py status

# Run evaluation
python evaluation.py --run --mode embedding --queries 20
```

---

## Configuration

Key parameters can be adjusted in `embeddings.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence transformer for embeddings |
| `CHUNK_SIZE` | `500` | Target words per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap between consecutive chunks |

RAG parameters in `rag.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |
| `RETRIEVAL_TOP_K` | `15` | Candidates fetched from hybrid search |
| `RERANK_TOP_K` | `3` | Candidates kept after reranking |
| `DENSE_WEIGHT` | `0.6` | Weight for dense vs sparse in RRF |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `poetry install` and make sure you're in `poetry shell` |
| `.env` not loading | Add `from dotenv import load_dotenv; load_dotenv(override=True)` at top of `rag.py` |
| `429 Too Many Requests` | You've hit rate limits — switch to Groq or Ollama |
| `413 Payload Too Large` | Reduce `RERANK_TOP_K` or `max_context_tokens` in `rag.py` |
| `faiss-cpu` won't install | Use `poetry add faiss-cpu --python ">=3.10,<3.15"` |
| Sections show `§Unknown` | PDF text may lack clean section headers — this is expected for some papers |
| `ASGI app not found` | Run from inside the `app/` directory: `cd app && uvicorn main:app --reload --port 8000` |