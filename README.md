# Alpha Research Portal

A RAG-powered research assistant for quantitative finance papers. Fetches papers from arXiv, builds hybrid search indices, and generates concise, citation-backed answers.

---

## Quick Start (5 minutes)

### 1. Install dependencies

```bash
cd "Alpha Research Portal"
poetry install
```

> Requires **Python 3.10+** and **[Poetry](https://python-poetry.org/docs/#installation)**.

### 2. Add your API key

Create a `.env` file in the project root:

```
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=<your-groq-api-key>
LLM_MODEL=llama-3.3-70b-versatile
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Start the server

```bash
cd app
poetry run uvicorn main:app --reload --port 8000
```

### 4. Open the UI

Go to **http://localhost:8000** in your browser.

### 5. Fetch papers & build index

1. **Fetch papers** — In the UI sidebar, click **Get Papers** (or run: `curl -X POST http://localhost:8000/getPapers -H "Content-Type: application/json" -d '{"max_papers": 20}'`)
2. **Build index** — Click **Build / Update Index** (or run: `curl -X POST http://localhost:8000/buildEmbeddings -H "Content-Type: application/json" -d '{}'`)
3. Wait for the status to say **Ready**

### 6. Ask a question

Type a question like: *"How do transaction costs affect portfolio optimization?"*

You'll get a concise, cited paragraph with source cards below.

---

## What You Can Do

| Feature | How |
|---------|-----|
| **Ask questions** | Type in the chat — get cited answers |
| **Research threads** | Click **New Thread** to group related questions |
| **Evidence table** | Click 📋 in the right panel — extracts claims with confidence |
| **Bibliography** | Click 📚 — annotated list of all referenced papers |
| **Synthesis memo** | Click 📝 — 800-word narrative with citations |
| **Download** | Click ⬇ Markdown or ⬇ CSV on any artifact |
| **Evaluate** | Click **Run Evaluation** — RAGAs-style metrics on 20 queries |

---

## Project Structure

```
Alpha Research Portal/
├── app/
│   ├── main.py          # FastAPI server
│   ├── rag.py           # RAG pipeline (retrieve → rerank → generate)
│   ├── embeddings.py    # PDF processing, FAISS + BM25 indexing
│   ├── evaluation.py    # RAGAs-style evaluation
│   ├── artifacts.py     # Evidence table, bibliography, synthesis memo
│   ├── threads.py       # Research thread persistence
│   ├── export.py        # Markdown/CSV export
│   └── scraper.py       # arXiv paper fetcher
├── chat.html            # Frontend UI
├── data/                # Downloaded PDFs (by arXiv category)
├── index/               # Search indices (auto-generated)
├── threads/             # Saved research threads
├── artifacts/           # Generated artifacts
├── eval_report.json     # Latest evaluation results
├── .env                 # API keys
└── pyproject.toml       # Dependencies
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI (Python 3.12) |
| Embeddings | all-MiniLM-L6-v2 (SentenceTransformers) |
| Search | FAISS (dense) + BM25 (sparse) with RRF fusion |
| Reranker | ms-marco-MiniLM-L-6-v2 (CrossEncoder) |
| LLM | Llama 3.3 70B via Groq API |
| Math rendering | KaTeX |
| Frontend | Vanilla HTML/CSS/JS |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/getPapers` | Fetch papers from arXiv |
| `GET`  | `/papers` | List indexed papers |
| `POST` | `/buildEmbeddings` | Build/update search index |
| `POST` | `/chat` | RAG Q&A with citations |
| `POST` | `/search` | Raw hybrid search |
| `POST` | `/evaluate` | Run evaluation (background) |
| `GET`  | `/evalReport` | Get evaluation results |
| `POST` | `/threads` | Create research thread |
| `GET`  | `/threads` | List all threads |
| `POST` | `/artifacts/evidenceTable` | Generate evidence table |
| `POST` | `/artifacts/bibliography` | Generate bibliography |
| `POST` | `/artifacts/synthesisMemo` | Generate synthesis memo |
| `GET`  | `/export/{thread_id}/{type}/{format}` | Export artifact |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Address already in use` | Run `lsof -ti :8000 \| xargs kill -9` then restart |
| `ModuleNotFoundError` | Run `poetry install` |
| `429 Too Many Requests` | Hit Groq rate limit — wait 60s or use a paid key |
| `ASGI app not found` | Make sure you run from inside the `app/` directory |
