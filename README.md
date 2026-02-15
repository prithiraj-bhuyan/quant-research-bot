# Quant Research Bot

A specialized RAG (Retrieval-Augmented Generation) system for Quantitative Finance research. This bot ingests academic papers, performs hybrid retrieval (Dense + Sparse), reranks results using a Cross-Encoder, and provides citation-backed answers using an LLM.

## 🚀 Features

- **Hybrid Search:** Combines FAISS (Dense Vector Search) and BM25 (Sparse Keyword Search) for optimal retrieval.
- **Reranking:** Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to re-score and rank the most relevant chunks.
- **Structured Citations:** Every answer includes precise citations (Paper Title, Section, Page Number).
- **Evaluation Suite:** RAGAs-style evaluation metrics (Context Precision, Faithfulness, Relevancy) with automated reporting.
- **Production Logging:** All queries and responses are logged to `rag_logs.jsonl` for audit and analysis.

## 🛠️ Setup

### Prerequisites
- Python 3.9+
- A valid OpenAI API Key (optional, for LLM generation)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd quant-research-bot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure `tf-keras` is installed if you encounter TensorFlow/Keras compatibility issues.*

4. **Environment Configuration:**
   Create a `.env` file in the root directory:
   ```bash
   LLM_API_KEY=your_openai_api_key_here
   LLM_MODEL=gpt-4o-mini
   ```

## 📚 Data Ingestion & Indexing

Before running queries, you must build the search index from your PDF library.

1. **Place PDF papers** in the `quant_library/` directory (organized by subfolders like `q-fin.PM`, `q-fin.ST`, etc.).

2. **Generate the Data Manifest:**
   ```bash
   python generate_manifest.py
   ```
   This creates `manifest.json`, indexing all available PDFs in `data/raw/`.

3. **Build the Index:**
   ```bash
   python build_index.py
   ```
   This parses PDFs, chunks text, generates embeddings, and saves the FAISS/BM25 indexes to the `index/` directory.

## 💻 Usage

### CLI Interface
Run a single query directly from the command line:

```bash
python run_rag.py "How is mean-variance optimization applied in portfolio construction?"
```

### Evaluation Suite
To run the full evaluation over 20 pre-defined queries (Direct, Synthesis, and Edge-Case):

```bash
python app/evaluation.py --run
```
This will:
1. Execute 20 queries against the RAG pipeline.
2. Calculate retrieval and reranking latencies.
3. Compute quality metrics (if LLM is available).
4. Save the results to `eval_report.json`.

## 🔍 Reproducibility

To reproduce the submitted `eval_report.json` and `rag_logs.jsonl`:

1. Ensure `data/raw` contains the provided dataset (120+ papers).
2. Run the indexing script:
   ```bash
   python build_index.py
   ```
3. Run the evaluation script:
   ```bash
   python app/evaluation.py --run
   ```
4. Check the outputs:
   - `eval_report.json`: Aggregate metrics and individual query performance.
   - `rag_logs.jsonl`: Detailed log of every retrieval and generation event.

## 📂 Project Structure

- `app/rag.py`: Main RAG pipeline (Retrieval, Reranking, Generation).
- `app/embeddings.py`: Indexing and search logic (FAISS + BM25).
- `app/evaluation.py`: Optimization and testing suite.
- `data/raw/`: Source PDF documents.
- `index/`: Persisted Vector DB and Sparse Index.
- `manifest.json`: Metadata for all ingested files.