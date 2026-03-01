"""
main.py — FastAPI app for the Quant Finance Research Portal.

Phase 2 endpoints:
  /chat              — RAG-powered Q&A with citations
  /retrieve          — Retrieval-only (no LLM generation)
  /evaluate          — Run RAGAs-style evaluation
  /evalReport        — Get the latest evaluation report

Phase 3 (MVP) endpoints:
  /threads           — Research thread CRUD
  /artifacts/*       — Artifact generation (evidence table, bibliography, memo)
  /export/*          — Export artifacts as Markdown/CSV

Existing endpoints:
  /getPapers, /papers, /buildEmbeddings, /rebuildEmbeddings,
  /buildStatus, /indexStatus, /search, /removePapers
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import Optional
from dataclasses import asdict
from pathlib import Path
import os
import json

from scraper import get_paper_links, download_and_bucket
from embeddings import (
    build_full_index, update_index, remove_papers,
    hybrid_search, dense_search, load_index, load_config,
    LIBRARY_DIR, INDEX_DIR,
)
from rag import ask, retrieve_only, RAGResponse
from evaluation import run_evaluation, DEFAULT_EVAL_QUERIES
from threads import (
    Thread, ThreadEntry, save_thread, load_thread,
    list_threads, delete_thread, add_entry_to_thread,
)
from artifacts import (
    generate_evidence_table, generate_bibliography,
    generate_synthesis_memo, _load_artifact,
)
from export import (
    export_evidence_table_csv, export_evidence_table_markdown,
    export_bibliography_markdown, export_bibliography_csv,
    export_synthesis_memo_markdown,
)

app = FastAPI(title="Quant Finance Research Portal", version="3.0.0")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Serve chat UI ──────────────────────────────────────────────────────
@app.get("/")
def serve_chat():
    return FileResponse(str(PROJECT_ROOT / "chat.html"))


# ── Status tracker ─────────────────────────────────────────────────────
_build_status = {"state": "idle", "detail": None}
_eval_report  = None


# ── Request models ─────────────────────────────────────────────────────
class GetPapersRequest(BaseModel):
    max_papers: int = Field(default=50, ge=1, le=1000)

class BuildRequest(BaseModel):
    chunk_size: int = Field(default=500, ge=100, le=2000)
    overlap: int = Field(default=80, ge=0, le=500)
    force_rebuild: bool = False

class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)

class ChatRequest(BaseModel):
    query: str
    conversation_history: Optional[list[dict]] = None
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True
    thread_id: Optional[str] = None

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = True

class EvalRequest(BaseModel):
    mode: str = Field(default="embedding", pattern="^(embedding|llm_judge)$")
    num_queries: int = Field(default=20, ge=1, le=50)

class RemovePapersRequest(BaseModel):
    filepaths: list[str]

class CreateThreadRequest(BaseModel):
    title: str = Field(default="Untitled Thread", max_length=200)

class ArtifactRequest(BaseModel):
    thread_id: str


# ── Background tasks ──────────────────────────────────────────────────
def _run_update(chunk_size, overlap, force_rebuild):
    global _build_status
    _build_status = {"state": "building", "detail": "processing…"}
    try:
        stats = update_index(chunk_size=chunk_size, overlap=overlap, force_rebuild=force_rebuild)
        _build_status = {"state": "done", "detail": stats}
    except Exception as e:
        _build_status = {"state": "error", "detail": str(e)}

def _run_rebuild(chunk_size, overlap):
    global _build_status
    _build_status = {"state": "building", "detail": "full rebuild…"}
    try:
        stats = build_full_index(chunk_size=chunk_size, overlap=overlap)
        _build_status = {"state": "done", "detail": stats}
    except Exception as e:
        _build_status = {"state": "error", "detail": str(e)}

def _run_eval(mode, num_queries):
    global _eval_report
    try:
        queries = DEFAULT_EVAL_QUERIES[:num_queries]
        report = run_evaluation(queries=queries, mode=mode,
                                output_path=str(PROJECT_ROOT / "eval_report.json"))
        _eval_report = asdict(report)
    except Exception as e:
        _eval_report = {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

# ── Paper management ──────────────────────────────────────────────────

@app.post("/getPapers")
def get_papers(req: GetPapersRequest):
    results = get_paper_links(req.max_papers)
    download_and_bucket(results)
    summaries = [{"title": r.title, "primary_category": r.primary_category,
                  "pdf_url": r.pdf_url, "published": str(r.published)} for r in results]
    return {"downloaded": len(summaries), "papers": summaries}

@app.get("/papers")
def list_papers():
    catalog = {}
    if not os.path.isdir(LIBRARY_DIR):
        return {"papers": catalog}
    for cat in sorted(os.listdir(LIBRARY_DIR)):
        cat_path = os.path.join(LIBRARY_DIR, cat)
        if os.path.isdir(cat_path):
            files = [f for f in sorted(os.listdir(cat_path)) if f.endswith(".pdf")]
            if files:
                catalog[cat] = files
    return {"papers": catalog}

@app.post("/removePapers")
def remove_papers_endpoint(req: RemovePapersRequest):
    return remove_papers(req.filepaths)


# ── Index management ──────────────────────────────────────────────────

@app.post("/buildEmbeddings")
def build_embeddings(req: BuildRequest, bg: BackgroundTasks):
    if _build_status["state"] == "building":
        raise HTTPException(409, "Build in progress.")
    bg.add_task(_run_update, req.chunk_size, req.overlap, req.force_rebuild)
    return {"status": "started", "mode": "force_rebuild" if req.force_rebuild else "incremental"}

@app.post("/rebuildEmbeddings")
def rebuild_embeddings(req: BuildRequest, bg: BackgroundTasks):
    if _build_status["state"] == "building":
        raise HTTPException(409, "Build in progress.")
    bg.add_task(_run_rebuild, req.chunk_size, req.overlap)
    return {"status": "started", "mode": "full_rebuild"}

@app.get("/buildStatus")
def build_status():
    return _build_status

@app.get("/indexStatus")
def index_status():
    cfg = load_config()
    return {"status": "ok", "config": cfg} if cfg else {"status": "no_index"}


# ── Search (raw) ──────────────────────────────────────────────────────

@app.post("/search")
def search_papers(req: SearchRequest):
    try:
        results = hybrid_search(req.query, top_k=req.top_k)
    except FileNotFoundError:
        raise HTTPException(503, "Index not built. Call /buildEmbeddings first.")
    return {"query": req.query, "results": results}


# ── RAG Chat (with optional thread persistence) ─────────────────────

@app.post("/chat")
def chat(req: ChatRequest):
    """RAG-powered Q&A with hybrid retrieval, reranking, and structured citations."""
    response = ask(
        query=req.query,
        conversation_history=req.conversation_history,
        rerank_top_k=req.top_k,
        use_reranker=req.use_reranker,
    )
    result = {
        "answer": response.answer,
        "citations": [asdict(c) for c in response.citations],
        "query": response.query,
        "timings": {
            "retrieval_ms": response.retrieval_time_ms,
            "rerank_ms": response.rerank_time_ms,
            "generation_ms": response.generation_time_ms,
        },
        "context_chunks_used": response.context_chunks_used,
        "model_used": response.model_used,
        "trust_signals": response.trust_signals,
    }

    # Persist to thread if thread_id is provided
    if req.thread_id:
        try:
            entry = ThreadEntry(
                query=req.query,
                answer=response.answer,
                citations=[asdict(c) for c in response.citations],
                evidence_chunks=[asdict(c) for c in response.citations],
                trust_signals=response.trust_signals,
            )
            add_entry_to_thread(req.thread_id, entry)
            result["thread_id"] = req.thread_id
        except FileNotFoundError:
            pass  # Thread doesn't exist - silently skip

    return result

@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    """Retrieval + reranking only — no LLM generation. For debugging and eval."""
    try:
        results = retrieve_only(req.query, top_k=req.top_k, use_reranker=req.use_reranker)
    except FileNotFoundError:
        raise HTTPException(503, "Index not built.")
    return {"query": req.query, "results": results}


# ── Evaluation ───────────────────────────────────────────────────────

@app.post("/evaluate")
def evaluate(req: EvalRequest, bg: BackgroundTasks):
    """Run RAGAs-style evaluation in the background."""
    bg.add_task(_run_eval, req.mode, req.num_queries)
    return {"status": "started", "mode": req.mode, "queries": req.num_queries}

@app.get("/evalReport")
def eval_report():
    """Get the latest evaluation report."""
    if _eval_report is None:
        # Try loading from disk
        report_path = PROJECT_ROOT / "eval_report.json"
        if report_path.exists():
            with open(report_path) as f:
                return json.load(f)
        return {"status": "no_report", "message": "Run /evaluate first."}
    return _eval_report


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — RESEARCH THREADS
# ═══════════════════════════════════════════════════════════════════════

@app.post("/threads")
def create_thread(req: CreateThreadRequest):
    """Create a new research thread."""
    thread = Thread(title=req.title)
    save_thread(thread)
    return {"id": thread.id, "title": thread.title, "created_at": thread.created_at}

@app.get("/threads")
def get_threads():
    """List all research threads."""
    return {"threads": list_threads()}

@app.get("/threads/{thread_id}")
def get_thread(thread_id: str):
    """Get a specific thread with all entries."""
    thread = load_thread(thread_id)
    if thread is None:
        raise HTTPException(404, f"Thread {thread_id} not found")
    return asdict(thread)

@app.delete("/threads/{thread_id}")
def remove_thread(thread_id: str):
    """Delete a research thread."""
    if delete_thread(thread_id):
        return {"status": "deleted", "thread_id": thread_id}
    raise HTTPException(404, f"Thread {thread_id} not found")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — ARTIFACT GENERATION
# ═══════════════════════════════════════════════════════════════════════

@app.post("/artifacts/evidenceTable")
def artifact_evidence_table(req: ArtifactRequest):
    """Generate an evidence table from a research thread."""
    try:
        return generate_evidence_table(req.thread_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Thread {req.thread_id} not found")

@app.post("/artifacts/bibliography")
def artifact_bibliography(req: ArtifactRequest):
    """Generate an annotated bibliography from a research thread."""
    try:
        return generate_bibliography(req.thread_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Thread {req.thread_id} not found")

@app.post("/artifacts/synthesisMemo")
def artifact_synthesis_memo(req: ArtifactRequest):
    """Generate a synthesis memo from a research thread."""
    try:
        return generate_synthesis_memo(req.thread_id)
    except FileNotFoundError:
        raise HTTPException(404, f"Thread {req.thread_id} not found")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — EXPORT
# ═══════════════════════════════════════════════════════════════════════

@app.get("/export/{thread_id}/{artifact_type}")
def export_artifact(thread_id: str, artifact_type: str, format: str = "markdown"):
    """
    Export a generated artifact as Markdown or CSV.
    artifact_type: evidenceTable | bibliography | synthesisMemo
    format: markdown | csv
    """
    # Map URL names to internal names
    type_map = {
        "evidenceTable": "evidence_table",
        "evidence_table": "evidence_table",
        "bibliography": "bibliography",
        "synthesisMemo": "synthesis_memo",
        "synthesis_memo": "synthesis_memo",
    }
    internal_type = type_map.get(artifact_type)
    if not internal_type:
        raise HTTPException(400, f"Unknown artifact type: {artifact_type}")

    data = _load_artifact(thread_id, internal_type)
    if data is None:
        raise HTTPException(404, f"No {artifact_type} artifact found for thread {thread_id}. Generate it first.")

    # Build export content
    if internal_type == "evidence_table":
        if format == "csv":
            content = export_evidence_table_csv(data)
            media_type = "text/csv"
            ext = "csv"
        else:
            content = export_evidence_table_markdown(data)
            media_type = "text/markdown"
            ext = "md"
    elif internal_type == "bibliography":
        if format == "csv":
            content = export_bibliography_csv(data)
            media_type = "text/csv"
            ext = "csv"
        else:
            content = export_bibliography_markdown(data)
            media_type = "text/markdown"
            ext = "md"
    elif internal_type == "synthesis_memo":
        content = export_synthesis_memo_markdown(data)
        media_type = "text/markdown"
        ext = "md"
    else:
        raise HTTPException(400, "Unsupported export type")

    filename = f"{artifact_type}_{thread_id}.{ext}"
    return PlainTextResponse(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )