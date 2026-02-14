"""
main.py — FastAPI app for the finance research portal.
Endpoints: getPapers, buildEmbeddings, updateEmbeddings, rebuildEmbeddings,
           removePapers, search, listPapers, buildStatus, indexStatus
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os
import json

from scraper import get_paper_links, download_and_bucket
from embeddings import (
    build_full_index,
    update_index,
    remove_papers,
    search as faiss_search,
    load_index,
    load_config,
    LIBRARY_DIR,
    INDEX_DIR,
)

app = FastAPI(title="Quant Finance Research Portal", version="0.2.0")

# ── In-memory build status tracker ──────────────────────────────────────
_build_status = {"state": "idle", "detail": None}


# ── Request / Response models ───────────────────────────────────────────
class GetPapersRequest(BaseModel):
    max_papers: int = Field(default=50, ge=1, le=1000)


class BuildRequest(BaseModel):
    chunk_size: int = Field(default=500, ge=100, le=2000)
    overlap: int = Field(default=80, ge=0, le=500)
    force_rebuild: bool = Field(
        default=False,
        description="If True, wipe and rebuild the entire index from scratch."
    )


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)


class RemovePapersRequest(BaseModel):
    filepaths: list[str] = Field(
        ...,
        description="List of paper filepaths to remove from the index."
    )


# ── Background task wrappers ───────────────────────────────────────────

def _run_update(chunk_size: int, overlap: int, force_rebuild: bool):
    global _build_status
    _build_status = {"state": "building", "detail": "processing…"}
    try:
        stats = update_index(
            library_dir=LIBRARY_DIR,
            index_dir=INDEX_DIR,
            chunk_size=chunk_size,
            overlap=overlap,
            force_rebuild=force_rebuild,
        )
        _build_status = {"state": "done", "detail": stats}
    except Exception as e:
        _build_status = {"state": "error", "detail": str(e)}


def _run_full_rebuild(chunk_size: int, overlap: int):
    global _build_status
    _build_status = {"state": "building", "detail": "full rebuild in progress…"}
    try:
        stats = build_full_index(
            library_dir=LIBRARY_DIR,
            index_dir=INDEX_DIR,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        _build_status = {"state": "done", "detail": stats}
    except Exception as e:
        _build_status = {"state": "error", "detail": str(e)}


# ── Endpoints ───────────────────────────────────────────────────────────

# ---------- Paper fetching ----------

@app.post("/getPapers")
def get_papers(req: GetPapersRequest):
    """Fetch papers from arXiv and download/bucket them."""
    results = get_paper_links(req.max_papers)
    download_and_bucket(results)
    summaries = [
        {
            "title": r.title,
            "primary_category": r.primary_category,
            "pdf_url": r.pdf_url,
            "published": str(r.published),
        }
        for r in results
    ]
    return {"downloaded": len(summaries), "papers": summaries}


@app.get("/papers")
def list_papers():
    """List all downloaded papers grouped by category."""
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


# ---------- Embedding build ----------

@app.post("/buildEmbeddings")
def build_embeddings(req: BuildRequest, bg: BackgroundTasks):
    """
    Smart incremental build — only processes NEW papers.
    Automatically does a full rebuild if model or chunk params changed.
    Set force_rebuild=True to wipe and rebuild everything.
    Runs in the background; poll /buildStatus to check progress.
    """
    if _build_status["state"] == "building":
        raise HTTPException(409, "A build is already in progress.")
    bg.add_task(_run_update, req.chunk_size, req.overlap, req.force_rebuild)
    return {
        "status": "started",
        "mode": "force_rebuild" if req.force_rebuild else "incremental",
        "message": "Embedding build launched in background. Poll /buildStatus.",
    }


@app.post("/rebuildEmbeddings")
def rebuild_embeddings(req: BuildRequest, bg: BackgroundTasks):
    """
    Force a full rebuild from scratch — re-extracts, re-chunks,
    and re-embeds every paper. Use when you want a clean slate.
    """
    if _build_status["state"] == "building":
        raise HTTPException(409, "A build is already in progress.")
    bg.add_task(_run_full_rebuild, req.chunk_size, req.overlap)
    return {
        "status": "started",
        "mode": "full_rebuild",
        "message": "Full rebuild launched in background. Poll /buildStatus.",
    }


@app.get("/buildStatus")
def build_status():
    """Check the current state of the embedding build."""
    return _build_status


@app.get("/indexStatus")
def index_status():
    """Return metadata about the current FAISS index (model, params, size)."""
    cfg = load_config()
    if cfg is None:
        return {"status": "no_index", "message": "No index has been built yet."}
    return {"status": "ok", "config": cfg}


# ---------- Search ----------

@app.post("/search")
def search_papers(req: SearchRequest):
    """Semantic search over the FAISS index."""
    try:
        results = faiss_search(req.query, top_k=req.top_k)
    except FileNotFoundError:
        raise HTTPException(
            503, "Index not built yet. Call /buildEmbeddings first."
        )
    return {"query": req.query, "results": results}


# ---------- Removal ----------

@app.post("/removePapers")
def remove_papers_endpoint(req: RemovePapersRequest):
    """
    Remove specific papers from the FAISS index by their filepaths.
    Note: this re-embeds the remaining chunks since FAISS doesn't
    support selective deletion on flat indices.
    """
    result = remove_papers(req.filepaths)
    return result