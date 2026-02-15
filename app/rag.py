from dotenv import load_dotenv
load_dotenv(override=True)

"""
rag.py — Research-grade RAG pipeline.

Features:
  • Hybrid retrieval (FAISS dense + BM25 sparse via RRF)
  • Cross-encoder reranking
  • Structured citations [Paper Title | Section | Page N]
  • Conversation history support
  • Configurable LLM backend (OpenAI-compatible API)
"""

import os
import re
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

from sentence_transformers import CrossEncoder

from embeddings import hybrid_search, dense_search, get_model, INDEX_DIR, EMBED_MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_TOP_K = 15       # fetch this many from hybrid search
RERANK_TOP_K    = 3        # keep this many after reranking
DENSE_WEIGHT    = 0.6      # weight for dense vs sparse in hybrid

# LLM config — set via environment variables
LLM_BASE_URL    = os.getenv("LLM_BASE_URL", None)  # None = default OpenAI, or "http://localhost:11434/v1" for Ollama
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY     = os.getenv("LLM_API_KEY", "")  # set your OpenAI key here or in .env


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class Citation:
    paper_title: str
    section: str
    page_number: int
    category: str
    relevance_score: float
    chunk_text: str  # the actual passage

@dataclass
class RAGResponse:
    answer: str
    citations: list[Citation]
    query: str
    retrieval_time_ms: float
    rerank_time_ms: float
    generation_time_ms: float
    context_chunks_used: int
    model_used: str


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
_reranker_cache = {}

def get_reranker(model_name: str = RERANKER_MODEL) -> CrossEncoder:
    if model_name not in _reranker_cache:
        log.info(f"🔄 Loading reranker: {model_name}")
        _reranker_cache[model_name] = CrossEncoder(model_name, max_length=512)
    return _reranker_cache[model_name]


def rerank(query: str, results: list[dict],
           top_k: int = RERANK_TOP_K,
           model_name: str = RERANKER_MODEL) -> list[dict]:
    """Re-score results with a cross-encoder and return top_k."""
    if not results:
        return results

    reranker = get_reranker(model_name)
    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)

    for i, r in enumerate(results):
        r["rerank_score"] = float(scores[i])

    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


# ---------------------------------------------------------------------------
# Context builder with structured citations
# ---------------------------------------------------------------------------

def build_context(chunks: list[dict],
                  max_context_tokens: int = 3000) -> tuple[str, list[Citation]]:
    """
    Build the context string for the LLM prompt and extract structured citations.
    Each chunk is truncated so total context stays within max_context_tokens.
    Rough estimate: 1 token ≈ 4 chars.
    """
    max_chars = max_context_tokens * 4
    chars_per_chunk = max_chars // max(len(chunks), 1)

    context_parts = []
    citations = []

    for i, chunk in enumerate(chunks):
        truncated_text = chunk["text"][:chars_per_chunk]

        citation = Citation(
            paper_title=chunk.get("paper_title", "Unknown"),
            section=chunk.get("section", "Unknown"),
            page_number=chunk.get("page_number", -1),
            category=chunk.get("paper_category", ""),
            relevance_score=chunk.get("rerank_score", chunk.get("score", 0)),
            chunk_text=truncated_text[:500],
        )
        citations.append(citation)

        cite_tag = f"[{i+1}]"
        header = (f"[{cite_tag}] \"{citation.paper_title}\" | "
                  f"§{citation.section} | p.{citation.page_number}")
        context_parts.append(f"{header}\n{truncated_text}\n")

    context = "\n".join(context_parts)
    return context, citations


def format_citations_block(citations: list[Citation]) -> str:
    """Format citations as a readable reference block."""
    lines = ["\n📚 References:"]
    for i, c in enumerate(citations):
        page_str = f"p.{c.page_number}" if c.page_number > 0 else "n/a"
        lines.append(
            f"  [{i+1}] {c.paper_title} — §{c.section}, {page_str} "
            f"({c.category}) [relevance: {c.relevance_score:.3f}]"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM Generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a quantitative finance research assistant. Answer using ONLY the provided sources. Cite inline with [1], [2] etc. If sources lack info, say so. Be precise and technical. Do NOT add a references section at the end — citations are handled separately."""


def _get_llm_client():
    """Get or create a cached OpenAI client."""
    from openai import OpenAI
    if not hasattr(_get_llm_client, "_client"):
        kwargs = {"api_key": LLM_API_KEY} if LLM_API_KEY else {}
        if LLM_BASE_URL:
            kwargs["base_url"] = LLM_BASE_URL
        _get_llm_client._client = OpenAI(**kwargs)
    return _get_llm_client._client


def _call_llm(messages: list[dict], model: str = LLM_MODEL) -> str:
    """
    Call LLM via the official OpenAI client library.
    Works with: OpenAI, Ollama (base_url), vLLM, Azure, etc.
    """
    try:
        client = _get_llm_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "rate" in error_msg.lower() or "429" in error_msg:
            log.warning(f"⚠️  Rate limited: {e}")
        elif "connect" in error_msg.lower() or "connection" in error_msg.lower():
            log.warning(f"⚠️  LLM API not available: {e}")
        else:
            log.error(f"LLM error: {e}")
        return None


def _build_fallback_answer(query: str, citations: list[Citation]) -> str:
    """When no LLM is available, return a structured summary of retrieved sources."""
    parts = [f"**Query:** {query}\n",
             "⚠️ *LLM not available — showing retrieved passages:*\n"]
    for i, c in enumerate(citations):
        parts.append(f"**[{i+1}] {c.paper_title}** — §{c.section}")
        parts.append(f"> {c.chunk_text[:300]}…\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_interaction(query: str, response: RAGResponse,
                    logfile: str = "rag_logs.jsonl") -> None:
    """Append the interaction to a JSONL log file."""
    entry = {
        "timestamp": time.time(),
        "query": query,
        "answer": response.answer,
        "model_used": response.model_used,
        "retrieval_ms": response.retrieval_time_ms,
        "generation_ms": response.generation_time_ms,
        "citations": [asdict(c) for c in response.citations]
    }
    try:
        with open(logfile, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.error(f"Failed to write log: {e}")


# ---------------------------------------------------------------------------
# Main RAG pipeline
# ---------------------------------------------------------------------------

def ask(query: str,
        conversation_history: Optional[list[dict]] = None,
        retrieval_top_k: int = RETRIEVAL_TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
        dense_weight: float = DENSE_WEIGHT,
        use_reranker: bool = True,
        index_dir: str = INDEX_DIR,
        model_name: str = EMBED_MODEL_NAME) -> RAGResponse:
    """
    Full RAG pipeline:
      1. Hybrid retrieval (dense + sparse)
      2. Cross-encoder reranking
      3. Context building with structured citations
      4. LLM generation with citation-backed answer
    """
    # 1. Retrieve
    t0 = time.time()
    try:
        retrieved = hybrid_search(
            query, top_k=retrieval_top_k,
            dense_weight=dense_weight, index_dir=index_dir,
            model_name=model_name,
        )
    except FileNotFoundError:
        return RAGResponse(
            answer="❌ No index found. Please build the index first via /buildEmbeddings.",
            citations=[], query=query,
            retrieval_time_ms=0, rerank_time_ms=0, generation_time_ms=0,
            context_chunks_used=0, model_used="none",
        )
    retrieval_ms = (time.time() - t0) * 1000

    # 2. Rerank
    t1 = time.time()
    if use_reranker and retrieved:
        top_chunks = rerank(query, retrieved, top_k=rerank_top_k)
    else:
        top_chunks = retrieved[:rerank_top_k]
    rerank_ms = (time.time() - t1) * 1000

    # 3. Build context + citations
    context, citations = build_context(top_chunks)

    # 4. Generate
    t2 = time.time()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history if provided
    if conversation_history:
        for msg in conversation_history[-6:]:  # last 3 exchanges max
            messages.append(msg)

    user_prompt = f"""SOURCES:
{context}

Q: {query}

Answer with [N] citations."""

    messages.append({"role": "user", "content": user_prompt})

    llm_answer = _call_llm(messages)
    generation_ms = (time.time() - t2) * 1000

    if llm_answer is None:
        llm_answer = _build_fallback_answer(query, citations)

    response = RAGResponse(
        answer=llm_answer,
        citations=citations,
        query=query,
        retrieval_time_ms=round(retrieval_ms, 1),
        rerank_time_ms=round(rerank_ms, 1),
        generation_time_ms=round(generation_ms, 1),
        context_chunks_used=len(top_chunks),
        model_used=LLM_MODEL,
    )
    
    # Log the interaction
    log_interaction(query, response)
    
    return response


# ---------------------------------------------------------------------------
# Retrieval-only mode (no LLM, just returns ranked passages)
# ---------------------------------------------------------------------------

def retrieve_only(query: str,
                  top_k: int = RERANK_TOP_K,
                  use_reranker: bool = True,
                  index_dir: str = INDEX_DIR) -> list[dict]:
    """
    Retrieve and optionally rerank — returns passages with citations
    but no LLM generation. Useful for evaluation and debugging.
    """
    retrieved = hybrid_search(query, top_k=top_k * 3, index_dir=index_dir)
    if use_reranker and retrieved:
        return rerank(query, retrieved, top_k=top_k)
    return retrieved[:top_k]