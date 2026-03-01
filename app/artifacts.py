"""
artifacts.py — Research artifact generation.

Produces structured artifacts from a research thread:
  • Evidence table: claim → evidence → citation → confidence
  • Annotated bibliography: source → method → limitations → why it matters
  • Synthesis memo: 800-1200 word narrative with inline citations
"""

import json
import os
import logging
from pathlib import Path
from typing import Optional
from dataclasses import asdict

from threads import load_thread, THREADS_DIR

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = str(PROJECT_ROOT / "artifacts")


# ---------------------------------------------------------------------------
# LLM helper (reuse rag.py's client)
# ---------------------------------------------------------------------------

def _call_llm(messages: list[dict]) -> Optional[str]:
    """Call the LLM for artifact generation — uses higher max_tokens than chat."""
    from rag import _get_llm_client, LLM_MODEL
    try:
        client = _get_llm_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        log.warning(f"Artifact LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_unique_papers(thread) -> list[dict]:
    """De-duplicate papers across all thread entries."""
    seen = set()
    papers = []
    for entry in thread.entries:
        for cit in entry.citations:
            title = cit.get("paper_title", cit.get("title", "Unknown"))
            if title not in seen:
                seen.add(title)
                papers.append(cit)
    return papers


def _collect_all_evidence(thread) -> list[dict]:
    """Gather all evidence chunks with their queries and answers."""
    evidence = []
    for entry in thread.entries:
        for i, chunk in enumerate(entry.evidence_chunks):
            evidence.append({
                "query": entry.query,
                "answer_excerpt": entry.answer[:300],
                "chunk_text": chunk.get("text", chunk.get("chunk_text", ""))[:500],
                "paper_title": chunk.get("paper_title", "Unknown"),
                "section": chunk.get("section", "Unknown"),
                "page_number": chunk.get("page_number", -1),
                "category": chunk.get("paper_category", chunk.get("category", "")),
                "score": chunk.get("rerank_score", chunk.get("score", 0)),
            })
    return evidence


def _save_artifact(thread_id: str, artifact_type: str, data: dict) -> str:
    """Save an artifact to disk.  Returns the file path."""
    out_dir = os.path.join(ARTIFACTS_DIR, thread_id)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{artifact_type}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def _load_artifact(thread_id: str, artifact_type: str) -> Optional[dict]:
    """Load a previously generated artifact."""
    path = os.path.join(ARTIFACTS_DIR, thread_id, f"{artifact_type}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Evidence Table
# ---------------------------------------------------------------------------

EVIDENCE_TABLE_PROMPT = """You are a research analyst. Given the following evidence chunks from academic papers, extract distinct factual claims and organize them into a structured evidence table.

For each claim, produce a JSON object with these fields:
- "claim": A concise, specific factual statement
- "evidence_snippet": The exact supporting text (max 150 words)
- "citation": The paper title and section
- "confidence": One of "High", "Medium", or "Low" based on how directly the evidence supports the claim
- "notes": Any caveats, conditions, or cross-references

Return a JSON array of these objects. Extract 8-15 distinct claims. Return ONLY valid JSON, no preamble."""


def generate_evidence_table(thread_id: str) -> dict:
    """Generate an evidence table from a thread's accumulated evidence."""
    thread = load_thread(thread_id)
    if thread is None:
        raise FileNotFoundError(f"Thread {thread_id} not found")
    if not thread.entries:
        return {"error": "Thread has no entries yet"}

    evidence = _collect_all_evidence(thread)
    if not evidence:
        return {"error": "No evidence chunks found in thread"}

    # Build context for the LLM
    evidence_text = ""
    for i, e in enumerate(evidence):
        evidence_text += (
            f"\n--- Evidence [{i+1}] from \"{e['paper_title']}\" §{e['section']} ---\n"
            f"{e['chunk_text']}\n"
            f"(Query that surfaced this: {e['query']})\n"
        )

    messages = [
        {"role": "system", "content": EVIDENCE_TABLE_PROMPT},
        {"role": "user", "content": evidence_text},
    ]

    raw = _call_llm(messages)
    if raw is None:
        return {"error": "LLM unavailable — cannot generate evidence table"}

    # Parse the JSON response
    try:
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        claims = json.loads(cleaned)
    except json.JSONDecodeError:
        claims = [{"claim": "Parse error", "evidence_snippet": raw[:500],
                    "citation": "", "confidence": "Low", "notes": "Raw LLM output"}]

    result = {
        "thread_id": thread_id,
        "thread_title": thread.title,
        "artifact_type": "evidence_table",
        "total_claims": len(claims),
        "total_sources": len(_collect_unique_papers(thread)),
        "claims": claims,
    }
    _save_artifact(thread_id, "evidence_table", result)
    return result


# ---------------------------------------------------------------------------
# Annotated Bibliography
# ---------------------------------------------------------------------------

BIBLIOGRAPHY_PROMPT = """You are a research librarian specializing in quantitative finance. Given the following paper excerpts and citations, produce an annotated bibliography of 8-12 unique sources.

For each source, produce a JSON object with these fields:
- "source": Full paper title
- "method": The primary method or framework used (1-2 sentences)
- "limitations": Key limitations or caveats (1-2 sentences)
- "why_it_matters": Why this paper is relevant to the research questions (1-2 sentences)

De-duplicate papers that appear multiple times. Return a JSON array of these objects. Return ONLY valid JSON, no preamble."""


def generate_bibliography(thread_id: str) -> dict:
    """Generate an annotated bibliography from a thread."""
    thread = load_thread(thread_id)
    if thread is None:
        raise FileNotFoundError(f"Thread {thread_id} not found")
    if not thread.entries:
        return {"error": "Thread has no entries yet"}

    papers = _collect_unique_papers(thread)
    evidence = _collect_all_evidence(thread)

    # Build context
    context = "PAPERS AND EXCERPTS:\n"
    for e in evidence:
        context += (
            f"\n[{e['paper_title']}] §{e['section']} (Category: {e['category']})\n"
            f"{e['chunk_text'][:400]}\n"
        )

    context += f"\n\nRESEARCH QUERIES ASKED:\n"
    for entry in thread.entries:
        context += f"- {entry.query}\n"

    messages = [
        {"role": "system", "content": BIBLIOGRAPHY_PROMPT},
        {"role": "user", "content": context},
    ]

    raw = _call_llm(messages)
    if raw is None:
        return {"error": "LLM unavailable — cannot generate bibliography"}

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        sources = json.loads(cleaned)
    except json.JSONDecodeError:
        sources = [{"source": "Parse error", "method": raw[:300],
                     "limitations": "", "why_it_matters": ""}]

    result = {
        "thread_id": thread_id,
        "thread_title": thread.title,
        "artifact_type": "bibliography",
        "total_sources": len(sources),
        "sources": sources,
    }
    _save_artifact(thread_id, "bibliography", result)
    return result


# ---------------------------------------------------------------------------
# Synthesis Memo
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """You are a senior quantitative researcher writing an internal research memo. Based on the following evidence and research queries, write a synthesis memo of 800-1200 words.

Requirements:
1. Use inline citations like [1], [2] throughout the text.
2. Structure with clear sections: Introduction, Key Findings, Methodological Considerations, Gaps & Future Directions, Conclusion.
3. Integrate findings across multiple papers — do NOT summarize paper-by-paper.
4. End with a numbered reference list matching your inline citations.
5. Be precise, technical, and actionable.

Return the memo as a JSON object with:
- "memo_text": The full memo in markdown format
- "word_count": Approximate word count
- "references": A list of {"number": N, "source": "paper title"} objects

Return ONLY valid JSON, no preamble."""


def generate_synthesis_memo(thread_id: str) -> dict:
    """Generate a synthesis memo from a thread."""
    thread = load_thread(thread_id)
    if thread is None:
        raise FileNotFoundError(f"Thread {thread_id} not found")
    if not thread.entries:
        return {"error": "Thread has no entries yet"}

    evidence = _collect_all_evidence(thread)

    # Build context
    context = "RESEARCH CONTEXT:\n\n"
    context += "Questions investigated:\n"
    for entry in thread.entries:
        context += f"- {entry.query}\n"

    context += "\n\nEVIDENCE FROM PAPERS:\n"
    for i, e in enumerate(evidence):
        context += (
            f"\n[Source {i+1}] \"{e['paper_title']}\" — §{e['section']}\n"
            f"{e['chunk_text']}\n"
        )

    context += f"\n\nANSWERS GENERATED:\n"
    for entry in thread.entries:
        context += f"\nQ: {entry.query}\nA: {entry.answer[:500]}\n"

    messages = [
        {"role": "system", "content": SYNTHESIS_PROMPT},
        {"role": "user", "content": context},
    ]

    raw = _call_llm(messages)
    if raw is None:
        return {"error": "LLM unavailable — cannot generate synthesis memo"}

    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        memo_data = json.loads(cleaned)
    except json.JSONDecodeError:
        memo_data = {
            "memo_text": raw,
            "word_count": len(raw.split()),
            "references": [],
        }

    result = {
        "thread_id": thread_id,
        "thread_title": thread.title,
        "artifact_type": "synthesis_memo",
        "memo_text": memo_data.get("memo_text", ""),
        "word_count": memo_data.get("word_count", 0),
        "references": memo_data.get("references", []),
    }
    _save_artifact(thread_id, "synthesis_memo", result)
    return result
