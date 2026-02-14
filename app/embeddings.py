"""
embeddings.py — PDF extraction, section-aware chunking, FAISS + BM25 indexing.

Phase 2 upgrades:
  • Section-aware chunking — detects paper sections (Abstract, Introduction,
    Methodology, Results, etc.) and never splits across section boundaries.
  • BM25 sparse index alongside FAISS dense index for hybrid retrieval.
  • Chunk metadata includes section name + page number for structured citations.
"""

import os
import re
import json
import logging
import pickle
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 80
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
INDEX_DIR        = str(PROJECT_ROOT / "index")
LIBRARY_DIR      = str(PROJECT_ROOT / "quant_library")
INDEX_CONFIG     = "index_config.json"


@dataclass
class PaperMeta:
    title: str
    category: str
    filename: str
    filepath: str

@dataclass
class Chunk:
    text: str
    chunk_index: int
    paper: PaperMeta
    section: str = "Unknown"
    page_number: int = -1
    start_char: int = 0
    end_char: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PDF TEXT EXTRACTION (page-aware)
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(filepath: str) -> list[dict]:
    doc = fitz.open(filepath)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append({"page": i + 1, "text": text.strip()})
    doc.close()
    return pages


def extract_all_papers(library_dir: str = LIBRARY_DIR) -> list[dict]:
    papers = []
    if not os.path.isdir(library_dir):
        log.warning(f"Library dir not found: {library_dir}")
        return papers

    for cat_folder in sorted(os.listdir(library_dir)):
        cat_path = os.path.join(library_dir, cat_folder)
        if not os.path.isdir(cat_path):
            continue
        for fname in sorted(os.listdir(cat_path)):
            if not fname.lower().endswith(".pdf"):
                continue
            fpath = os.path.join(cat_path, fname)
            log.info(f"📄 Extracting: {fpath}")
            try:
                pages = extract_text_from_pdf(fpath)
                meta = PaperMeta(
                    title=fname.replace(".pdf", "").replace("_", " "),
                    category=cat_folder,
                    filename=fname,
                    filepath=fpath,
                )
                papers.append({"meta": meta, "pages": pages})
            except Exception as e:
                log.warning(f"⚠️  Failed on {fpath}: {e}")
    log.info(f"Extracted {len(papers)} papers total.")
    return papers


# ═══════════════════════════════════════════════════════════════════════════
# 2.  SECTION-AWARE CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

_SECTION_PATTERNS = re.compile(
    r"^\s*(?:\d+\.?\s*)?"
    r"(Abstract|Introduction|Background|Related\s+Work|Literature\s+Review|"
    r"Methodology|Methods?|Model|Framework|Data(?:\s+and\s+Methods?)?|"
    r"Theoretical\s+Framework|Problem\s+(?:Setup|Formulation|Statement)|"
    r"Approach|Algorithm|Implementation|"
    r"Results?|Experiments?|Empirical\s+(?:Results?|Analysis)|Findings|"
    r"Analysis|Discussion|Evaluation|Performance|"
    r"Conclusion|Conclusions?|Summary|"
    r"Future\s+Work|Limitations|"
    r"Appendix|Appendices|References|Bibliography|"
    r"Acknowledgm?ents?|Notation|Preliminaries|"
    r"Proof|Theorem|Lemma|Proposition|Corollary|Remark|Definition|"
    r"Portfolio\s+(?:Construction|Optimization|Selection)|"
    r"Risk\s+(?:Management|Measures?|Analysis)|"
    r"Market\s+(?:Model|Microstructure|Impact)|"
    r"Pricing|Hedging|Calibration|Backtesting|Simulation)"
    r"\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_DISPLAY_MATH = re.compile(
    r"(\$\$.*?\$\$|\\\[.*?\\\]"
    r"|\\begin\{(equation|align|gather|multline|eqnarray|split|cases)\*?\}.*?"
    r"\\end\{\2\*?\})",
    re.DOTALL,
)
_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
_UNICODE_MATH = re.compile(
    r"[\u0391-\u03C9\u2200-\u22FF\u2A00-\u2AFF"
    r"\u00B1\u00D7\u00F7\u221A-\u221E"
    r"=<>≤≥≠≈∝∂∇∑∏∫]{3,}"
)


def _is_inside_math(text: str, pos: int) -> bool:
    for pat in (_DISPLAY_MATH, _INLINE_MATH, _UNICODE_MATH):
        for m in pat.finditer(text):
            if m.start() <= pos < m.end():
                return True
    return False


def _find_safe_split(text: str, target: int, window: int = 120) -> int:
    lo, hi = max(0, target - window), min(len(text), target + window)
    region = text[lo:hi]
    candidates = []
    for m in re.finditer(r"\n\n+", region):
        candidates.append((lo + m.end(), 0))
    for m in re.finditer(r"[.;]\s", region):
        candidates.append((lo + m.end(), 1))
    for m in re.finditer(r"\s", region):
        candidates.append((lo + m.end(), 2))
    candidates.sort(key=lambda c: (c[1], abs(c[0] - target)))
    for abs_pos, _ in candidates:
        if not _is_inside_math(text, abs_pos):
            return abs_pos
    return target


def _detect_sections(full_text: str) -> list[dict]:
    matches = list(_SECTION_PATTERNS.finditer(full_text))
    if not matches:
        return [{"section_name": "Full Text", "text": full_text,
                 "start_char": 0, "end_char": len(full_text)}]

    sections = []
    if matches[0].start() > 0:
        pre = full_text[:matches[0].start()].strip()
        if pre:
            sections.append({"section_name": "Header/Preamble", "text": pre,
                             "start_char": 0, "end_char": matches[0].start()})

    for i, match in enumerate(matches):
        name = re.sub(r"^\d+\.?\s*", "", match.group(0).strip()).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()
        if text:
            sections.append({"section_name": name, "text": text,
                             "start_char": start, "end_char": end})
    return sections


def _chunk_section_text(text: str, chunk_size: int = CHUNK_SIZE,
                        overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks, word_starts, idx = [], [], 0
    for w in words:
        pos = text.index(w, idx)
        word_starts.append(pos)
        idx = pos + len(w)

    step = chunk_size - overlap
    i = 0
    while i < len(words):
        end_word = min(i + chunk_size, len(words))
        sc = word_starts[i]
        ec = word_starts[end_word] if end_word < len(words) else len(text)
        if end_word < len(words):
            ec = _find_safe_split(text, ec)
        chunk = text[sc:ec].strip()
        if chunk:
            chunks.append(chunk)
        if end_word >= len(words):
            break
        i += step
    return chunks


def chunk_paper(paper: dict, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    meta = paper["meta"]
    pages = paper["pages"]
    full_text = "\n\n".join(p["text"] for p in pages)

    page_boundaries = []
    offset = 0
    for p in pages:
        page_boundaries.append((offset, offset + len(p["text"]), p["page"]))
        offset += len(p["text"]) + 2

    def _get_page(char_pos):
        for s, e, pn in page_boundaries:
            if s <= char_pos < e:
                return pn
        return -1

    sections = _detect_sections(full_text)
    result, gidx = [], 0

    for sec in sections:
        for ct in _chunk_section_text(sec["text"], chunk_size, overlap):
            start = sec["text"].find(ct)
            abs_start = sec["start_char"] + (start if start != -1 else 0)
            result.append(Chunk(
                text=ct, chunk_index=gidx, paper=meta,
                section=sec["section_name"],
                page_number=_get_page(abs_start),
                start_char=abs_start, end_char=abs_start + len(ct),
            ))
            gidx += 1
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3.  INDEX BUILD + SEARCH
# ═══════════════════════════════════════════════════════════════════════════

_model_cache: dict[str, SentenceTransformer] = {}

def get_model(name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    if name not in _model_cache:
        log.info(f"🔄 Loading model: {name}")
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


def embed_chunks(chunks: list[Chunk], model_name: str = EMBED_MODEL_NAME,
                 batch_size: int = 64) -> np.ndarray:
    model = get_model(model_name)
    texts = [c.text for c in chunks]
    log.info(f"⚡ Embedding {len(texts)} chunks…")
    emb = model.encode(texts, batch_size=batch_size,
                       show_progress_bar=True, normalize_embeddings=True)
    return np.array(emb, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embeddings)
    log.info(f"✅ FAISS: {idx.ntotal} vectors, dim={dim}")
    return idx


def build_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    tokenized = [c.text.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    log.info(f"✅ BM25: {len(tokenized)} docs")
    return bm25


def _chunk_to_dict(c: Chunk) -> dict:
    return {
        "text": c.text, "chunk_index": c.chunk_index,
        "section": c.section, "page_number": c.page_number,
        "start_char": c.start_char, "end_char": c.end_char,
        "paper_title": c.paper.title, "paper_category": c.paper.category,
        "paper_filename": c.paper.filename, "paper_filepath": c.paper.filepath,
    }

def _dict_to_chunk(m: dict) -> Chunk:
    return Chunk(
        text=m["text"], chunk_index=m["chunk_index"],
        section=m.get("section", "Unknown"), page_number=m.get("page_number", -1),
        start_char=m["start_char"], end_char=m["end_char"],
        paper=PaperMeta(title=m["paper_title"], category=m["paper_category"],
                        filename=m["paper_filename"], filepath=m["paper_filepath"]),
    )


def save_index(faiss_idx, bm25, chunks, index_dir=INDEX_DIR,
               model_name=EMBED_MODEL_NAME, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(faiss_idx, os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    meta = [_chunk_to_dict(c) for c in chunks]
    with open(os.path.join(index_dir, "chunks_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    config = {"model_name": model_name, "chunk_size": chunk_size,
              "overlap": overlap, "embedding_dim": faiss_idx.d,
              "total_vectors": faiss_idx.ntotal}
    with open(os.path.join(index_dir, INDEX_CONFIG), "w") as f:
        json.dump(config, f, indent=2)
    log.info(f"💾 Saved to {index_dir}/")


def load_index(index_dir=INDEX_DIR):
    idx_path = os.path.join(index_dir, "faiss.index")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"No index at {idx_path}")
    faiss_idx = faiss.read_index(idx_path)
    with open(os.path.join(index_dir, "chunks_meta.json")) as f:
        chunks_meta = json.load(f)
    bm25 = None
    bm25_path = os.path.join(index_dir, "bm25.pkl")
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
    return faiss_idx, bm25, chunks_meta


def load_config(index_dir=INDEX_DIR) -> Optional[dict]:
    p = os.path.join(index_dir, INDEX_CONFIG)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _needs_rebuild(index_dir, model_name, chunk_size, overlap):
    cfg = load_config(index_dir)
    if not cfg:
        return False
    return (cfg.get("model_name") != model_name or
            cfg.get("chunk_size") != chunk_size or
            cfg.get("overlap") != overlap)


# ── Hybrid search ──────────────────────────────────────────────────────

def hybrid_search(query: str, top_k: int = 10, dense_weight: float = 0.6,
                  index_dir: str = INDEX_DIR, model_name: str = EMBED_MODEL_NAME) -> list[dict]:
    faiss_idx, bm25, chunks_meta = load_index(index_dir)
    model = get_model(model_name)

    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    fetch_k = min(top_k * 3, faiss_idx.ntotal)
    dense_scores, dense_ids = faiss_idx.search(q_vec, fetch_k)

    bm25_scores = bm25.get_scores(query.lower().split()) if bm25 else np.zeros(len(chunks_meta))

    rrf_k, rrf = 60, {}
    for rank, idx in enumerate(dense_ids[0]):
        if idx >= 0:
            rrf[int(idx)] = rrf.get(int(idx), 0) + dense_weight / (rrf_k + rank + 1)

    sparse_ranking = np.argsort(-bm25_scores)[:fetch_k]
    sw = 1.0 - dense_weight
    for rank, idx in enumerate(sparse_ranking):
        if bm25_scores[idx] > 0:
            rrf[int(idx)] = rrf.get(int(idx), 0) + sw / (rrf_k + rank + 1)

    sorted_r = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for rank, (idx, score) in enumerate(sorted_r):
        entry = dict(chunks_meta[idx])
        entry["score"] = float(score)
        entry["rank"] = rank + 1
        results.append(entry)
    return results


def dense_search(query, top_k=5, index_dir=INDEX_DIR, model_name=EMBED_MODEL_NAME):
    faiss_idx, _, meta = load_index(index_dir)
    model = get_model(model_name)
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = faiss_idx.search(q, top_k)
    results = []
    for rank, (s, i) in enumerate(zip(scores[0], ids[0])):
        if i < 0: continue
        e = dict(meta[i]); e["score"] = float(s); e["rank"] = rank + 1
        results.append(e)
    return results


# ── Build pipelines ────────────────────────────────────────────────────

def build_full_index(library_dir=LIBRARY_DIR, index_dir=INDEX_DIR,
                     chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
                     model_name=EMBED_MODEL_NAME) -> dict:
    papers = extract_all_papers(library_dir)
    if not papers:
        return {"status": "error", "message": "No papers found"}
    all_chunks = []
    for p in papers:
        all_chunks.extend(chunk_paper(p, chunk_size, overlap))
    log.info(f"🔪 {len(all_chunks)} chunks from {len(papers)} papers")
    emb = embed_chunks(all_chunks, model_name)
    fi = build_faiss_index(emb)
    bi = build_bm25_index(all_chunks)
    save_index(fi, bi, all_chunks, index_dir, model_name, chunk_size, overlap)
    return {"status": "ok", "mode": "full_rebuild",
            "papers": len(papers), "chunks": len(all_chunks), "dim": int(emb.shape[1])}


def update_index(library_dir=LIBRARY_DIR, index_dir=INDEX_DIR,
                 chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
                 model_name=EMBED_MODEL_NAME, force_rebuild=False) -> dict:
    if force_rebuild or _needs_rebuild(index_dir, model_name, chunk_size, overlap):
        return build_full_index(library_dir, index_dir, chunk_size, overlap, model_name)
    try:
        _, _, existing = load_index(index_dir)
        already = {m["paper_filepath"] for m in existing}
    except FileNotFoundError:
        existing, already = [], set()
    all_papers = extract_all_papers(library_dir)
    new = [p for p in all_papers if p["meta"].filepath not in already]
    if not new:
        return {"status": "ok", "mode": "no_update", "papers": len(all_papers), "chunks": len(existing)}
    return build_full_index(library_dir, index_dir, chunk_size, overlap, model_name)


def remove_papers(filepaths, index_dir=INDEX_DIR, model_name=EMBED_MODEL_NAME,
                  chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    try:
        _, _, existing = load_index(index_dir)
    except FileNotFoundError:
        return {"status": "error", "message": "No index"}
    rm = set(filepaths)
    keep = [m for m in existing if m["paper_filepath"] not in rm]
    removed = len(existing) - len(keep)
    if not removed:
        return {"status": "ok", "message": "Nothing to remove"}
    if keep:
        chunks = [_dict_to_chunk(m) for m in keep]
        emb = embed_chunks(chunks, model_name)
        fi = build_faiss_index(emb)
        bi = build_bm25_index(chunks)
        save_index(fi, bi, chunks, index_dir, model_name, chunk_size, overlap)
    else:
        for f in ("faiss.index", "bm25.pkl", "chunks_meta.json", INDEX_CONFIG):
            p = os.path.join(index_dir, f)
            if os.path.exists(p): os.remove(p)
    return {"status": "ok", "removed": removed, "remaining": len(keep)}