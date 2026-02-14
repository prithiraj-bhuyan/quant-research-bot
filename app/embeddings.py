"""
embeddings.py — PDF text extraction, formula-aware chunking, and FAISS index building
for the finance research portal.

Key design decisions:
  • PyMuPDF (fitz) for extraction — preserves layout and unicode math symbols.
  • Regex-based detection of math/formula blocks so the chunker never splits
    mid-equation (LaTeX delimiters, Unicode math runs, aligned environments).
  • Overlapping sliding-window chunker that respects formula boundaries.
  • sentence-transformers + FAISS-CPU for dense retrieval.
  • Incremental index updates — only new papers are processed on subsequent runs.
  • Resume support — interrupted builds pick up where they left off.
  • Auto full-rebuild if embedding model or chunk params change.
"""

import os
import re
import json
import logging
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
CHUNK_SIZE       = 500   # target tokens (≈ words) per chunk
CHUNK_OVERLAP    = 80    # overlap tokens between consecutive chunks
INDEX_DIR        = os.path.join(os.path.dirname(__file__), "..", "index")
LIBRARY_DIR      = os.path.join(os.path.dirname(__file__), "..", "quant_library")
INDEX_CONFIG     = "index_config.json"   # tracks model + params for rebuild detection

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
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
    start_char: int = 0
    end_char: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PDF TEXT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(filepath: str) -> str:
    """
    Extract text from a PDF while preserving mathematical notation as much
    as possible.  PyMuPDF keeps Unicode math symbols (∑, ∫, σ, √, etc.)
    intact, which is important for quant papers.
    """
    doc = fitz.open(filepath)
    pages: list[str] = []

    for page in doc:
        text = page.get_text("text")
        pages.append(text)

    doc.close()
    full_text = "\n\n".join(pages)

    # Light cleanup — collapse excessive whitespace but keep paragraph breaks
    full_text = re.sub(r"[ \t]+", " ", full_text)
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    return full_text.strip()


def extract_all_papers(library_dir: str = LIBRARY_DIR) -> list[dict]:
    """
    Walk the library directory, extract text from every PDF, and return
    a list of {meta, text} dicts.
    """
    papers = []
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
                text = extract_text_from_pdf(fpath)
                meta = PaperMeta(
                    title=fname.replace(".pdf", "").replace("_", " "),
                    category=cat_folder,
                    filename=fname,
                    filepath=fpath,
                )
                papers.append({"meta": meta, "text": text})
            except Exception as e:
                log.warning(f"⚠️  Failed on {fpath}: {e}")
    log.info(f"Extracted {len(papers)} papers total.")
    return papers


# ═══════════════════════════════════════════════════════════════════════════
# 2.  FORMULA-AWARE CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

# ---------- regex patterns that identify "math zones" ----------
_DISPLAY_MATH = re.compile(
    r"(\$\$.*?\$\$"
    r"|\\\[.*?\\\]"
    r"|\\begin\{(equation|align|gather|multline|"
    r"eqnarray|split|cases)\*?\}.*?"
    r"\\end\{\2\*?\})",
    re.DOTALL,
)

_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")

_UNICODE_MATH = re.compile(
    r"["
    r"\u0391-\u03C9"       # Greek letters  (Α-ω)
    r"\u2200-\u22FF"       # Mathematical Operators block
    r"\u2A00-\u2AFF"       # Supplemental Mathematical Operators
    r"\u00B1\u00D7\u00F7"  # ± × ÷
    r"\u221A-\u221E"       # √ ∛ ∜ ∝ ∞
    r"=<>≤≥≠≈∝∂∇∑∏∫"
    r"]{3,}"               # 3+ consecutive math-ish characters
)


def _is_inside_math(text: str, pos: int) -> bool:
    """Return True if *pos* falls inside any detected math span."""
    for pattern in (_DISPLAY_MATH, _INLINE_MATH, _UNICODE_MATH):
        for m in pattern.finditer(text):
            if m.start() <= pos < m.end():
                return True
    return False


def _find_safe_split(text: str, target_pos: int, window: int = 120) -> int:
    """
    Starting from *target_pos*, search for the nearest paragraph break or
    sentence boundary that is NOT inside a math expression.

    Priority:  paragraph break  >  sentence end (. or ;)  >  any whitespace
    Falls back to *target_pos* if nothing better is found.
    """
    lo = max(0, target_pos - window)
    hi = min(len(text), target_pos + window)
    region = text[lo:hi]

    candidates: list[tuple[int, int]] = []  # (abs_pos, priority)

    for m in re.finditer(r"\n\n+", region):
        abs_pos = lo + m.end()
        candidates.append((abs_pos, 0))

    for m in re.finditer(r"[.;]\s", region):
        abs_pos = lo + m.end()
        candidates.append((abs_pos, 1))

    for m in re.finditer(r"\s", region):
        abs_pos = lo + m.end()
        candidates.append((abs_pos, 2))

    candidates.sort(key=lambda c: (c[1], abs(c[0] - target_pos)))

    for abs_pos, _ in candidates:
        if not _is_inside_math(text, abs_pos):
            return abs_pos

    return target_pos


def chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split *text* into overlapping chunks of roughly *chunk_size* words,
    choosing split points that avoid breaking math expressions.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: list[str] = []
    word_starts: list[int] = []
    idx = 0
    for w in words:
        pos = text.index(w, idx)
        word_starts.append(pos)
        idx = pos + len(w)

    step = chunk_size - overlap
    i = 0
    while i < len(words):
        end_word = min(i + chunk_size, len(words))

        start_char = word_starts[i]
        end_char = (word_starts[end_word] if end_word < len(words)
                    else len(text))

        if end_word < len(words):
            end_char = _find_safe_split(text, end_char)

        chunk = text[start_char:end_char].strip()
        if chunk:
            chunks.append(chunk)

        if end_word >= len(words):
            break
        i += step

    return chunks


def chunk_paper(paper: dict,
                chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """
    Chunk a single paper dict (with 'meta' and 'text' keys) and return
    a list of Chunk objects.
    """
    raw_chunks = chunk_text(paper["text"], chunk_size, overlap)
    meta = paper["meta"]
    result: list[Chunk] = []

    offset = 0
    for idx, ct in enumerate(raw_chunks):
        start = paper["text"].find(ct, offset)
        end = start + len(ct) if start != -1 else -1
        result.append(Chunk(
            text=ct,
            chunk_index=idx,
            paper=meta,
            start_char=max(start, 0),
            end_char=max(end, 0),
        ))
        if start != -1:
            offset = start + 1

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3.  EMBEDDING GENERATION + FAISS INDEX
# ═══════════════════════════════════════════════════════════════════════════

_model_cache: dict[str, SentenceTransformer] = {}


def get_model(name: str = EMBED_MODEL_NAME) -> SentenceTransformer:
    """Load (or return cached) sentence-transformer model."""
    if name not in _model_cache:
        log.info(f"🔄 Loading embedding model: {name}")
        _model_cache[name] = SentenceTransformer(name)
    return _model_cache[name]


def embed_chunks(chunks: list[Chunk],
                 model_name: str = EMBED_MODEL_NAME,
                 batch_size: int = 64) -> np.ndarray:
    """
    Encode chunk texts → numpy array of shape (N, dim).
    """
    model = get_model(model_name)
    texts = [c.text for c in chunks]
    log.info(f"⚡ Embedding {len(texts)} chunks (batch_size={batch_size})…")
    embeddings = model.encode(texts, batch_size=batch_size,
                              show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS index.  IndexFlatIP (inner-product / cosine sim)
    because embeddings are L2-normalized.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info(f"✅ FAISS index built — {index.ntotal} vectors, dim={dim}")
    return index


def save_index(index: faiss.IndexFlatIP,
               chunks: list[Chunk],
               index_dir: str = INDEX_DIR,
               model_name: str = EMBED_MODEL_NAME,
               chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> None:
    """Persist FAISS index + chunk metadata + build config to disk."""
    os.makedirs(index_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))

    meta = []
    for c in chunks:
        d = {
            "text": c.text,
            "chunk_index": c.chunk_index,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "paper_title": c.paper.title,
            "paper_category": c.paper.category,
            "paper_filename": c.paper.filename,
            "paper_filepath": c.paper.filepath,
        }
        meta.append(d)

    with open(os.path.join(index_dir, "chunks_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save build config so we can detect model/param changes later
    config = {
        "model_name": model_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_dim": index.d,
        "total_vectors": index.ntotal,
    }
    with open(os.path.join(index_dir, INDEX_CONFIG), "w") as f:
        json.dump(config, f, indent=2)

    log.info(f"💾 Index + metadata + config saved to {index_dir}/")


def load_index(index_dir: str = INDEX_DIR):
    """Load FAISS index and chunk metadata from disk."""
    idx_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "chunks_meta.json")

    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"No FAISS index at {idx_path}")

    index = faiss.read_index(idx_path)
    with open(meta_path) as f:
        chunks_meta = json.load(f)

    log.info(f"📂 Loaded index ({index.ntotal} vectors) + {len(chunks_meta)} chunk records")
    return index, chunks_meta


def load_config(index_dir: str = INDEX_DIR) -> Optional[dict]:
    """Load the build config (model name, chunk params) if it exists."""
    cfg_path = os.path.join(index_dir, INDEX_CONFIG)
    if not os.path.exists(cfg_path):
        return None
    with open(cfg_path) as f:
        return json.load(f)


def _needs_full_rebuild(index_dir: str,
                        model_name: str,
                        chunk_size: int,
                        overlap: int) -> bool:
    """
    Check if a full rebuild is needed because the embedding model or
    chunking parameters changed since the last build.
    """
    cfg = load_config(index_dir)
    if cfg is None:
        return False  # no existing index
    if cfg.get("model_name") != model_name:
        log.warning(f"⚠️  Model changed: {cfg.get('model_name')} → {model_name}. Full rebuild required.")
        return True
    if cfg.get("chunk_size") != chunk_size or cfg.get("overlap") != overlap:
        log.warning(f"⚠️  Chunk params changed. Full rebuild required.")
        return True
    return False


def search(query: str,
           top_k: int = 5,
           index_dir: str = INDEX_DIR,
           model_name: str = EMBED_MODEL_NAME) -> list[dict]:
    """
    Embed a query string and return the top-k most similar chunks.
    """
    index, chunks_meta = load_index(index_dir)
    model = get_model(model_name)

    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_vec, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0:
            continue
        entry = dict(chunks_meta[idx])
        entry["score"] = float(score)
        entry["rank"] = rank + 1
        results.append(entry)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4.  FULL REBUILD  (wipes and recreates everything)
# ═══════════════════════════════════════════════════════════════════════════

def build_full_index(library_dir: str = LIBRARY_DIR,
                     index_dir: str = INDEX_DIR,
                     chunk_size: int = CHUNK_SIZE,
                     overlap: int = CHUNK_OVERLAP,
                     model_name: str = EMBED_MODEL_NAME) -> dict:
    """
    End-to-end: extract → chunk → embed → index → save.
    Wipes any existing index and rebuilds from scratch.
    """
    papers = extract_all_papers(library_dir)
    if not papers:
        return {"status": "error", "message": "No papers found"}

    all_chunks: list[Chunk] = []
    for p in papers:
        all_chunks.extend(chunk_paper(p, chunk_size, overlap))
    log.info(f"🔪 {len(all_chunks)} chunks from {len(papers)} papers")

    embeddings = embed_chunks(all_chunks, model_name)
    index = build_faiss_index(embeddings)
    save_index(index, all_chunks, index_dir, model_name, chunk_size, overlap)

    return {
        "status": "ok",
        "mode": "full_rebuild",
        "papers_processed": len(papers),
        "total_chunks": len(all_chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "index_path": index_dir,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  INCREMENTAL UPDATE + RESUME
# ═══════════════════════════════════════════════════════════════════════════

def update_index(library_dir: str = LIBRARY_DIR,
                 index_dir: str = INDEX_DIR,
                 chunk_size: int = CHUNK_SIZE,
                 overlap: int = CHUNK_OVERLAP,
                 model_name: str = EMBED_MODEL_NAME,
                 force_rebuild: bool = False) -> dict:
    """
    Smart build: only processes papers not already in the index.
    Also serves as a resume if a previous build was interrupted.

    Automatically triggers a full rebuild if the embedding model or
    chunk parameters have changed (or if force_rebuild=True).
    """
    # Check if params changed — if so, full rebuild
    if force_rebuild or _needs_full_rebuild(index_dir, model_name, chunk_size, overlap):
        log.info("🔄 Running full rebuild…")
        return build_full_index(library_dir, index_dir, chunk_size, overlap, model_name)

    # Load existing index + metadata, or start fresh
    try:
        index, existing_meta = load_index(index_dir)
        already_indexed = {m["paper_filepath"] for m in existing_meta}
        log.info(f"📂 Existing index has {len(existing_meta)} chunks "
                 f"from {len(already_indexed)} papers")
    except FileNotFoundError:
        index = None
        existing_meta = []
        already_indexed = set()
        log.info("No existing index — starting fresh")

    # Extract ALL papers from library
    all_papers = extract_all_papers(library_dir)
    new_papers = [p for p in all_papers if p["meta"].filepath not in already_indexed]

    if not new_papers:
        return {
            "status": "ok",
            "mode": "no_update_needed",
            "message": "All papers already indexed",
            "total_papers": len(all_papers),
            "total_chunks": len(existing_meta),
        }

    log.info(f"🆕 {len(new_papers)} new papers to process "
             f"(skipping {len(all_papers) - len(new_papers)} already indexed)")

    # Chunk + embed only the new papers
    new_chunks: list[Chunk] = []
    for p in new_papers:
        new_chunks.extend(chunk_paper(p, chunk_size, overlap))
    log.info(f"🔪 {len(new_chunks)} new chunks from {len(new_papers)} papers")

    new_embeddings = embed_chunks(new_chunks, model_name)

    # Build or extend the FAISS index
    if index is None:
        index = build_faiss_index(new_embeddings)
    else:
        index.add(new_embeddings)
        log.info(f"➕ Added {new_embeddings.shape[0]} vectors — "
                 f"index now has {index.ntotal} total")

    # Merge metadata: existing + new
    all_chunks_for_save: list[Chunk] = []

    for m in existing_meta:
        all_chunks_for_save.append(Chunk(
            text=m["text"],
            chunk_index=m["chunk_index"],
            start_char=m["start_char"],
            end_char=m["end_char"],
            paper=PaperMeta(
                title=m["paper_title"],
                category=m["paper_category"],
                filename=m["paper_filename"],
                filepath=m["paper_filepath"],
            ),
        ))
    all_chunks_for_save.extend(new_chunks)

    save_index(index, all_chunks_for_save, index_dir, model_name, chunk_size, overlap)

    return {
        "status": "ok",
        "mode": "incremental_update",
        "new_papers": len(new_papers),
        "new_chunks": len(new_chunks),
        "total_papers": len(already_indexed) + len(new_papers),
        "total_chunks": index.ntotal,
        "embedding_dim": index.d,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6.  DELETE PAPERS FROM INDEX
# ═══════════════════════════════════════════════════════════════════════════

def remove_papers(filepaths: list[str],
                  index_dir: str = INDEX_DIR,
                  model_name: str = EMBED_MODEL_NAME,
                  chunk_size: int = CHUNK_SIZE,
                  overlap: int = CHUNK_OVERLAP) -> dict:
    """
    Remove specific papers from the index by filepath.
    Since FAISS IndexFlatIP doesn't support selective deletion natively,
    we rebuild the index from the remaining chunks.
    """
    try:
        _, existing_meta = load_index(index_dir)
    except FileNotFoundError:
        return {"status": "error", "message": "No index exists"}

    remove_set = set(filepaths)
    keep_meta = [m for m in existing_meta if m["paper_filepath"] not in remove_set]
    removed_count = len(existing_meta) - len(keep_meta)

    if removed_count == 0:
        return {"status": "ok", "message": "No matching papers found in index"}

    keep_chunks = [
        Chunk(
            text=m["text"],
            chunk_index=m["chunk_index"],
            start_char=m["start_char"],
            end_char=m["end_char"],
            paper=PaperMeta(
                title=m["paper_title"],
                category=m["paper_category"],
                filename=m["paper_filename"],
                filepath=m["paper_filepath"],
            ),
        )
        for m in keep_meta
    ]

    if keep_chunks:
        embeddings = embed_chunks(keep_chunks, model_name)
        index = build_faiss_index(embeddings)
        save_index(index, keep_chunks, index_dir, model_name, chunk_size, overlap)
    else:
        for fname in ("faiss.index", "chunks_meta.json", INDEX_CONFIG):
            path = os.path.join(index_dir, fname)
            if os.path.exists(path):
                os.remove(path)

    return {
        "status": "ok",
        "removed_chunks": removed_count,
        "remaining_chunks": len(keep_meta),
    }


# ---------------------------------------------------------------------------
# Quick CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    usage = """
Usage:
  python embeddings.py build          Full rebuild from scratch
  python embeddings.py update         Incremental update (only new papers)
  python embeddings.py search <query> Semantic search
  python embeddings.py status         Show index stats
"""

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "build":
        stats = build_full_index()
        print(json.dumps(stats, indent=2))

    elif cmd == "update":
        stats = update_index()
        print(json.dumps(stats, indent=2))

    elif cmd == "search":
        query = " ".join(sys.argv[2:]) or "portfolio optimization risk"
        results = search(query, top_k=5)
        for r in results:
            print(f"\n[{r['rank']}] score={r['score']:.4f}  "
                  f"({r['paper_category']}) {r['paper_title']}")
            print(f"    {r['text'][:200]}…")

    elif cmd == "status":
        cfg = load_config()
        if cfg:
            print(json.dumps(cfg, indent=2))
        else:
            print("No index built yet.")

    else:
        print(usage)