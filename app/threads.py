"""
threads.py — Research thread persistence (file-based JSON).

A thread captures an entire research session: every query, the retrieved
evidence chunks, the generated answer, and structured citations.
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THREADS_DIR = str(PROJECT_ROOT / "threads")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ThreadEntry:
    query: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    evidence_chunks: list[dict] = field(default_factory=list)
    trust_signals: dict = field(default_factory=dict)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class Thread:
    id: str = ""
    title: str = "Untitled Thread"
    created_at: str = ""
    updated_at: str = ""
    entries: list[ThreadEntry] = field(default_factory=list)

    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def _thread_path(thread_id: str) -> str:
    return os.path.join(THREADS_DIR, f"{thread_id}.json")


def save_thread(thread: Thread) -> str:
    """Save a thread to disk.  Returns the thread id."""
    thread.updated_at = datetime.now(timezone.utc).isoformat()
    os.makedirs(THREADS_DIR, exist_ok=True)
    data = asdict(thread)
    with open(_thread_path(thread.id), "w") as f:
        json.dump(data, f, indent=2)
    return thread.id


def load_thread(thread_id: str) -> Optional[Thread]:
    """Load a thread from disk, or return None."""
    path = _thread_path(thread_id)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    entries = [ThreadEntry(**e) for e in data.get("entries", [])]
    return Thread(
        id=data["id"],
        title=data["title"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        entries=entries,
    )


def list_threads() -> list[dict]:
    """Return summaries of all threads (id, title, created, entry count)."""
    os.makedirs(THREADS_DIR, exist_ok=True)
    summaries = []
    for fname in sorted(os.listdir(THREADS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(THREADS_DIR, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            summaries.append({
                "id": data["id"],
                "title": data["title"],
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "entry_count": len(data.get("entries", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    # Most recent first
    summaries.sort(key=lambda s: s["updated_at"], reverse=True)
    return summaries


def delete_thread(thread_id: str) -> bool:
    """Delete a thread file.  Returns True if deleted."""
    path = _thread_path(thread_id)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def add_entry_to_thread(thread_id: str, entry: ThreadEntry) -> Thread:
    """Append an entry to an existing thread and save."""
    thread = load_thread(thread_id)
    if thread is None:
        raise FileNotFoundError(f"Thread {thread_id} not found")
    thread.entries.append(entry)
    save_thread(thread)
    return thread
