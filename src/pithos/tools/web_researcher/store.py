"""Per-run ChromaDB-backed excerpt store with hash + semantic dedup."""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Optional

from .models import Excerpt

logger = logging.getLogger(__name__)


def _sanitize_collection_name(name: str) -> str:
    """ChromaDB collection names must be 3-63 chars, alphanum + _- + ., start/end alnum."""
    sanitized = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", name).strip("._-")
    if len(sanitized) < 3:
        sanitized = (sanitized + "_run")[:63]
    return sanitized[:63]


class ExcerptStore:
    """Stores excerpts in a per-run ChromaDB collection.

    Dedup happens in two stages on :meth:`add`:
    1. **Hash dedup** - rejected if an entry with the same ``content_hash``
       is already present.
    2. **Semantic dedup** - the candidate is embedded and queried against
       the collection; if the nearest neighbour's distance is below
       ``1 - similarity_threshold`` (i.e. very similar) it's rejected.

    Returns True iff the excerpt was actually inserted.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./data/research",
        similarity_threshold: float = 0.92,
        client: Optional[Any] = None,
    ) -> None:
        """Initialise the store.

        Args:
            collection_name: Logical name for this run's collection.
            persist_directory: Where ChromaDB files live.
            similarity_threshold: Cosine similarity above which excerpts are
                considered duplicates (0..1). Distance threshold becomes
                ``1 - similarity_threshold``.
            client: Optional pre-built ChromaDB client (used by tests).
        """
        if not (0.0 < similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in (0, 1]")

        self.collection_name = _sanitize_collection_name(collection_name)
        self.persist_directory = persist_directory
        self.similarity_threshold = similarity_threshold
        self._distance_threshold = 1.0 - similarity_threshold
        self._hashes: set[str] = set()
        self._excerpts: list[Excerpt] = []
        self._counter = 0

        if client is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError as exc:
                raise RuntimeError(
                    "ChromaDB is required for ExcerptStore. Install with: pip install chromadb"
                ) from exc

            os.makedirs(persist_directory, exist_ok=True)
            client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

        self._client = client
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"created": datetime.now().isoformat(), "kind": "web_research"},
        )

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, excerpt: Excerpt) -> bool:
        """Add ``excerpt`` if it passes hash + semantic dedup. Returns True on insert."""
        text = (excerpt.text or "").strip()
        if not text:
            return False

        if excerpt.content_hash in self._hashes:
            return False

        # Semantic dedup - only if we already have something to compare against.
        if self._counter > 0 and self._distance_threshold < 1.0:
            try:
                hits = self._collection.query(
                    query_texts=[text],
                    n_results=1,
                )
                distances = (hits.get("distances") or [[None]])[0] if hits else [None]
                nearest = distances[0] if distances else None
                if nearest is not None and nearest <= self._distance_threshold:
                    return False
            except Exception as exc:  # pragma: no cover - chroma backend errors
                logger.debug("semantic dedup query failed: %s", exc)

        entry_id = f"ex_{self._counter}_{excerpt.content_hash[:12]}"
        self._counter += 1
        metadata = {
            "url": excerpt.url or "",
            "title": excerpt.title or "",
            "relevance": (
                float(excerpt.relevance) if excerpt.relevance is not None else -1.0
            ),
            "ts": time.time(),
        }
        for k, v in (excerpt.metadata or {}).items():
            if isinstance(v, (str, int, float, bool)):
                metadata[f"m_{k}"] = v

        try:
            self._collection.add(ids=[entry_id], documents=[text], metadatas=[metadata])
        except Exception as exc:  # pragma: no cover
            logger.warning("ExcerptStore add failed: %s", exc)
            return False

        self._hashes.add(excerpt.content_hash)
        excerpt.metadata.setdefault("id", entry_id)
        self._excerpts.append(excerpt)
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def all(self) -> list[Excerpt]:
        """Return all stored excerpts in insertion order."""
        return list(self._excerpts)

    def __len__(self) -> int:
        return len(self._excerpts)

    def sources(self) -> list[str]:
        """Return unique source URLs in insertion order."""
        seen: set[str] = set()
        out: list[str] = []
        for ex in self._excerpts:
            if ex.url and ex.url not in seen:
                seen.add(ex.url)
                out.append(ex.url)
        return out

    def query(self, text: str, k: int = 5) -> list[Excerpt]:
        """Return the ``k`` most relevant stored excerpts for ``text``."""
        if not text or self._counter == 0:
            return []
        try:
            hits = self._collection.query(query_texts=[text], n_results=k)
        except Exception:  # pragma: no cover
            return []
        ids = (hits.get("ids") or [[]])[0]
        by_id = {ex.metadata.get("id"): ex for ex in self._excerpts}
        return [by_id[i] for i in ids if i in by_id]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Delete this run's collection (used unless ``keep_collection`` is set)."""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception as exc:  # pragma: no cover
            logger.debug("collection delete failed: %s", exc)
