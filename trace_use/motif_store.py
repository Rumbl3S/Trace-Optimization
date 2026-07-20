"""PersistentMotifStore — MotifStore that saves and loads across sessions.

Drop-in replacement for MotifStore. Persists to a JSON file so motifs
learned on Monday are still available on Friday.

    store = PersistentMotifStore(embedder)          # loads ~/.trace_use/motifs.json
    store = PersistentMotifStore(embedder, path="./my_project_motifs.json")

Embeddings are stored as float lists alongside the motif metadata.
Writes are atomic (write to .tmp then os.replace) to survive interruption.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from .brain import FailureMotif, MotifStore
from .config import _DEFAULT_STORAGE_PATH


class PersistentMotifStore(MotifStore):
    """MotifStore that persists motifs to disk across Python sessions."""

    def __init__(self, embedder: Callable, path: str = _DEFAULT_STORAGE_PATH):
        super().__init__(embedder)
        self._path    = path
        self._io_lock = threading.Lock()
        self._load()

    @property
    def path(self) -> str:
        return self._path

    # ── overrides that trigger save ───────────────────────────────────────────

    def add(self, motif: FailureMotif, signal_text: str = "") -> None:
        super().add(motif, signal_text)
        self._save()

    def update(self, update_id: str, motif: FailureMotif) -> bool:
        changed = super().update(update_id, motif)
        if changed:
            self._save()
        return changed

    def clear(self) -> None:
        """Remove all motifs and wipe the backing file."""
        with self._lock:
            self._motifs.clear()
            self._vecs.clear()
        self._save()

    # ── persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        dir_ = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(dir_, exist_ok=True)
        with self._lock:
            records = [
                {
                    "id":                  m.id,
                    "name":                m.name,
                    "description":         m.description,
                    "required_condition":  m.required_condition,
                    "violation_condition": m.violation_condition,
                    "recommendation":      m.recommendation,
                    "examples":            m.examples,
                    "source":              m.source,
                    "embedding":           v.tolist(),
                }
                for v, m in self._vecs
            ]
        payload = {
            "version":  1,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "motifs":   records,
        }
        with self._io_lock:
            tmp = self._path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, self._path)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with self._io_lock:
                with open(self._path) as f:
                    data = json.load(f)
            loaded = 0
            for rec in data.get("motifs", []):
                emb = rec.pop("embedding", None)
                motif = FailureMotif(
                    id                  = rec["id"],
                    name                = rec["name"],
                    description         = rec["description"],
                    required_condition  = rec.get("required_condition", ""),
                    violation_condition = rec.get("violation_condition", ""),
                    recommendation      = rec.get("recommendation", ""),
                    examples            = rec.get("examples", []),
                    source              = rec.get("source", "learned"),
                )
                if emb:
                    vec  = np.asarray(emb, dtype="float32")
                    norm = np.linalg.norm(vec)
                    if norm > 1e-9:
                        vec /= norm
                    with self._lock:
                        self._motifs.append(motif)
                        self._vecs.append((vec, motif))
                else:
                    super().add(motif)   # re-embed if no stored embedding
                loaded += 1
            if loaded:
                print(f"[MOTIF STORE] Loaded {loaded} motif(s) from {self._path}")
        except Exception as e:
            print(f"[MOTIF STORE] Load failed ({e!r}) — starting fresh")
