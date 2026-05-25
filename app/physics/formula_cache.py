from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = ROOT / "outputs" / "cache" / "physics_formula_cache.sqlite3"


def normalize_formula_query(question: str) -> str:
    text = question.lower()
    text = re.sub(r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b", "<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cache_key(question: str) -> str:
    return hashlib.sha256(normalize_formula_query(question).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CachedFormulaContext:
    key: str
    normalized_query: str
    search_query: str
    context: str
    sources: list[dict[str, Any]]
    created_at: str
    last_used_at: str
    hits: int


class FormulaCache:
    def __init__(self, path: Path = DEFAULT_CACHE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS formula_contexts (
                    key TEXT PRIMARY KEY,
                    normalized_query TEXT NOT NULL,
                    search_query TEXT NOT NULL,
                    context TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL,
                    hits INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_formula_contexts_query ON formula_contexts(normalized_query)")

    def get(self, question: str) -> CachedFormulaContext | None:
        key = cache_key(question)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM formula_contexts WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE formula_contexts SET hits = hits + 1, last_used_at = ? WHERE key = ?",
                (now, key),
            )
        return CachedFormulaContext(
            key=row["key"],
            normalized_query=row["normalized_query"],
            search_query=row["search_query"],
            context=row["context"],
            sources=json.loads(row["sources_json"] or "[]"),
            created_at=row["created_at"],
            last_used_at=now,
            hits=int(row["hits"]) + 1,
        )

    def set(self, question: str, search_query: str, context: str, sources: list[dict[str, Any]]) -> CachedFormulaContext:
        key = cache_key(question)
        normalized = normalize_formula_query(question)
        now = datetime.now(timezone.utc).isoformat()
        sources_json = json.dumps(sources, ensure_ascii=True, sort_keys=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO formula_contexts
                    (key, normalized_query, search_query, context, sources_json, created_at, last_used_at, hits)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(key) DO UPDATE SET
                    search_query = excluded.search_query,
                    context = excluded.context,
                    sources_json = excluded.sources_json,
                    last_used_at = excluded.last_used_at
                """,
                (key, normalized, search_query, context, sources_json, now, now),
            )
        return CachedFormulaContext(
            key=key,
            normalized_query=normalized,
            search_query=search_query,
            context=context,
            sources=sources,
            created_at=now,
            last_used_at=now,
            hits=0,
        )
