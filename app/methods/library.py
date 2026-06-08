"""Persistent, scored registry of Method instances.

The library holds every Method the system can apply to a problem:
  * Built-ins from ``app/methods/impl/``.
  * User-registered methods from config.
  * Methods discovered at runtime by ``app.methods.discovery`` (Level 6).

Persistence
-----------
Discovered methods (and historical scores) are stored as JSON under
``models/methods.json`` so a method found on question N is reused on
question N+1 without another web search. Built-ins are NOT persisted — they
are reloaded from code on every startup so a code change is the source of
truth for them. The persistence file is append-then-merge: only entries with
``source ∈ {DISCOVERED_*, USER_REGISTERED}`` round-trip.

Scoring
-------
Each method tracks ``(uses, successes, accumulated_confidence)`` so the
planner can prefer methods that have actually worked. A method's score does
NOT influence its applicability; it is a tiebreaker after applicability is
computed.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.methods.types import Method, MethodFamily, MethodSource

# Library file lives next to the model manifest so all stateful artifacts
# share a directory.
_DEFAULT_LIBRARY_PATH = Path(__file__).resolve().parents[2] / "models" / "methods.json"


@dataclass
class MethodStats:
    """Lifetime usage statistics for a single Method."""

    uses: int = 0
    successes: int = 0
    abstains: int = 0
    errors: int = 0
    accumulated_confidence: float = 0.0
    last_used_iso: str | None = None

    @property
    def success_rate(self) -> float:
        if self.uses == 0:
            return 0.0
        return self.successes / self.uses

    @property
    def avg_confidence(self) -> float:
        if self.uses == 0:
            return 0.0
        return self.accumulated_confidence / self.uses

    def to_dict(self) -> dict[str, Any]:
        return {
            "uses": self.uses,
            "successes": self.successes,
            "abstains": self.abstains,
            "errors": self.errors,
            "accumulated_confidence": self.accumulated_confidence,
            "last_used_iso": self.last_used_iso,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MethodStats":
        return cls(
            uses=int(data.get("uses", 0) or 0),
            successes=int(data.get("successes", 0) or 0),
            abstains=int(data.get("abstains", 0) or 0),
            errors=int(data.get("errors", 0) or 0),
            accumulated_confidence=float(data.get("accumulated_confidence", 0.0) or 0.0),
            last_used_iso=data.get("last_used_iso"),
        )


@dataclass
class _LibraryEntry:
    method: Method
    stats: MethodStats = field(default_factory=MethodStats)


class MethodLibrary:
    """Thread-safe registry of methods, with optional JSON persistence.

    Methods are keyed by ``method_id`` (unique per logical method) AND by
    ``signature`` (a structural fingerprint) so that two LLM-discovered methods
    that boil down to the same procedure collapse to a single entry.
    """

    def __init__(self, *, persistence_path: Path | None = None) -> None:
        self._entries: dict[str, _LibraryEntry] = {}
        self._signatures: dict[str, str] = {}  # signature -> method_id
        self._lock = threading.RLock()
        self._persistence_path = persistence_path

    # ----- registration -----------------------------------------------------

    def register(self, method: Method) -> bool:
        """Register a method. Returns True if newly added.

        Re-registering the same method_id is a no-op (idempotent reloads).
        Registering a method whose ``signature`` matches an existing entry is
        treated as a duplicate and the new copy is dropped — but if the new
        copy has a higher source-trust (e.g. promoting a provisional discovery
        to verified) the entry is upgraded in place.
        """
        with self._lock:
            sig = method.signature()
            if method.method_id in self._entries:
                # Idempotent: same id already there.
                existing = self._entries[method.method_id].method
                # Allow source upgrade: provisional -> verified.
                if (existing.source == MethodSource.DISCOVERED_PROVISIONAL
                        and method.source == MethodSource.DISCOVERED_VERIFIED):
                    self._entries[method.method_id].method = method
                return False
            if sig in self._signatures:
                # Same procedure already registered under a different id.
                # Don't add a duplicate; keep the existing one.
                return False
            self._entries[method.method_id] = _LibraryEntry(method=method)
            self._signatures[sig] = method.method_id
            return True

    def deregister(self, method_id: str) -> bool:
        """Remove a method (e.g. discovered method that proved unreliable)."""
        with self._lock:
            entry = self._entries.pop(method_id, None)
            if entry is None:
                return False
            sig = entry.method.signature()
            self._signatures.pop(sig, None)
            return True

    # ----- lookup -----------------------------------------------------------

    def all(self) -> list[Method]:
        """Return all registered methods (snapshot)."""
        with self._lock:
            return [entry.method for entry in self._entries.values()]

    def by_family(self, family: MethodFamily) -> list[Method]:
        with self._lock:
            return [
                entry.method
                for entry in self._entries.values()
                if entry.method.family == family
            ]

    def get(self, method_id: str) -> Method | None:
        with self._lock:
            entry = self._entries.get(method_id)
            return entry.method if entry else None

    def stats_for(self, method_id: str) -> MethodStats | None:
        with self._lock:
            entry = self._entries.get(method_id)
            return entry.stats if entry else None

    # ----- scoring updates --------------------------------------------------

    def record_use(self, method_id: str, *, success: bool, abstained: bool,
                   error: bool, confidence: float, when_iso: str) -> None:
        """Update lifetime stats after a method runs."""
        with self._lock:
            entry = self._entries.get(method_id)
            if entry is None:
                return
            stats = entry.stats
            stats.uses += 1
            if success:
                stats.successes += 1
            if abstained:
                stats.abstains += 1
            if error:
                stats.errors += 1
            stats.accumulated_confidence += float(confidence)
            stats.last_used_iso = when_iso

    # ----- persistence ------------------------------------------------------

    def load(self) -> int:
        """Load discovered methods + stats from the persistence file.

        Returns the number of entries loaded. Built-in methods are not loaded
        from disk (they are imported from code).
        """
        if self._persistence_path is None or not self._persistence_path.exists():
            return 0
        try:
            data = json.loads(self._persistence_path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        loaded = 0
        for entry_data in data.get("entries", []):
            method_id = str(entry_data.get("method_id") or "")
            if not method_id:
                continue
            with self._lock:
                if method_id in self._entries:
                    # Apply persisted stats to an existing in-memory entry.
                    self._entries[method_id].stats = MethodStats.from_dict(
                        entry_data.get("stats") or {}
                    )
                    loaded += 1
                # Note: re-instantiating a discovered Method requires the
                # discovery module to know its serialization format. We delegate
                # that to ``app.methods.discovery.rehydrate_persisted_method``
                # (called by ``get_default_library``).
        return loaded

    def persist(self) -> None:
        """Write discovered methods + stats to the persistence file."""
        if self._persistence_path is None:
            return
        with self._lock:
            payload = {
                "version": 1,
                "entries": [
                    {
                        "method_id": entry.method.method_id,
                        "family": entry.method.family.value,
                        "source": entry.method.source.value,
                        "signature": entry.method.signature(),
                        "stats": entry.stats.to_dict(),
                    }
                    for entry in self._entries.values()
                    if entry.method.source in {
                        MethodSource.DISCOVERED_VERIFIED,
                        MethodSource.DISCOVERED_PROVISIONAL,
                        MethodSource.USER_REGISTERED,
                    }
                ],
            }
        self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
        self._persistence_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Default library: lazy-built singleton populated with all built-in methods.
# ---------------------------------------------------------------------------

_DEFAULT_LIBRARY: MethodLibrary | None = None
_DEFAULT_LIBRARY_LOCK = threading.RLock()


def get_default_library() -> MethodLibrary:
    """Return (and lazily build) the process-wide default ``MethodLibrary``.

    The default library is populated with every built-in method on first call
    and is persisted under ``models/methods.json``. Tests that need a clean
    library should construct their own ``MethodLibrary()`` directly.
    """
    global _DEFAULT_LIBRARY
    with _DEFAULT_LIBRARY_LOCK:
        if _DEFAULT_LIBRARY is None:
            persistence = _DEFAULT_LIBRARY_PATH
            if os.environ.get("URA_METHODS_PERSISTENCE", "").strip() == "0":
                persistence = None
            lib = MethodLibrary(persistence_path=persistence)
            # Lazy import to avoid a startup cycle: impl modules import types.
            from app.methods.impl.builtin_loader import register_builtins
            register_builtins(lib)
            lib.load()
            _DEFAULT_LIBRARY = lib
        return _DEFAULT_LIBRARY


def reset_default_library() -> None:
    """Test-only helper: drop the cached default library."""
    global _DEFAULT_LIBRARY
    with _DEFAULT_LIBRARY_LOCK:
        _DEFAULT_LIBRARY = None
