"""Semantic atom matching via sentence embeddings.

Resolves paraphrase mismatches that cause the deterministic FOL compiler to
abstain (Cluster A). When exact token-set equality fails between a rule atom
and a fact atom, this module computes cosine similarity of their sentence
embeddings and unifies them if similarity >= threshold.

Deterministic: same inputs → same cosine similarity → same match decision.
The model is loaded lazily on first use to avoid import-time GPU allocation.

Graceful degradation: if the model fails to load (missing package, OOM, etc.),
semantic matching is disabled and the compiler falls back to exact matching only.
"""

from __future__ import annotations

import logging
import os
# Force CPU execution for sentence-transformers to avoid CUDA initialization deadlocks under WSL2/WDDM memory pressure.
# Must be set at module load time before any torch-dependent imports are executed.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_CUDA"] = "0"
os.environ["FORCE_CPU"] = "1"

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.logic.dsl_compiler import _Atom, _Theory

logger = logging.getLogger(__name__)

_ASYMMETRIC_MODIFIERS = {
    "actively", "only", "except", "unless", "never", "always", "strictly",
    "solely", "exclusively", "specifically", "particularly", "specially",
    "partially", "mostly", "mainly", "primarily", "predominantly",
    "virtually", "almost", "nearly", "barely", "scarcely", "hardly",
    "seldom", "rarely", "frequently", "often", "usually", "generally",
    "normally", "typically", "constantly", "continually", "regularly",
    "consistently", "periodically", "occasionally", "sometimes",
    "exceptionally", "extremely", "highly", "very", "too",
    "sufficiently", "adequately", "properly", "satisfactorily", "fully",
    "completely", "entirely", "totally", "absolutely", "perfectly",
    "slightly", "moderately", "significantly", "substantially"
}

# ---------------------------------------------------------------------------
# Singleton AtomMatcher
# ---------------------------------------------------------------------------


class AtomMatcher:
    """Semantic atom matching via sentence embeddings.

    Used by the FOL compiler to match rule antecedent atoms against fact atoms
    when exact token-set equality fails.
    """

    _instance: "AtomMatcher | None" = None

    def __init__(self, threshold: float = 0.80):
        """Initialize the AtomMatcher with a similarity threshold."""
        self._model = None
        self._model_load_failed = False
        self.threshold = threshold

    @classmethod
    def get_instance(cls, threshold: float = 0.80) -> "AtomMatcher":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(threshold=threshold)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def _ensure_model(self) -> bool:
        """Lazy-load the sentence-transformer model. Returns True if available."""
        if self._model is not None:
            return True
        if self._model_load_failed:
            return False
        # Allow disabling semantic matching via env var (e.g. when GPU memory
        # is needed for the LLM inference backend, or during CI without the model).
        if os.environ.get("URA_DISABLE_SEMANTIC_MATCHING", "").strip().lower() in ("1", "true", "yes"):
            self._model_load_failed = True
            logger.info("AtomMatcher: semantic matching disabled via URA_DISABLE_SEMANTIC_MATCHING")
            return False
        try:
            # Force CPU execution for sentence-transformers to avoid CUDA initialization deadlocks under WSL2/WDDM memory pressure.
            # Must be set before importing sentence_transformers or torch.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["USE_CUDA"] = "0"
            os.environ["FORCE_CPU"] = "1"
            from sentence_transformers import SentenceTransformer
            device = os.environ.get("URA_EMBEDDING_DEVICE", "cpu")
            self._model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
            # Pre-warm the model to force lazy PyTorch/initialization
            self._model.encode(["warmup"], normalize_embeddings=True)
            logger.info("AtomMatcher: loaded and warmed up all-MiniLM-L6-v2 model on %s", device)
            return True
        except Exception as exc:
            self._model_load_failed = True
            logger.warning("AtomMatcher: model load failed (%s), falling back to exact matching", exc)
            return False

    def _atom_to_phrase(self, atom: "_Atom") -> str:
        """Convert an atom's key tuple back to a readable phrase for embedding."""
        return " ".join(atom.key)

    def phrases_match(self, phrase_a: str, phrase_b: str) -> bool:
        """True if the two phrases are semantically equivalent (cosine >= threshold).

        Args:
            phrase_a: The first string phrase.
            phrase_b: The second string phrase.

        Returns:
            True if semantically equivalent, False otherwise.
        """
        if phrase_a == phrase_b:
            return True
        if not self._ensure_model():
            return False
        try:
            embeddings = self._model.encode([phrase_a, phrase_b], normalize_embeddings=True)
            # Cosine similarity of normalized vectors = dot product
            similarity = float(embeddings[0] @ embeddings[1])
            return similarity >= self.threshold
        except Exception as exc:
            logger.warning("AtomMatcher: embedding comparison failed (%s)", exc)
            return False

    def similarity(self, phrase_a: str, phrase_b: str) -> float:
        """Return cosine similarity between two phrases. -1.0 on failure.

        Args:
            phrase_a: The first string phrase.
            phrase_b: The second string phrase.

        Returns:
            The cosine similarity score, or -1.0 on failure.
        """
        if phrase_a == phrase_b:
            return 1.0
        if not self._ensure_model():
            return -1.0
        try:
            embeddings = self._model.encode([phrase_a, phrase_b], normalize_embeddings=True)
            return float(embeddings[0] @ embeddings[1])
        except Exception:
            return -1.0

    def keys_match(self, key_a: tuple[str, ...], key_b: tuple[str, ...]) -> bool:
        """True if two atom keys are semantically equivalent.

        Uses a combined strategy:
        1. Exact equality (fast path).
        2. Subset check: if one key's tokens are a proper subset of the other's
           AND the embedding similarity is above a relaxed threshold (0.70),
           they match. This handles the common case where a fact adds filler
           tokens (e.g. "cours", "hour", "requir") to a rule's core predicate.
        3. Full embedding similarity >= threshold.

        Args:
            key_a: The token tuple of the first atom.
            key_b: The token tuple of the second atom.

        Returns:
            True if keys match semantically, False otherwise.
        """
        if key_a == key_b:
            return True
        if not self._ensure_model():
            return False

        set_a = set(key_a)
        set_b = set(key_b)
        is_subset = set_a < set_b or set_b < set_a

        phrase_a = " ".join(key_a)
        phrase_b = " ".join(key_b)
        sim = self.similarity(phrase_a, phrase_b)

        if sim >= self.threshold:
            return True
        # Relaxed threshold when one is a subset of the other (strong structural signal)
        if is_subset:
            diff = set_a ^ set_b
            has_asymmetric = any(w.lower() in _ASYMMETRIC_MODIFIERS for w in diff)
            if not has_asymmetric and sim >= (self.threshold - 0.10):
                return True
        return False

    def atom_matches(self, rule_atom: "_Atom", fact_atom: "_Atom") -> bool:
        """True if a rule atom semantically matches a fact atom.

        Fast path: exact key equality (no model needed).
        Fallback: embedding similarity when exact match fails.
        Negation polarity must match exactly.

        Args:
            rule_atom: The compiled atom from a rule.
            fact_atom: The compiled atom from a ground fact.

        Returns:
            True if the atoms match, False otherwise.
        """
        # Negation polarity must match
        if rule_atom.negated != fact_atom.negated:
            return False
        # Use keys_match which handles exact, subset, and semantic matching
        return self.keys_match(rule_atom.key, fact_atom.key)


# ---------------------------------------------------------------------------
# Predicate Unification Pass
# ---------------------------------------------------------------------------
def unify_theory_atoms(theory: "_Theory", threshold: float = 0.80) -> tuple["_Theory", list[str]]:
    """Run semantic predicate unification on a compiled theory.

    Identifies semantically-equivalent property atoms across rules and facts,
    and maps them to the same canonical atom so Z3 sees unified predicates.

    Algorithm:
    1. Collect all unique non-negated property atoms from rules + facts + query.
    2. For each pair where exact key != : compute embedding similarity.
    3. If similarity >= threshold: unify under the canonical atom (shorter key,
       or the one from a rule if lengths are equal).
    4. Replace all occurrences of non-canonical atoms with the canonical one.

    Soundness safeguards:
    - Only unifies property atoms (multi-token keys), not class atoms (single
      generic-class token keys) — class membership must be exact.
    - Only unifies atoms with the same negation polarity.
    - Never unifies atoms where one is negated and the other is not.
    - Deterministic: same embeddings → same cosine → same decision.

    Args:
        theory: The compiled _Theory to unify.
        threshold: Cosine similarity threshold for mapping.

    Returns:
        A tuple of (unified_theory, unification_log_list).
    """
    from app.logic.dsl_compiler import _Atom, _Theory, _TheoryRule, _TheoryFact, _TheoryQuery

    matcher = AtomMatcher.get_instance(threshold=threshold)
    if not matcher._ensure_model():
        # Model unavailable — return theory unchanged (graceful degradation)
        return theory, ["semantic_model_unavailable:falling_back_to_exact_matching"]

    # Step 1: Collect all unique property atoms (non-class, multi-token keys)
    all_atoms: set[tuple[str, ...]] = set()

    def _collect_atoms_from_list(atoms: tuple["_Atom", ...]) -> None:
        for atom in atoms:
            # Only consider property atoms (multi-token keys)
            # Single-token keys that are class atoms should not be unified
            if len(atom.key) > 1:
                all_atoms.add(atom.key)

    for rule in theory.rules:
        _collect_atoms_from_list(rule.antecedents)
        _collect_atoms_from_list(rule.consequents)
    for fact in theory.facts:
        _collect_atoms_from_list(fact.atoms)
    if theory.query:
        _collect_atoms_from_list(theory.query.goal)
        _collect_atoms_from_list(theory.query.hypotheses)

    if len(all_atoms) < 2:
        return theory, []

    # Step 2: Find semantically equivalent pairs
    atom_list = sorted(all_atoms, key=lambda k: (len(k), k))  # deterministic order
    # Build a union-find mapping: non-canonical → canonical
    canonical_map: dict[tuple[str, ...], tuple[str, ...]] = {}

    for i in range(len(atom_list)):
        if atom_list[i] in canonical_map:
            continue
        for j in range(i + 1, len(atom_list)):
            if atom_list[j] in canonical_map:
                continue
            if matcher.keys_match(atom_list[i], atom_list[j]):
                # Unify: pick the shorter key as canonical (rule atoms tend to be
                # shorter/more concise). If same length, pick lexicographically first.
                canon = atom_list[i]
                non_canon = atom_list[j]
                if len(non_canon) < len(canon):
                    canon, non_canon = non_canon, canon
                elif len(non_canon) == len(canon) and non_canon < canon:
                    canon, non_canon = non_canon, canon
                canonical_map[non_canon] = canon

    if not canonical_map:
        return theory, []

    # Step 3: Build unification log
    unification_log: list[str] = []
    for non_canon, canon in sorted(canonical_map.items()):
        unification_log.append(
            f"unified: ({' '.join(non_canon)}) → ({' '.join(canon)})"
        )

    # Step 4: Replace atoms throughout the theory
    def _remap_atom(atom: "_Atom") -> "_Atom":
        if atom.key in canonical_map:
            return _Atom(canonical_map[atom.key], atom.negated)
        return atom

    def _remap_atoms(atoms: tuple["_Atom", ...]) -> tuple["_Atom", ...]:
        return tuple(_remap_atom(a) for a in atoms)

    new_rules = [
        _TheoryRule(
            premise_id=rule.premise_id,
            antecedents=_remap_atoms(rule.antecedents),
            consequents=_remap_atoms(rule.consequents),
        )
        for rule in theory.rules
    ]
    new_facts = [
        _TheoryFact(
            premise_id=fact.premise_id,
            entity=fact.entity,
            atoms=_remap_atoms(fact.atoms),
        )
        for fact in theory.facts
    ]
    new_query = None
    if theory.query:
        new_query = _TheoryQuery(
            kind=theory.query.kind,
            goal=_remap_atoms(theory.query.goal),
            entity=theory.query.entity,
            hypotheses=_remap_atoms(theory.query.hypotheses),
        )

    new_theory = _Theory(
        rules=new_rules,
        facts=new_facts,
        query=new_query,
        ground_entities=theory.ground_entities,
    )

    logger.info("AtomMatcher unified %d atom pairs: %s", len(canonical_map), unification_log)
    return new_theory, unification_log
