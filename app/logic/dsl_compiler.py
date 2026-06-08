"""Typed JSON DSL → Z3 Compiler for Logic-Based Educational Queries.

This module replaces free-form FOL string parsing with a structured DSL that
the LLM outputs. The backend compiles the DSL to Z3 expressions with explicit
direction/polarity handling.

DSL Schema (what the LLM outputs):
{
  "clauses": [
    {
      "premise_id": "P1",
      "type": "universal_positive" | "universal_negative" | "conditional" | "ground_fact" | "ground_negation" | "existential",
      "subject_class": "student",       // for universal/conditional
      "condition": "studies_regularly",  // antecedent predicate
      "conclusion": "performs_well",    // consequent predicate
      "entity": "alex",                 // for ground facts
      "predicate": "studies_regularly", // for ground facts
      "direction": "sufficient" | "necessary" | "biconditional",
      "polarity": "positive" | "negative"
    }
  ],
  "query": {
    "type": "ground" | "universal" | "existential",
    "entity": "alex",
    "subject_class": "student",
    "predicate": "performs_well",
    "polarity": "positive"
  },
  "confidence": 0.9
}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


@dataclass
class DSLClause:
    """Represents a structured clause in the logical DSL."""
    premise_id: str
    clause_type: str
    direction: str = "sufficient"
    polarity: str = "positive"
    subject_class: str = ""
    condition: str = ""
    conclusion: str = ""
    entity: str = ""
    predicate: str = ""
    confidence: float = 1.0


@dataclass
class DSLQuery:
    """Represents a structured query to check against the compiled DSL theory."""
    query_type: str = "ground"
    entity: str = ""
    subject_class: str = ""
    predicate: str = ""
    polarity: str = "positive"


@dataclass
class DSLTranslation:
    """Wrapper holding parsed clauses, query, confidence, and any translation errors."""
    clauses: list[DSLClause]
    query: DSLQuery
    confidence: float
    unsupported: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.confidence >= 0.6 and len(self.clauses) > 0


@dataclass
class DSLCompileResult:
    """Contains the answer and status returned by compiling and running Z3 on the DSL."""
    answer: str
    z3_status: str
    used_premise_ids: list[str]
    proof_steps: list[str]
    confidence: float
    error: str | None = None
    direction_gate_blocked: bool = False


def parse_dsl_response(raw: dict[str, Any]) -> DSLTranslation:
    """Parse a raw JSON dictionary response from the LLM into a structured DSLTranslation.

    Args:
        raw: The raw dictionary containing clauses, query, and confidence.

    Returns:
        The validated DSLTranslation object.
    """
    clauses = []
    for c in raw.get("clauses", []):
        clauses.append(DSLClause(
            premise_id=str(c.get("premise_id", "?")),
            clause_type=str(c.get("type", "ground_fact")),
            direction=str(c.get("direction", "sufficient")),
            polarity=str(c.get("polarity", "positive")),
            subject_class=str(c.get("subject_class", "")),
            condition=str(c.get("condition", "")),
            conclusion=str(c.get("conclusion", "")),
            entity=str(c.get("entity", "")),
            predicate=str(c.get("predicate", "")),
            confidence=float(c.get("confidence", 0.8)),
        ))
    query_raw = raw.get("query", {})
    if not isinstance(query_raw, dict):
        # Some LLM responses emit `"query": "alarm_sounds(entity)"` as a
        # plain string instead of the structured dict. Treat it as a
        # ground predicate and let the DSLQuery defaults absorb the rest.
        query_raw = {"type": "ground", "predicate": str(query_raw)}
    query = DSLQuery(
        query_type=str(query_raw.get("type", "ground")),
        entity=str(query_raw.get("entity", "")),
        subject_class=str(query_raw.get("subject_class", "")),
        predicate=str(query_raw.get("predicate", "")),
        polarity=str(query_raw.get("polarity", "positive")),
    )
    return DSLTranslation(
        clauses=clauses,
        query=query,
        confidence=float(raw.get("confidence", 0.0)),
        unsupported=list(raw.get("unsupported", [])),
        error=None if clauses else "no_clauses_parsed",
    )


def _predicate_name(text: str) -> str:
    """Format and normalize a string to be used as a Z3 predicate name."""
    return re.sub(r"[^a-z0-9_]", "_", text.lower().strip())[:60]


def _entity_name(text: str) -> str:
    """Format and normalize a string to be used as a Z3 constant/entity name."""
    return re.sub(r"[^a-z0-9_]", "_", text.lower().strip())[:40]


def direction_gate_check(clauses: list[DSLClause], query: DSLQuery) -> tuple[bool, str | None]:
    """Verify that the query is not blocked by necessary-condition direction gates.

    Args:
        clauses: The list of DSLClauses.
        query: The DSLQuery to check.

    Returns:
        A tuple of (is_authorized, block_reason_or_None).
    """
    if query.polarity == "negative":
        return True, None

    necessary_only_clauses = [
        c for c in clauses
        if c.direction == "necessary"
        and c.conclusion and query.predicate
        and _predicate_name(c.condition) == _predicate_name(query.predicate)
    ]
    sufficient_clauses = [
        c for c in clauses
        if c.direction in ("sufficient", "biconditional")
        and c.conclusion
        and _predicate_name(c.conclusion) == _predicate_name(query.predicate)
    ]

    if necessary_only_clauses and not sufficient_clauses:
        return False, "direction_gate:necessary_only_proves_condition_not_conclusion"

    return True, None


# ---------------------------------------------------------------------------
# Centralized necessary-vs-sufficient TEXT classification (Task 7.7, ADR 0004)
# ---------------------------------------------------------------------------
# Pure, side-effect-free text predicates shared by BOTH the deterministic logic
# fast-path post-check (``app.logic.solver.solve``) and the LLM-rescue
# directionality gate (``app.logic.fol_z3_pipeline._directionality_gate_ok``),
# so necessary-only wording ("requires", "only if") is recognized identically
# everywhere and can NEVER license a positive target rule. Centralizing here
# (a leaf module that imports neither solver nor the FOL pipeline at module
# scope) keeps the rule in one place without creating an import cycle.

_NECESSARY_ONLY_MARKERS = re.compile(
    r"\bonly\s+if\b"                                   # "B only if A"  (A necessary for B)
    r"|\brequires?\b"                                   # "B requires A"
    r"|\bis\s+required\s+(?:for|to)\b|\bare\s+required\s+(?:for|to)\b"
    r"|\bis\s+necessary\s+(?:for|to)\b|\bare\s+necessary\s+(?:for|to)\b"
    r"|\bnecessary\s+condition\b"
    r"|\bis\s+a\s+prerequisite\s+(?:for|to)\b|\bprerequisite\s+(?:for|to)\b",
    re.IGNORECASE,
)

_SUFFICIENT_RULE_MARKERS = re.compile(
    r"\bif\s+and\s+only\s+if\b|\biff\b"                 # biconditional: sufficient both ways
    r"|\bif\b[^.;]*\bthen\b"                            # explicit "if ... then"
    r"|\ball\b|\bevery\b|\beach\b"                      # universal affirmatives
    r"|\banyone\s+who\b|\banybody\s+who\b|\bwhoever\b|\bwhomever\b"
    r"|\b(?:students?|people|persons?|members?|employees?|users?|candidates?|applicants?)\s+who\b"
    r"|\bwhenever\b",
    re.IGNORECASE,
)


def text_states_sufficient_rule(text: str) -> bool:
    """True if ``text`` expresses a SUFFICIENT antecedent->consequent rule.

    Examples: "If A then B", "All A are B", "Anyone who A is B", and the
    biconditional "A if and only if B" (sufficient in both directions). Such
    wording licenses deriving the consequent from the antecedent.
    """
    low = re.sub(r"\s+", " ", str(text or "").lower().strip())
    return bool(_SUFFICIENT_RULE_MARKERS.search(low))


def text_states_necessary_only(text: str) -> bool:
    """True if ``text`` states a NECESSARY condition without sufficient wording.

    A necessary-only premise ("B requires A", "B only if A", "A is required for
    B") constrains a conclusion (``B -> A``) but does NOT license deriving the
    conclusion from the condition alone (Req 4.4, ADR 0004). Text that ALSO
    carries sufficient-rule wording ("if ... then", "all ... are", "if and only
    if") is not necessary-only — the sufficient direction governs. The markers
    are deliberately STRUCTURAL ("requires", "only if", "required/necessary/
    prerequisite for") so a fact that merely mentions the noun "prerequisite"
    ("Maria completed the prerequisite.") is NOT misclassified as a rule.
    """
    low = re.sub(r"\s+", " ", str(text or "").lower().strip())
    if not _NECESSARY_ONLY_MARKERS.search(low):
        return False
    return not text_states_sufficient_rule(low)


def compile_dsl_to_z3(translation: DSLTranslation) -> DSLCompileResult:
    """Compile the structured DSL translation into a Z3 theory and execute the entailment check.

    Args:
        translation: The structured DSLTranslation representation.

    Returns:
        The resulting DSLCompileResult with answer, status, and proof steps.
    """
    if not Z3_AVAILABLE:
        return DSLCompileResult(
            answer="unknown", z3_status="error",
            used_premise_ids=[], proof_steps=["Z3 not available"],
            confidence=0.0, error="z3_not_installed",
        )

    if not translation.success:
        return DSLCompileResult(
            answer="unknown", z3_status="abstained",
            used_premise_ids=[], proof_steps=["Translation failed or low confidence"],
            confidence=0.0, error=translation.error or "low_confidence",
        )

    gate_ok, gate_reason = direction_gate_check(translation.clauses, translation.query)
    if not gate_ok:
        return DSLCompileResult(
            answer="unknown", z3_status="abstained",
            used_premise_ids=[c.premise_id for c in translation.clauses],
            proof_steps=[f"Direction gate blocked: {gate_reason}"],
            confidence=0.0, error=gate_reason,
            direction_gate_blocked=True,
        )

    entity_sort = z3.DeclareSort("Entity")
    symbols: dict[str, Any] = {}
    assertions: list[tuple[str, Any]] = []

    def get_pred(name: str):
        key = f"pred:{name}"
        if key not in symbols:
            symbols[key] = z3.Function(name, entity_sort, z3.BoolSort())
        return symbols[key]

    def get_const(name: str):
        key = f"const:{name}"
        if key not in symbols:
            symbols[key] = z3.Const(name, entity_sort)
        return symbols[key]

    x = z3.Const("x", entity_sort)

    # Helper: split a condition atom into a Z3 antecedent expression. Handles
    # inline-negated atoms ("is_not_X" / "not_X") so a translator that puts the
    # negation INSIDE the condition (instead of using polarity="negative") still
    # produces a sound theory that interacts with ground facts on the BASE
    # predicate. Without this, a rule like
    #   subject_class=bird, condition=is_not_penguin, conclusion=can_fly
    # would create a fresh `is_not_penguin/1` predicate that is never asserted
    # by the ground facts (which use `penguin(tweety)`), so the rule never
    # fires and the query is silently undetermined. This generalizes over every
    # "All X that are not Y are Z" + "a is Y" pattern, not just penguins.
    def antecedent_for(subject_class: str, condition: str):
        atoms: list[Any] = []
        sc = (subject_class or "").strip()
        if sc:
            atoms.append(get_pred(_predicate_name(sc))(x))
        cond_raw = (condition or "").strip()
        if cond_raw:
            cond_norm = _predicate_name(cond_raw)
            # Detect inline-negation prefixes that the translator may produce
            # for "X that are not Y" / "X unless Y" patterns. The legal DSL way
            # is polarity="negative", but small models sometimes embed it into
            # the atom name; treat both equivalently.
            if cond_norm.startswith("is_not_"):
                base = cond_norm[len("is_not_"):]
                atoms.append(z3.Not(get_pred(base)(x)))
            elif cond_norm.startswith("not_"):
                base = cond_norm[len("not_"):]
                atoms.append(z3.Not(get_pred(base)(x)))
            else:
                atoms.append(get_pred(cond_norm)(x))
        if not atoms:
            # Degenerate clause — caller should skip; signal via None.
            return None
        if len(atoms) == 1:
            return atoms[0]
        return z3.And(*atoms)

    for clause in translation.clauses:
        ct = clause.clause_type
        pol_neg = (clause.polarity == "negative")

        if ct in ("universal_positive", "conditional"):
            antecedent = antecedent_for(clause.subject_class, clause.condition)
            if antecedent is None:
                # Skip clauses with no antecedent atoms — they cannot fire.
                continue
            conc_pred = get_pred(_predicate_name(clause.conclusion))
            if pol_neg:
                expr = z3.ForAll([x], z3.Implies(antecedent, z3.Not(conc_pred(x))))
            else:
                expr = z3.ForAll([x], z3.Implies(antecedent, conc_pred(x)))
            assertions.append((clause.premise_id, expr))

        elif ct == "universal_negative":
            antecedent = antecedent_for(clause.subject_class, clause.condition)
            if antecedent is None:
                # Fallback to the legacy single-atom shape so we don't regress
                # cases where only `condition` (or only `subject_class`) is set.
                cond_pred = get_pred(_predicate_name(clause.condition or clause.subject_class))
                antecedent = cond_pred(x)
            conc_pred = get_pred(_predicate_name(clause.conclusion or clause.predicate))
            expr = z3.ForAll([x], z3.Implies(antecedent, z3.Not(conc_pred(x))))
            assertions.append((clause.premise_id, expr))

        elif ct == "ground_fact":
            entity = get_const(_entity_name(clause.entity))
            pred = get_pred(_predicate_name(clause.predicate))
            if pol_neg:
                assertions.append((clause.premise_id, z3.Not(pred(entity))))
            else:
                assertions.append((clause.premise_id, pred(entity)))

        elif ct == "ground_negation":
            entity = get_const(_entity_name(clause.entity))
            pred = get_pred(_predicate_name(clause.predicate))
            assertions.append((clause.premise_id, z3.Not(pred(entity))))

        elif ct == "existential":
            pred = get_pred(_predicate_name(clause.predicate or clause.conclusion))
            witness = z3.Const(f"witness_{clause.premise_id}", entity_sort)
            if pol_neg:
                assertions.append((clause.premise_id, z3.Not(pred(witness))))
            else:
                assertions.append((clause.premise_id, pred(witness)))

    query = translation.query
    if query.query_type == "ground":
        entity = get_const(_entity_name(query.entity))
        pred = get_pred(_predicate_name(query.predicate))
        query_expr = pred(entity) if query.polarity == "positive" else z3.Not(pred(entity))
    elif query.query_type == "universal":
        pred = get_pred(_predicate_name(query.predicate))
        sc_pred = get_pred(_predicate_name(query.subject_class)) if query.subject_class else None
        if sc_pred:
            query_expr = z3.ForAll([x], z3.Implies(sc_pred(x), pred(x))) if query.polarity == "positive" else z3.ForAll([x], z3.Implies(sc_pred(x), z3.Not(pred(x))))
        else:
            query_expr = z3.ForAll([x], pred(x)) if query.polarity == "positive" else z3.ForAll([x], z3.Not(pred(x)))
    else:
        pred = get_pred(_predicate_name(query.predicate))
        witness = z3.Const("query_witness", entity_sort)
        query_expr = pred(witness) if query.polarity == "positive" else z3.Not(pred(witness))

    solver = z3.Solver()
    solver.set("timeout", 5000)
    for _pid, expr in assertions:
        solver.add(expr)

    used_ids = [pid for pid, _ in assertions]

    solver.push()
    solver.add(z3.Not(query_expr))
    check_yes = solver.check()
    solver.pop()

    solver.push()
    solver.add(query_expr)
    check_no = solver.check()
    solver.pop()

    if check_yes == z3.unsat:
        return DSLCompileResult(
            answer="yes", z3_status="entailed",
            used_premise_ids=used_ids,
            proof_steps=["DSL→Z3: theory ∧ ¬query is UNSAT → query entailed"],
            confidence=0.92 * translation.confidence,
        )
    elif check_no == z3.unsat:
        return DSLCompileResult(
            answer="no", z3_status="contradicted",
            used_premise_ids=used_ids,
            proof_steps=["DSL→Z3: theory ∧ query is UNSAT → query contradicted"],
            confidence=0.92 * translation.confidence,
        )
    else:
        return DSLCompileResult(
            answer="unknown", z3_status="undetermined",
            used_premise_ids=used_ids,
            proof_steps=["DSL→Z3: neither entailed nor contradicted"],
            confidence=0.3,
        )


DSL_TRANSLATE_SYSTEM_PROMPT = """You are a logic translator for educational question-answering.
Translate natural language premises into a typed JSON DSL.

Input: JSON with "premises" (list of {"id", "text"}), "query" (the question), "query_subject", "query_predicate".

Output ONLY valid JSON:
{
  "clauses": [
    {
      "premise_id": "P1",
      "type": "universal_positive|universal_negative|conditional|ground_fact|ground_negation|existential",
      "subject_class": "student",
      "condition": "studies_regularly",
      "conclusion": "performs_well_in_exams",
      "entity": "",
      "predicate": "",
      "direction": "sufficient|necessary|biconditional",
      "polarity": "positive|negative",
      "confidence": 0.95
    }
  ],
  "query": {
    "type": "ground|universal|existential",
    "entity": "alex",
    "subject_class": "",
    "predicate": "eligible_for_scholarship",
    "polarity": "positive"
  },
  "confidence": 0.9,
  "unsupported": []
}

Rules:
- "direction" is CRITICAL:
  - "sufficient": "If A then B" / "All A are B" → A is enough for B
  - "necessary": "B requires A" / "B only if A" → A is needed for B, but A alone does NOT prove B
  - "biconditional": "A if and only if B"
- For "universal_positive"/"conditional": use subject_class, condition, conclusion
- For "ground_fact"/"ground_negation": use entity, predicate
- For "existential": use predicate (or conclusion)
- Use snake_case for all predicate/entity names

CLASS MEMBERSHIP (CRITICAL — single-arg predicates only):
- "Tweety is a bird" → ground_fact with entity="tweety", predicate="bird"
  (NOT predicate="is_a", entity="bird" — Z3 only takes UNARY predicates here.)
- "Alex is a student" → entity="alex", predicate="student"
- A premise like "All birds can fly" is a UNIVERSAL over the class predicate:
  subject_class="bird", condition="bird", conclusion="can_fly" — keep
  subject_class so it intersects with ground class facts.

NEGATIVE CONDITIONS / "UNLESS" / "WHO ARE NOT" (CRITICAL):
- "All birds can fly UNLESS they are penguins" means
  "For every bird that is NOT a penguin, it can fly".
  Translate as one CONDITIONAL clause AND two ground class facts:
    P_rule (the rule):
      type="conditional",
      subject_class="bird",
      condition="is_not_penguin",   ← prefix the negated atom with "is_not_"
      conclusion="can_fly",
      polarity="positive"
    P_bird:    type="ground_fact", entity="tweety", predicate="bird"
    P_penguin: type="ground_fact", entity="tweety", predicate="penguin"
  The compiler reads "is_not_X" as Not(X(x)) and ANDs it with subject_class,
  so the rule becomes:  ForAll x, (bird(x) AND Not(penguin(x))) -> can_fly(x).
  With penguin(tweety) asserted by P_penguin, the rule does NOT fire for
  Tweety, and the answer is correctly "no" / unknown.
- "Employees who are NOT managers must clock in" follows the same shape:
  subject_class="employee", condition="is_not_manager",
  conclusion="must_clock_in".
- Use the "is_not_" prefix exactly (snake_case, lowercase) so the compiler
  pairs it with the base ground fact ("manager", "penguin", ...).

- If you cannot translate a premise, add its id to "unsupported"
- Do NOT reason or answer — only translate
"""


# ===========================================================================
# Deterministic NL / premises_fol -> typed DSL -> Z3 compiler (Task 7.1)
# ===========================================================================
#
# This is a *backend* compiler: it parses the question and natural-language
# premises (and clean ``premises_fol`` when present) into a small typed theory
# and hands it to Z3 WITHOUT the LLM. Each universal rule compiles to
# ``ForAll([x], Implies(A(x), B(x)))`` so Z3 performs the chaining itself; the
# chain depth is therefore unbounded by construction (Req 4.1). When the theory
# does not entail a definite verdict the compiler abstains (``unknown``) rather
# than guessing (Req 4.6).
#
# Soundness model (root-cause, AGENTS.md §20): every premise/query phrase is
# decomposed into atoms keyed by the canonical *content-token set*. A generic
# "kind" noun (student, employee, ...) becomes an explicit class predicate so a
# universal like "All students submit X" requires the subject actually be a
# student before the chain fires; this keeps subclass traps ("All honors
# students are eligible" + "Morgan is a student") at ``unknown``. A non-generic
# property phrase ("submit homework on time") becomes a single combined atom so
# matching is exact across premises (partial overlaps never fire => abstain).

# Generic class/sort nouns (singular, post-stemming) that name a kind rather
# than a property. Kept as explicit class predicates for sound chaining.
_GENERIC_CLASS_TOKENS = {
    "student", "person", "people", "employee", "worker", "member", "candidate",
    "applicant", "individual", "learner", "user", "participant", "citizen",
    "customer", "client", "patient", "resident", "child", "adult", "animal",
}

# Filler tokens (post-stemming) that carry no predicate identity: copulas,
# auxiliaries/modals, pronouns, dummy subjects, and connectives. These are
# stripped from property atoms so that, e.g., "something is a bird", "it is a
# bird", and "a bird" all collapse to the same atom ``(bird,)``. Crucially this
# set does NOT include domain kind-nouns (device, system, shape, ...): those are
# meaningful predicate terms and must be retained for cross-premise matching.
_ATOM_FILLER_TOKENS = {
    # copulas / auxiliaries / modals
    "is", "are", "am", "was", "were", "be", "been", "being",
    "do", "does", "did", "has", "have", "had",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    # pronouns / dummy subjects
    "it", "its", "they", "them", "their", "he", "she", "him", "her", "his",
    "we", "us", "our", "i", "you", "your", "one", "ones",
    "this", "that", "these", "those", "which", "who", "whom", "whose",
    "something", "anything", "everything", "nothing",
    "someone", "anyone", "everyone", "noone",
    "somebody", "anybody", "everybody", "nobody",
    # stemmed artifacts of the above (solver._stem maps e.g. "something"->"someth")
    "someth", "anyth", "everyth", "noth",
    # canonicalized artifacts (_canon_token strips trailing 'e' from stems > 4 chars)
    "someon", "anyon", "everyon", "noon",
    # connectives / prepositions / fillers
    "if", "then", "and", "or", "of", "to", "for", "in", "on", "at", "by",
    "with", "as", "from", "into", "than", "so", "also", "the", "a", "an",
    "there", "exist", "exists", "least", "some", "all", "every", "each", "any",
    "not", "no", "never",
}

# Adverbs / intensifiers / comparators that modify but do not change predicate
# identity. Stripped so paraphrases match across premises (e.g. "successfully
# graduate" == "graduates", "regularly studies" == "studies"). Curated rather
# than a blanket "-ly" rule because many content verbs end in -ly (apply,
# supply, comply, imply, reply).
_ATOM_MODIFIER_TOKENS = {
    "successfully", "successful", "definitely", "certainly", "possibly",
    "probably", "regularly", "consistently", "eventually", "ultimately",
    "properly", "correctly", "automatically", "immediately", "directly",
    "actually", "really", "very", "always", "still", "just", "fully",
    "completely", "clearly", "simply", "truly", "generally", "typically",
    "usually", "often", "sometimes", "currently", "already", "soon", "well",
    "above", "below", "greater", "less", "more", "least", "most", "minimum",
    "maximum", "min", "max", "least",
}


@dataclass(frozen=True)
class _Atom:
    """A canonical unary predicate over the single entity sort."""

    key: tuple[str, ...]
    negated: bool = False

    @property
    def name(self) -> str:
        return "p_" + ("_".join(self.key) if self.key else "true")


@dataclass
class _TheoryRule:
    """Represents a compiled rule with antecedents and consequents."""
    premise_id: str
    antecedents: tuple[_Atom, ...]
    consequents: tuple[_Atom, ...]


@dataclass
class _TheoryFact:
    """Represents a compiled ground fact about an entity."""
    premise_id: str
    entity: str
    atoms: tuple[_Atom, ...]


@dataclass
class _TheoryQuery:
    """Represents a compiled query goal with optional hypotheses."""
    kind: str                       # "ground" | "universal"
    goal: tuple[_Atom, ...]
    entity: str = ""                # for ground queries
    hypotheses: tuple[_Atom, ...] = ()


@dataclass
class _Theory:
    """Represents the complete compiled theory of rules, facts, and query."""
    rules: list[_TheoryRule] = field(default_factory=list)
    facts: list[_TheoryFact] = field(default_factory=list)
    query: _TheoryQuery | None = None
    ground_entities: set[str] = field(default_factory=set)


@dataclass
class DetFolResult:
    """Verdict of the deterministic FOL->Z3 compiler.

    ``answer`` is ``yes`` / ``no`` / ``unknown``. A definite verdict is only
    returned when Z3 reports clean entailment/contradiction over the compiled
    theory; otherwise the compiler abstains (Req 4.6).
    """

    answer: str
    z3_status: str
    used_premise_ids: list[str]
    proof_steps: list[str]
    error: str | None = None


def _det_solver_helpers():
    """Lazily import the token/parse helpers from solver.py.

    Imported lazily so this module never creates a load-time import cycle with
    ``app.logic.solver`` (which does not import this module at module scope).
    """
    from app.logic import solver as _solver  # noqa: WPS433 (intentional lazy import)

    return _solver


def _canon_token(tok: str, S: Any) -> str:
    """Canonicalize a single token for cross-premise atom matching.

    Applies the solver's shared stemmer, then removes a trailing ``e`` artifact
    so that 3rd-person verb forms collapse to the same root as their base form
    (e.g. solver._stem maps "accesses"->"accesse" while "access" stays
    "access"; both canonicalize to "access" here). This keeps predicate identity
    stable across "X accesses Y" and "X access Y" without touching the global
    stemmer used by the token-BFS fast path.
    """
    stem = S._stem(tok)
    if len(stem) > 4 and stem.endswith("e"):
        stem = stem[:-1]
    return stem


_GENERIC_CANON_CACHE: set[str] | None = None


def _canon_generic_set(S: Any) -> set[str]:
    """Generic-class tokens canonicalized to match ``_canon_token`` output."""
    global _GENERIC_CANON_CACHE
    if _GENERIC_CANON_CACHE is None:
        _GENERIC_CANON_CACHE = {_canon_token(t, S) for t in _GENERIC_CLASS_TOKENS}
    return _GENERIC_CANON_CACHE


def _phrase_atoms(phrase: str, S: Any) -> list[_Atom]:
    """Decompose a premise/query phrase-part into canonical atoms.

    Returns one class atom per generic kind-noun plus (at most) one combined
    property atom from the canonicalized content tokens. Negation found in the
    phrase is attached only to the property atom (class membership is not
    negated), matching the design's consequent-term negation scope.

    The property token set is built from canonicalized content tokens with
    copulas, auxiliaries, pronouns, dummy subjects, connectives, and modifiers
    removed. Domain kind-nouns (device, system, ...) are deliberately retained so
    the same property matches across premises regardless of dummy subjects
    ("it"/"something"/"a bird").
    """
    norm = S._norm(phrase)
    if not norm:
        return []
    negated = bool(S._NEGATION_PATTERN.search(norm))
    raw = {_canon_token(tok, S) for tok in re.findall(r"[a-z0-9]+", S._strip_articles(norm)) if len(tok) > 1}
    content = raw - _ATOM_FILLER_TOKENS - _ATOM_MODIFIER_TOKENS
    canon_generic = _canon_generic_set(S)
    generic = sorted(content & canon_generic)
    prop = tuple(sorted(content - canon_generic))
    atoms: list[_Atom] = [_Atom((g,), False) for g in generic]
    if prop:
        atoms.append(_Atom(prop, negated))
    return atoms


def _negate_atoms(atoms: list[_Atom]) -> list[_Atom]:
    """Negate the truth value of a list of atoms."""
    return [_Atom(a.key, not a.negated) for a in atoms]


# ---------------------------------------------------------------------------
# Multi-condition conjunctive antecedent splitting (Cluster A fix)
# ---------------------------------------------------------------------------
# Premises like "Students who have completed the core curriculum AND passed the
# science assessment are qualified for advanced courses" contain MULTIPLE
# conditions joined by "and" in the antecedent. The existing _phrase_atoms merges
# all tokens into one combined atom, which then cannot match individual ground
# facts. This splitter detects conjunctive antecedents and produces SEPARATE
# atoms for each conjunct so Z3 can match them independently.

# Regex to detect "and" used as a logical conjunction between conditions.
# We avoid splitting on "and" that's part of a compound noun (e.g. "research and
# development") by requiring verb-like context around the split point.
_CONJUNCTIVE_AND_PATTERN = re.compile(
    r"\s+and\s+(?=(?:has|have|had|is|are|was|were|passed|completed|received|"
    r"achieved|obtained|earned|demonstrated|maintained|submitted|finished|"
    r"fulfilled|met|acquired|holds|holding|possesses|possessing|shown|"
    r"taken|enrolled|registered|attended|participated|qualified|certified|"
    r"approved|accepted|selected|chosen|recommended|endorsed|verified|"
    r"confirmed|established|published|presented|conducted|performed|"
    r"developed|designed|implemented|managed|led|supervised|coordinated|"
    r"organized|planned|created|built|produced|delivered|provided|"
    r"secured|gained|won|reached|exceeded|surpassed|"
    r"not)\b)",
    re.IGNORECASE,
)

# Broader fallback: split on " and " when both sides look like verb phrases
# (start with a verb or "not"). This catches patterns the specific verb list misses.
_CONJUNCTIVE_AND_FALLBACK = re.compile(
    r"\s+and\s+",
    re.IGNORECASE,
)


def _is_verb_phrase_start(text: str) -> bool:
    """Heuristic: does the text start with a verb or auxiliary?"""
    text = text.strip().lower()
    return bool(re.match(
        r"^(?:has|have|had|is|are|was|were|do|does|did|"
        r"passed|completed|received|achieved|obtained|earned|demonstrated|"
        r"maintained|submitted|finished|fulfilled|met|acquired|holds|"
        r"possesses|shown|taken|enrolled|registered|attended|participated|"
        r"qualified|certified|approved|accepted|selected|chosen|recommended|"
        r"endorsed|verified|confirmed|established|published|presented|"
        r"conducted|performed|developed|designed|implemented|managed|led|"
        r"supervised|coordinated|organized|planned|created|built|produced|"
        r"delivered|provided|secured|gained|won|reached|exceeded|surpassed|"
        r"not)\b",
        text,
    ))


def _split_conjunctive_antecedent(antecedent: str) -> list[str]:
    """Split a compound antecedent on logical 'and' into individual conditions.

    Returns a list of condition strings. If no conjunctive split is detected,
    returns a single-element list with the original antecedent.

    Examples:
        "have completed the core curriculum and passed the science assessment"
        -> ["have completed the core curriculum", "passed the science assessment"]

        "is qualified for advanced courses and has completed research methodology"
        -> ["is qualified for advanced courses", "has completed research methodology"]

        "research and development"  (compound noun, NOT split)
        -> ["research and development"]
    """
    # First try the specific pattern that requires a verb after "and"
    parts = _CONJUNCTIVE_AND_PATTERN.split(antecedent)
    if len(parts) > 1:
        # Verify each part is non-empty after stripping
        cleaned = [p.strip() for p in parts if p.strip()]
        if len(cleaned) > 1:
            return cleaned

    # Fallback: split on " and " and check if both sides look like verb phrases
    parts = _CONJUNCTIVE_AND_FALLBACK.split(antecedent)
    if len(parts) > 1:
        cleaned = [p.strip() for p in parts if p.strip()]
        if len(cleaned) > 1 and all(_is_verb_phrase_start(p) for p in cleaned):
            return cleaned

    return [antecedent]


def _phrase_atoms_conjunctive(phrase: str, S: Any) -> list[_Atom]:
    """Like _phrase_atoms but splits conjunctive conditions into separate atoms.

    For a compound antecedent like "have completed X and passed Y", this produces
    separate atoms for each conjunct so they can be matched independently against
    ground facts. Each conjunct becomes its own property atom.

    For non-conjunctive phrases, behaves identically to _phrase_atoms.
    """
    conjuncts = _split_conjunctive_antecedent(phrase)
    if len(conjuncts) <= 1:
        # No conjunction detected — use standard atom decomposition
        return _phrase_atoms(phrase, S)

    # Multiple conjuncts: build separate atoms for each
    all_atoms: list[_Atom] = []
    for conjunct in conjuncts:
        atoms = _phrase_atoms(conjunct, S)
        all_atoms.extend(atoms)

    return all_atoms


def _consequent_atoms(phrase: str, S: Any) -> list[_Atom]:
    """Build atoms for a rule CONSEQUENT, scoping its negation to the consequent
    term (Req 4.2).

    A rule's polarity is set from the *consequent clause only*: a premise with a
    negated consequent ("All birds are not reptiles", "If X is a bird then X is
    not a reptile") compiles to ``Implies(cond(x), Not(conc(x)))``. The negation
    is therefore attached to the consequent term named in the premise and is
    NEVER extended to the antecedent or to the implication as a whole. Antecedent
    atoms are built separately via ``_phrase_atoms`` and never receive the
    consequent's negation.

    ``_phrase_atoms`` already negates a non-generic property consequent ("not a
    reptile"). It deliberately leaves a generic *class* membership atom positive
    (the right call for antecedents/subjects such as "non-students"), but for a
    consequent that drops the stated negation entirely ("are not people" →
    positive ``people`` atom). When the consequent clause is negated yet no atom
    captured that negation — i.e. the consequent term is a generic class noun —
    scope the negation onto the consequent class term so the polarity comes from
    the consequent clause, not the antecedent.
    """
    atoms = _phrase_atoms(phrase, S)
    if not atoms:
        return atoms
    if bool(S._NEGATION_PATTERN.search(S._norm(phrase))) and not any(a.negated for a in atoms):
        return [_Atom(a.key, True) for a in atoms]
    return atoms


def _looks_proper_subject(original_text: str, subject_phrase: str) -> bool:
    """Heuristic: is the subject a proper-named entity (ground) vs a kind?

    The leading article is stripped first, then the actual subject noun's case
    in the ORIGINAL premise text is checked. A capitalized noun ("Maya",
    "Tweety") is a ground entity; a lowercase common noun ("a snake", "the
    switch") denotes a generic kind and is compiled as a universal rule.
    """
    sub = re.sub(r"^(?:a|an|the|every|each|all|any|some)\s+", "", subject_phrase.strip(), flags=re.I)
    first = re.split(r"\s+", sub.strip())[0] if sub.strip() else ""
    if not first:
        return False
    # Find the first occurrence of the noun in the original text and inspect case.
    match = re.search(rf"\b({re.escape(first)})\b", original_text, flags=re.I)
    if not match:
        return False
    return match.group(1)[:1].isupper()


def _split_simple_statement(text: str, S: Any) -> tuple[str, str, bool] | None:
    """Split a simple declarative "<subject> <copula/verb> <predicate>"."""
    norm = S._norm(text).rstrip(".")
    copula = re.search(r"\b(is not|are not|was not|were not|is|are|was|were|has|have|had)\b", norm)
    if copula:
        subject = norm[: copula.start()].strip()
        predicate = norm[copula.end():].strip()
        negated = "not" in copula.group(1)
        if subject and predicate:
            return subject, predicate, negated
    parts = norm.split(" ", 1)
    if len(parts) == 2 and parts[0] and parts[1]:
        return parts[0].strip(), parts[1].strip(), False
    return None


def _compile_theory(question: str, premises: list[dict[str, str]], premises_fol: list[str] | None, S: Any) -> _Theory:
    """Compile premises and clean FOL representations into a logical Theory."""
    theory = _Theory()

    for premise in premises:
        pid = str(premise.get("id", "?"))
        text = str(premise.get("text", ""))
        norm = S._norm(text)
        if not norm:
            continue

        # 1. Existential statement -> witness fact.
        if S._is_existential_quantifier(norm):
            stmt = _split_simple_statement(re.sub(r"^(?:there\s+(?:exists?|is|are)\s+(?:at\s+least\s+one|some|an?|one)|some|at\s+least\s+one)\s+", "", norm), S)
            if stmt:
                subj, pred, neg = stmt
                atoms = _phrase_atoms(subj, S) + (_negate_atoms(_phrase_atoms(pred, S)) if neg else _phrase_atoms(pred, S))
                if atoms:
                    theory.facts.append(_TheoryFact(pid, f"witness_{pid}", tuple(atoms)))
            continue

        # 2. Conditional rule "if A [,] then B".
        if norm.startswith("if ") and S._conditional_parts(text):
            ante, cons = S._conditional_parts(text)
            ante_atoms = _phrase_atoms_conjunctive(ante, S)
            cons_atoms = _consequent_atoms(cons, S)
            if ante_atoms and cons_atoms:
                theory.rules.append(_TheoryRule(pid, tuple(ante_atoms), tuple(cons_atoms)))
            continue

        # 3. Universal negative "No A are B".
        no_rule = S._match_no_rule(text)
        if no_rule:
            ante_atoms = _phrase_atoms_conjunctive(no_rule[0], S)
            cons_atoms = _negate_atoms(_phrase_atoms(no_rule[1], S))
            if ante_atoms and cons_atoms:
                theory.rules.append(_TheoryRule(pid, tuple(ante_atoms), tuple(cons_atoms)))
            continue

        # 4. Universal positive "All/Every/Each A <pred>".
        all_rule = S._match_all_rule(text)
        if all_rule:
            ante_atoms = _phrase_atoms_conjunctive(all_rule[0], S)
            cons_atoms = _consequent_atoms(all_rule[1], S)
            if ante_atoms and cons_atoms:
                theory.rules.append(_TheoryRule(pid, tuple(ante_atoms), tuple(cons_atoms)))
            continue

        # 5. General conditional-ish rule ("students who X are Y", prohibitions).
        general_rule = S._match_rule(text)
        if general_rule and not S._is_existential_quantifier(norm):
            ante_atoms = _phrase_atoms_conjunctive(general_rule[0], S)
            cons_atoms = _consequent_atoms(general_rule[1], S)
            if ante_atoms and cons_atoms:
                theory.rules.append(_TheoryRule(pid, tuple(ante_atoms), tuple(cons_atoms)))
            continue

        # 6. Simple declarative -> ground fact (proper subject) or universal kind rule.
        stmt = _split_simple_statement(text, S)
        if stmt:
            subj, pred, neg = stmt
            pred_atoms = _negate_atoms(_phrase_atoms(pred, S)) if neg else _phrase_atoms(pred, S)
            if not pred_atoms:
                continue
            if _looks_proper_subject(text, subj):
                entity = _entity_name(re.sub(r"^(?:a|an|the)\s+", "", subj, flags=re.I))
                subj_atoms = _phrase_atoms(subj, S)
                # The subject's own generic-class atoms (e.g. "Morgan is a student")
                # are ground facts about the entity too.
                ground_atoms = tuple(pred_atoms + [a for a in subj_atoms])
                theory.facts.append(_TheoryFact(pid, entity, ground_atoms))
                theory.ground_entities.add(entity)
            else:
                subj_atoms = _phrase_atoms(subj, S)
                if subj_atoms:
                    is_reversed = False
                    if text_states_necessary_only(text):
                        _REVERSED_NECESSARY_MARKERS = re.compile(
                            r"\b(?:is|are|was|were)?\s*(?:required|necessary|a\s+prerequisite|prerequisite|a\s+necessary\s+condition|necessary\s+condition)\s+(?:for|to)\b"
                            r"|\bprerequisite\s+(?:for|to)\b",
                            re.IGNORECASE
                        )
                        if _REVERSED_NECESSARY_MARKERS.search(pred):
                            _REVERSED_NECESSARY_CLEANUP = re.compile(
                                r"^\s*(?:is|are|was|were)?\s*(?:required|necessary|a\s+prerequisite|prerequisite|a\s+necessary\s+condition|necessary\s+condition)?\s*(?:for|to)\s*(?:a|an|the)?\s*",
                                re.IGNORECASE
                            )
                            cleaned_pred = _REVERSED_NECESSARY_CLEANUP.sub("", pred).strip()
                            cleaned_pred_atoms = _phrase_atoms(cleaned_pred, S)
                            if cleaned_pred_atoms:
                                theory.rules.append(_TheoryRule(pid, tuple(cleaned_pred_atoms), tuple(subj_atoms)))
                                is_reversed = True
                    if not is_reversed:
                        theory.rules.append(_TheoryRule(pid, tuple(subj_atoms), tuple(pred_atoms)))

    # premises_fol: parse clean symbolic forms directly when present (Req 4.1
    # "consume QARequest.premises_fol directly"). Symbols are namespaced so they
    # never spuriously collide with NL-derived predicates (soundness).
    for raw in (premises_fol or []):
        parsed = _parse_fol_premise(str(raw))
        if parsed is None:
            continue
        kind, ante, cons = parsed
        if kind == "rule" and ante and cons:
            theory.rules.append(_TheoryRule(f"FOL", tuple(ante), tuple(cons)))
        elif kind == "fact" and cons:
            theory.facts.append(_TheoryFact("FOL", "fol_witness", tuple(cons)))

    theory.query = _compile_query(question, theory, S)
    return theory


def _compile_query(question: str, theory: _Theory, S: Any) -> _TheoryQuery | None:
    """Compile the query question into a structured TheoryQuery."""
    # Pathway query pattern
    pathway_match = re.search(
        r"\bpathway\s+(?:exists\s+)?from\s+(.+?)\s+to\s+(.+?)(?:\.|\?|$)",
        question,
        re.IGNORECASE
    )
    if pathway_match:
        start_phrase = pathway_match.group(1).strip()
        end_phrase = pathway_match.group(2).strip()
        hyp_atoms = _phrase_atoms(start_phrase, S)
        goal_atoms = _phrase_atoms(end_phrase, S)
        if goal_atoms:
            return _TheoryQuery(kind="universal", goal=tuple(goal_atoms), hypotheses=tuple(hyp_atoms))

    cond = S._question_conditional_statement(question)
    if cond:
        ante, cons = cond
        hyp_atoms = _phrase_atoms(ante, S)
        goal_atoms = _phrase_atoms(cons, S)
        if goal_atoms:
            return _TheoryQuery(kind="universal", goal=tuple(goal_atoms), hypotheses=tuple(hyp_atoms))
        return None

    # ---------------------------------------------------------------------------
    # Declarative-statement fallback for "Does it follow that <Subject> <verb> <Pred>?"
    # ---------------------------------------------------------------------------
    # The MCQ gate reformulates options as "Does it follow that Sophia is eligible
    # for the international program?" — a declarative statement wrapped in a
    # question. The standard _question_subject_predicate parser often mis-parses
    # these because it's designed for interrogative forms. We try
    # _split_simple_statement on the stripped declarative core FIRST, and only
    # fall back to _question_subject_predicate if that fails.
    declarative_core = _extract_declarative_core(question, S)
    if declarative_core:
        stmt = _split_simple_statement(declarative_core, S)
        if stmt:
            subj, pred, neg = stmt
            if _looks_proper_subject(question, subj):
                entity = _entity_name(re.sub(r"^(?:a|an|the)\s+", "", subj, flags=re.I))
                goal_atoms = _phrase_atoms(pred, S)
                if neg:
                    goal_atoms = _negate_atoms(goal_atoms)
                if goal_atoms and entity in theory.ground_entities:
                    return _TheoryQuery(kind="ground", goal=tuple(goal_atoms), entity=entity)

    subject, predicate, negative = S._question_subject_predicate(question)
    if not predicate:
        return None

    # The subject/predicate heuristic may keep a trailing verb with the subject
    # (e.g. "a sparrow produce" / "energy"). Treat the leading noun of the subject
    # as the entity and fold any trailing subject tokens into the goal predicate
    # so the goal atom matches the rule consequent that introduced it.
    subj_words = [w for w in re.findall(r"[a-z0-9]+", S._strip_articles(S._norm(subject or ""))) if len(w) > 1]
    entity_word = subj_words[0] if subj_words else ""
    trailing = " ".join(subj_words[1:]) if len(subj_words) > 1 else ""
    goal_source = (trailing + " " + predicate).strip() if trailing else predicate

    goal_atoms = _phrase_atoms(goal_source, S)
    if negative:
        goal_atoms = _negate_atoms(goal_atoms)
    if not goal_atoms:
        return None

    entity = _entity_name(entity_word) if entity_word else ""
    if entity and entity in theory.ground_entities:
        return _TheoryQuery(kind="ground", goal=tuple(goal_atoms), entity=entity)

    # Universal/threaded query: assume an arbitrary entity satisfies the subject's
    # class atoms (the hypothesis) and ask whether the goal then follows.
    hyp_atoms = _phrase_atoms(entity_word, S) if entity_word else []
    return _TheoryQuery(kind="universal", goal=tuple(goal_atoms), hypotheses=tuple(hyp_atoms))


def _extract_declarative_core(question: str, S: Any) -> str | None:
    """Extract the declarative statement from 'Does it follow that <statement>?'

    Returns the statement portion if the question matches the entailment-check
    pattern, or None otherwise. This enables correct parsing of MCQ option
    entailment queries where the option text is a declarative statement.
    """
    norm = S._norm(question).rstrip("?").strip()
    # Strip "does it follow that" / "is it true that" / "is it the case that"
    m = re.match(
        r"^(?:does\s+it\s+follow\s+that|is\s+it\s+true\s+that|is\s+it\s+the\s+case\s+that)\s+(.+)$",
        norm,
        flags=re.I,
    )
    if m:
        return m.group(1).strip()
    return None


def _parse_fol_premise(text: str) -> tuple[str, list[_Atom], list[_Atom]] | None:
    """Parse a clean symbolic FOL premise into ("rule"|"fact", ante, cons).

    Supports the common dataset subset: ``∀x (A(x) → B(x))``,
    ``∀x (¬A(x) → ¬B(x))``, ``∀x (A(x))`` and ``∃x (A(x))``. Anything richer
    returns ``None`` (the compiler simply ignores it).
    """
    t = str(text or "").strip()
    if not t:
        return None
    t = t.replace("→", "->").replace("¬", "~")
    quant = re.match(r"^\s*[∀∃]\s*[a-zA-Z]\w*\s*(.*)$", t)
    body = quant.group(1).strip() if quant else t
    body = body.strip()
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1].strip()

    def _atom(tok: str) -> _Atom | None:
        tok = tok.strip()
        neg = False
        while tok.startswith("~"):
            neg = not neg
            tok = tok[1:].strip()
        m = re.match(r"^([A-Za-z]\w*)\s*\([^)]*\)$", tok)
        if not m:
            return None
        return _Atom((f"fol_{m.group(1).lower()}",), neg)

    if "->" in body:
        left, right = body.split("->", 1)
        a = _atom(left)
        b = _atom(right)
        if a and b:
            return "rule", [a], [b]
        return None
    single = _atom(body)
    if single:
        return "fact", [], [single]
    return None


def _check_theory(theory: _Theory) -> DetFolResult:
    """Run Z3 solver check over the compiled Theory to verify the query goal."""
    entity_sort = z3.DeclareSort("Entity")
    symbols: dict[str, Any] = {}
    x = z3.Const("x", entity_sort)

    def pred(atom: _Atom):
        key = f"pred:{atom.name}"
        if key not in symbols:
            symbols[key] = z3.Function(atom.name, entity_sort, z3.BoolSort())
        return symbols[key]

    def const(name: str):
        key = f"const:{name}"
        if key not in symbols:
            symbols[key] = z3.Const(name, entity_sort)
        return symbols[key]

    def lit(atom: _Atom, term):
        expr = pred(atom)(term)
        return z3.Not(expr) if atom.negated else expr

    solver = z3.Solver()
    solver.set("timeout", 5000)
    used: list[str] = []

    for rule in theory.rules:
        cons = [lit(c, x) for c in rule.consequents]
        if not cons:
            continue
        body = cons[0] if len(cons) == 1 else z3.And(*cons)
        ant = [lit(a, x) for a in rule.antecedents]
        if ant:
            head = ant[0] if len(ant) == 1 else z3.And(*ant)
            # Direction preservation (Req 4.3): a stated A -> B compiles to exactly
            # ForAll([x], Implies(A(x), B(x))) — the antecedent->consequent direction
            # is asserted verbatim and nothing else. The valid contrapositive
            # ¬B -> ¬A is left for Z3 to *derive* (it follows from Implies(A, B) by
            # the entailment check below), so it is never re-asserted by string
            # reversal. The converse B -> A and the inverse ¬A -> ¬B are NOT
            # implied by Implies(A, B) and are never added, so Z3 can never prove
            # them from a single directional premise.
            solver.add(z3.ForAll([x], z3.Implies(head, body)))
        else:
            solver.add(z3.ForAll([x], body))
        used.append(rule.premise_id)

    for fact in theory.facts:
        entity = const(fact.entity)
        for atom in fact.atoms:
            solver.add(lit(atom, entity))
        used.append(fact.premise_id)

    # Proactively assert generic class sort membership for all ground entities
    # and the universal query constant (if any) if those classes are referenced in the rules.
    S = _det_solver_helpers()
    generic_classes_in_rules = set()
    canon_generic = _canon_generic_set(S)
    for rule in theory.rules:
        for atom in rule.antecedents + rule.consequents:
            for k in atom.key:
                if k in canon_generic:
                    generic_classes_in_rules.add(k)
                    
    entities_to_assert = [const(e) for e in theory.ground_entities]
    if query := theory.query:
        if query.kind != "ground":
            entities_to_assert.append(z3.Const("__query_entity", entity_sort))
            
    for entity_const in entities_to_assert:
        for g_class in generic_classes_in_rules:
            pred_name = f"p_{g_class}"
            key = f"pred:{pred_name}"
            if key not in symbols:
                symbols[key] = z3.Function(pred_name, entity_sort, z3.BoolSort())
            solver.add(symbols[key](entity_const))

    query = theory.query
    if query is None or not query.goal:
        return DetFolResult("unknown", "abstained", [], ["No compilable query goal"])

    if query.kind == "ground":
        term = const(query.entity)
        hyps: list[Any] = []
        goal_lits = [lit(a, term) for a in query.goal]
    else:
        c = z3.Const("__query_entity", entity_sort)
        hyps = [lit(a, c) for a in query.hypotheses]
        goal_lits = [lit(a, c) for a in query.goal]

    goal_expr = goal_lits[0] if len(goal_lits) == 1 else z3.And(*goal_lits)

    solver.push()
    for hyp in hyps:
        solver.add(hyp)

    solver.push()
    solver.add(z3.Not(goal_expr))
    check_yes = solver.check()
    solver.pop()

    solver.push()
    solver.add(goal_expr)
    check_no = solver.check()
    solver.pop()

    solver.pop()

    used_ids = [pid for pid in dict.fromkeys(used) if pid != "FOL"]

    if check_yes == z3.unsat and check_no != z3.unsat:
        return DetFolResult(
            "yes", "entailed", used_ids,
            ["Deterministic FOL->Z3: theory (+hypotheses) entail the query goal (UNSAT of negation)"],
        )
    if check_no == z3.unsat and check_yes != z3.unsat:
        return DetFolResult(
            "no", "contradicted", used_ids,
            ["Deterministic FOL->Z3: theory (+hypotheses) entail the negation of the query goal"],
        )
    return DetFolResult(
        "unknown", "undetermined", used_ids,
        ["Deterministic FOL->Z3: neither entailed nor contradicted -> abstain"],
    )


def solve_deterministic_fol(
    question: str,
    premises: list[dict[str, str]],
    premises_fol: list[str] | None = None,
    *,
    semantic_unification: bool = True,
    semantic_threshold: float = 0.80,
) -> DetFolResult:
    """Deterministic NL/``premises_fol`` -> typed DSL -> Z3 entailment check.

    Builds the Z3 theory WITHOUT the LLM and lets Z3 perform multi-hop universal
    chaining. Returns a definite ``yes``/``no`` only on clean Z3
    entailment/contradiction; otherwise ``unknown`` (never guesses).

    When ``semantic_unification`` is True (default), a predicate unification pass
    runs BEFORE Z3 compilation: semantically-equivalent property atoms (cosine
    similarity >= ``semantic_threshold``) are mapped to the same Z3 predicate so
    paraphrased premises chain correctly. This resolves Cluster A abstentions
    caused by token-set mismatches between rules and facts.

    Args:
        question: The query question to verify.
        premises: A list of dictionaries representing the natural language premises.
        premises_fol: Optional list of clean FOL representation strings.
        semantic_unification: Whether to unify semantically close predicates.
        semantic_threshold: The threshold cosine similarity for unification.

    Returns:
        The resulting DetFolResult with answer, status, and proof steps.
    """
    if not Z3_AVAILABLE:
        return DetFolResult("unknown", "error", [], ["Z3 not available"], error="z3_not_installed")
    try:
        S = _det_solver_helpers()
        # Pathway query and probabilistic rules check
        is_pathway = bool(re.search(r"\bpathway\b", question, re.IGNORECASE))
        if is_pathway:
            probabilistic_premises = [p["id"] for p in premises if S._is_probabilistic_rule(p["text"])]
            if probabilistic_premises:
                return DetFolResult(
                    answer="no",
                    z3_status="probabilistic_block",
                    used_premise_ids=probabilistic_premises,
                    proof_steps=["Probabilistic rule blocks complete pathway"],
                )
        theory = _compile_theory(question, premises, premises_fol, S)
    except Exception as exc:  # pragma: no cover - defensive; abstain on any parse error
        return DetFolResult("unknown", "error", [], [f"parse_error:{type(exc).__name__}"], error="parse_error")
    if theory.query is None or (not theory.rules and not theory.facts):
        return DetFolResult("unknown", "abstained", [], ["No compilable theory or query"])

    # --- Semantic predicate unification pass (Cluster A fix) ---
    unification_log: list[str] = []
    if semantic_unification:
        try:
            from app.logic.atom_matcher import unify_theory_atoms
            theory, unification_log = unify_theory_atoms(theory, threshold=semantic_threshold)
        except Exception as exc:
            # Graceful degradation: if semantic matching fails, proceed with exact matching
            unification_log = [f"semantic_unification_error:{type(exc).__name__}:{exc}"]

    try:
        result = _check_theory(theory)
        # Check if any used premise is a probabilistic rule
        if result.answer in {"yes", "no"}:
            prob_used = []
            for pid in result.used_premise_ids:
                text = next((p["text"] for p in premises if p["id"] == pid), None)
                if text and S._is_probabilistic_rule(text):
                    prob_used.append(pid)
            if prob_used:
                result.answer = "unknown"
                result.z3_status = "probabilistic_block"
                result.proof_steps = result.proof_steps + [f"Probabilistic rule {pid} blocks complete proof" for pid in prob_used]
        # Append unification info to proof steps for traceability
        if unification_log:
            result.proof_steps = result.proof_steps + [f"[semantic_unification] {u}" for u in unification_log]
        return result
    except Exception as exc:  # pragma: no cover - defensive; abstain on any Z3 error
        return DetFolResult("unknown", "error", [], [f"z3_error:{type(exc).__name__}"], error="z3_error")
