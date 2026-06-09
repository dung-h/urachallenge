"""FOL + Z3 Pipeline for Logic-Based Educational Queries.

Architecture (see AGENTS.md §20.5):
  question + premises
  → [LLM: FOL Translator]   — translate NL → typed FOL clauses
  → [Z3 Theorem Prover]     — check entailment / contradiction / unknown
  → [LLM: Explanation Writer] — narrate the proof trace in natural language
  → [Backend: Schema Validation]

Invariant: **The LLM translates and explains. Z3 decides.**

The LLM is called twice:
  1. fol_translate()    — structured output: FOL for each premise + query
  2. fol_explain()      — structured output: explanation from proof trace

If FOL translation fails or confidence is below threshold, the pipeline
abstains and the caller falls back to the token-BFS symbolic solver.

Z3 usage:
  - Each universal statement "All X are Y"  → ForAll([x], Implies(X(x), Y(x)))
  - Each conditional "If P then Q"          → ForAll([x], Implies(P(x), Q(x)))
  - Each ground fact "Alex is Y"            → Y(alex)
  - Each negated fact "Alex is not Y"       → Not(Y(alex))
  - Entailment check: theory ⊨ query?
    Check sat(theory ∧ ¬query) — if UNSAT → "yes"
    Check sat(theory ∧ query)  — if UNSAT → "no"
    Otherwise → "unknown"
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.explanation_worker import build_explanation_trace, generate_explanation, validate_explanation_rewrite


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class FOLClause:
    """A single FOL-translated clause."""
    premise_id: str
    natural_language: str
    fol: str
    clause_type: str  # "universal_positive" | "universal_negative" | "conditional" | "ground_fact" | "ground_negation"
    confidence: float = 1.0


@dataclass
class FOLTranslation:
    """Result of FOL translation for all premises + query."""
    clauses: list[FOLClause]
    query_fol: str
    query_negated_fol: str
    query_subject: str
    query_predicate: str
    translation_confidence: float
    unsupported_premise_ids: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.translation_confidence >= 0.6


@dataclass
class Z3Result:
    """Result of Z3 entailment check."""
    answer: str              # "yes" | "no" | "unknown"
    z3_status: str           # "entailed" | "contradicted" | "undetermined" | "abstained" | "error"
    used_premise_ids: list[str]
    proof_steps: list[str]   # Human-readable proof trace from the Z3 derivation
    confidence: float
    latency_ms: float
    error: str | None = None


@dataclass
class FolZ3Solution:
    """Final solution from the FOL+Z3 pipeline."""
    answer: str
    explanation: str
    premises: list[str]      # Used premise IDs
    fol_query: str
    z3_status: str
    proof_steps: list[str]
    confidence: float
    llm_calls: int
    latency_ms: float
    method: str = "fol_z3_pipeline"
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None and self.answer in {"yes", "no", "unknown"}


# ---------------------------------------------------------------------------
# Step 1: LLM → FOL Translation
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    """Retrieve an integer value from environment variables with a default fallback."""
    try:
        value = int(os.environ.get(name, "").strip())
    except Exception:
        return default
    return value if value > 0 else default


def _env_bool(name: str, default: bool) -> bool:
    """Retrieve a boolean value from environment variables with a default fallback."""
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _norm(text: str) -> str:
    """Normalize whitespace and lowercase a string."""
    return re.sub(r"\s+", " ", str(text or "").lower().strip())


def _directionality_gate_ok(
    answer: str,
    used_premise_texts: list[str],
    question: str,
) -> tuple[bool, str | None]:
    """Reject brittle necessary→sufficient rescues.

    This is intentionally conservative: if the proof only cites necessary-only
    language ("requires", "only if") without any sufficient-rule language
    ("if ... then", "all ... are", "who ... may ..."), we abstain.

    The necessary-only / sufficient-rule classification is centralized in
    ``app.logic.dsl_compiler`` (``text_states_necessary_only`` /
    ``text_states_sufficient_rule``) per ADR 0004 so the SAME rule governs the
    deterministic fast-path post-check in ``solver.solve`` and this LLM-rescue
    gate. Necessary-only wording can therefore NEVER license a positive "yes"
    on either path (Req 4.4).
    """

    # The known high-risk failure mode is necessary-only text being used to
    # incorrectly justify a positive ("yes") conclusion.
    if answer != "yes":
        return True, None

    texts = [_norm(t) for t in used_premise_texts if str(t or "").strip()]
    if not texts:
        return False, "directionality_gate:no_used_premises"

    from app.logic.dsl_compiler import (
        text_states_necessary_only,
        text_states_sufficient_rule,
    )

    necessary_only = any(text_states_necessary_only(t) for t in texts)
    has_sufficient = any(text_states_sufficient_rule(t) for t in texts)

    if necessary_only and not has_sufficient:
        return False, "directionality_gate:necessary_only_without_sufficient_rule"

    # Extra conservative: if the question is an eligibility/status claim, do not
    # accept a proof that only cites requirement phrasing.
    q = _norm(question)
    if necessary_only and any(token in q for token in ["eligible", "qualify", "permitted", "allowed", "may register", "can register"]):
        if not has_sufficient:
            return False, "directionality_gate:eligibility_requires_sufficient_rule"

    return True, None

_TRANSLATE_SYSTEM_PROMPT = """You are a First-Order Logic (FOL) translator.
Your job is to translate natural language premises into FOL clauses.
You receive a JSON object with:
  - "premises": list of {"id": "P1", "text": "..."}
  - "query": "..." (the yes/no question to answer)
  - "query_subject": "..." (the entity being asked about, e.g. "Alex")
  - "query_predicate": "..." (what is being claimed, e.g. "eligible for the internship")

Output ONLY a valid JSON object with this structure:
{
  "clauses": [
    {
      "premise_id": "P1",
      "fol": "ForAll([x], Implies(student(x), eligible(x)))",
      "clause_type": "universal_positive",
      "confidence": 0.95
    },
    ...
  ],
  "query_fol": "eligible(alex)",
  "query_negated_fol": "Not(eligible(alex))",
  "translation_confidence": 0.90,
  "unsupported": []
}

clause_type must be one of:
  "universal_positive"  — "All X are Y", "Every X is Y", "Anyone who X..."
  "universal_negative"  — "No X are Y", "No X can Y"
  "conditional"         — "If P then Q" (may still be universal)
  "ground_fact"         — "Alex is Y", "Alex has Y"
  "ground_negation"     — "Alex is not Y", "Alex does not have Y"

Rules:
- Use single-word predicate names in snake_case (e.g., "eligible_for_scholarship")
- Subject entities must be lowercase identifiers (e.g., "alex", "john")
- Universal subjects use a variable [x]
- If you cannot translate a premise with confidence > 0.6, add its id to "unsupported"
- Do NOT reason or answer the question — only translate
"""

_EXPLAIN_SYSTEM_PROMPT = """You are an explanation writer for a logic reasoning system.
You receive a JSON object with:
  - "answer": "yes" | "no" | "unknown"
  - "query": the original question
  - "proof_steps": list of strings describing the proof trace
  - "used_premises": list of {"id": "P1", "text": "..."}

Write a clear, concise natural-language explanation (2-4 sentences) of why the answer is correct.
- Ground every claim in a specific cited premise by its ID (e.g., "P1 states that...")
- Follow the proof_steps in order
- Do not introduce facts not in the used_premises
- End with "Therefore the answer is [yes/no/unknown]."

Output ONLY a JSON object:
{
  "explanation": "..."
}
"""


def _call_llm_structured(
    llm_client: Any,
    system_prompt: str,
    user_content: str,
    max_tokens: int = 300,
) -> dict | None:
    """Call the LLM client with structured output request.

    Returns parsed JSON dict or None on failure.
    Compatible with OpenAI-compatible API clients.
    """
    try:
        json_chat_with_system = getattr(llm_client, "json_chat_with_system", None)
        if callable(json_chat_with_system):
            return json_chat_with_system(system_prompt, user_content, max_tokens=max_tokens, role="fol_z3_pipeline")
        chat = getattr(llm_client, "chat", None)
        if callable(chat):
            # Project-local OpenAICompatibleLLMClient shape: chat(role, user, ...).
            result = chat("fol_z3_pipeline", user_content, max_tokens=max_tokens, response_format=True)
            if getattr(result, "error", None) or not str(getattr(result, "content", "")).strip():
                return None
            content = result.content
        elif hasattr(llm_client, "chat") and hasattr(llm_client.chat, "completions"):
            # Raw OpenAI SDK-compatible shape.
            response = llm_client.chat.completions.create(
                model=getattr(llm_client, "_model", getattr(llm_client, "model", "default")),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
        elif callable(llm_client):
            # Simple callable: llm_client(prompt) -> str
            content = llm_client(system_prompt + "\n\n" + user_content)
        else:
            return None

        # Strip markdown fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content.strip())
        return json.loads(content)
    except Exception:
        return None


def fol_translate(
    premises: list[dict[str, str]],  # [{"id": "P1", "text": "..."}]
    query: str,
    query_subject: str,
    query_predicate: str,
    llm_client: Any,
) -> FOLTranslation:
    """Translate premises and query to FOL using LLM.

    Args:
        premises: List of premise dictionaries with keys id and text.
        query: The logical query question.
        query_subject: The subject entity of the query.
        query_predicate: The predicate property of the query.
        llm_client: The LLM model client.

    Returns:
        The generated FOLTranslation.
    """
    user_content = json.dumps({
        "premises": premises,
        "query": query,
        "query_subject": query_subject,
        "query_predicate": query_predicate,
    }, ensure_ascii=False, indent=2)

    result = _call_llm_structured(llm_client, _TRANSLATE_SYSTEM_PROMPT, user_content, max_tokens=1200)
    if result is None:
        return FOLTranslation(
            clauses=[], query_fol="", query_negated_fol="",
            query_subject=query_subject, query_predicate=query_predicate,
            translation_confidence=0.0,
            error="LLM translation call failed or returned invalid JSON",
        )

    clauses = []
    for c in result.get("clauses", []):
        clauses.append(FOLClause(
            premise_id=c.get("premise_id", "?"),
            natural_language=next(
                (p["text"] for p in premises if p["id"] == c.get("premise_id")), ""
            ),
            fol=c.get("fol", ""),
            clause_type=c.get("clause_type", "ground_fact"),
            confidence=float(c.get("confidence", 0.8)),
        ))

    return FOLTranslation(
        clauses=clauses,
        query_fol=result.get("query_fol", ""),
        query_negated_fol=result.get("query_negated_fol", ""),
        query_subject=query_subject,
        query_predicate=query_predicate,
        translation_confidence=float(result.get("translation_confidence", 0.0)),
        unsupported_premise_ids=result.get("unsupported", []),
    )


# ---------------------------------------------------------------------------
# Step 2: Z3 Theorem Proving
# ---------------------------------------------------------------------------

_FOL_TOKEN_RE = re.compile(r"\s*(ForAll|Exists|Implies|And|Or|Not|[A-Za-z][A-Za-z0-9_]*|\[|\]|\(|\)|,)\s*")


class _RestrictedFOLParser:
    """Parse the tiny FOL subset emitted by the translator.

    This deliberately supports only unary predicates over one entity sort and
    the small set of boolean/quantifier constructors used in prompts. Unsupported
    syntax raises ValueError and the caller abstains.
    """

    def __init__(self, text: str, z3: Any, entity_sort: Any, symbols: dict[str, Any]) -> None:
        self.tokens = self._tokenize(text)
        self.pos = 0
        self.z3 = z3
        self.entity_sort = entity_sort
        self.symbols = symbols
        self.variables: dict[str, Any] = {"x": z3.Const("x", entity_sort)}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens: list[str] = []
        pos = 0
        while pos < len(text):
            match = _FOL_TOKEN_RE.match(text, pos)
            if not match:
                if text[pos:].strip():
                    raise ValueError("unsupported FOL token")
                break
            tokens.append(match.group(1))
            pos = match.end()
        return tokens

    def parse(self) -> Any:
        expr = self._expr()
        if self.pos != len(self.tokens):
            raise ValueError("trailing FOL tokens")
        return expr

    def _peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _take(self, expected: str | None = None) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("unexpected end of FOL")
        if expected is not None and token != expected:
            raise ValueError(f"expected {expected}")
        self.pos += 1
        return token

    def _ident(self) -> str:
        token = self._take()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            raise ValueError("expected identifier")
        return token

    def _expr(self) -> Any:
        name = self._ident()
        self._take("(")
        if name in {"ForAll", "Exists"}:
            result = self._quantifier(name)
        elif name in {"Implies", "And", "Or"}:
            first = self._expr()
            self._take(",")
            items = [first, self._expr()]
            while self._peek() == ",":
                self._take(",")
                items.append(self._expr())
            if name == "Implies" and len(items) != 2:
                raise ValueError("Implies expects two arguments")
            result = getattr(self.z3, name)(*items)
        elif name == "Not":
            result = self.z3.Not(self._expr())
        else:
            arg_name = self._ident()
            result = self._predicate(name)(self._term(arg_name))
        self._take(")")
        return result

    def _quantifier(self, name: str) -> Any:
        var_name = self._bound_var()
        self._take(",")
        body = self._expr()
        var = self.variables.setdefault(var_name, self.z3.Const(var_name, self.entity_sort))
        return getattr(self.z3, name)([var], body)

    def _bound_var(self) -> str:
        if self._peek() == "[":
            self._take("[")
            var_name = self._ident()
            self._take("]")
            return var_name
        return self._ident()

    def _term(self, name: str) -> Any:
        if name in self.variables:
            return self.variables[name]
        key = f"const:{name}"
        if key not in self.symbols:
            self.symbols[key] = self.z3.Const(name, self.entity_sort)
        return self.symbols[key]

    def _predicate(self, name: str) -> Any:
        key = f"pred:{name}"
        if key not in self.symbols:
            self.symbols[key] = self.z3.Function(name, self.entity_sort, self.z3.BoolSort())
        return self.symbols[key]


def _parse_z3_expr(fol: str, z3: Any, entity_sort: Any, symbols: dict[str, Any]) -> Any:
    """Parse a FOL string into a Z3 expression using RestrictedFOLParser."""
    return _RestrictedFOLParser(fol, z3, entity_sort, symbols).parse()


def _build_z3_theory(translation: FOLTranslation) -> tuple[Any, Any, dict, list[str]] | None:
    """Build Z3 solver from FOL translation.

    Returns (solver, query_expr, symbols, used_premise_ids) or None if Z3 unavailable.
    The FOL strings are parsed by _RestrictedFOLParser. Do not use eval here.
    """
    try:
        import z3
    except ImportError:
        return None

    entity_sort = z3.DeclareSort("Entity")
    symbols: dict[str, Any] = {}
    solver = z3.Solver()
    used_premise_ids: list[str] = []
    axioms: list[Any] = []

    for clause in translation.clauses:
        if not clause.fol or clause.confidence < 0.5:
            continue
        try:
            expr = _parse_z3_expr(clause.fol, z3, entity_sort, symbols)
            axioms.append(expr)
            used_premise_ids.append(clause.premise_id)
        except Exception:
            continue  # Skip untranslatable clause

    if not axioms:
        return None

    solver.add(*axioms)

    # Evaluate query
    try:
        query_expr = _parse_z3_expr(translation.query_fol, z3, entity_sort, symbols)
    except Exception:
        return None

    return solver, query_expr, symbols, used_premise_ids


def z3_check(translation: FOLTranslation) -> Z3Result:
    """Run Z3 entailment check on the translated FOL theory.

    Args:
        translation: The FOLTranslation containing theory clauses and query.

    Returns:
        The resulting Z3Result containing verdict, status, and proof trace.
    """
    t0 = time.perf_counter()

    try:
        import z3
    except ImportError:
        return Z3Result(
            answer="unknown", z3_status="abstained",
            used_premise_ids=[], proof_steps=["Z3 not installed — abstaining"],
            confidence=0.0, latency_ms=0.0,
            error="z3 package not installed",
        )

    if not translation.success:
        return Z3Result(
            answer="unknown", z3_status="abstained",
            used_premise_ids=[], proof_steps=["FOL translation below confidence threshold"],
            confidence=0.0, latency_ms=0.0,
            error=f"Translation failed: {translation.error or 'low confidence'}",
        )

    built = _build_z3_theory(translation)
    if built is None:
        return Z3Result(
            answer="unknown", z3_status="abstained",
            used_premise_ids=[], proof_steps=["Could not build Z3 theory from FOL clauses"],
            confidence=0.0, latency_ms=0.0,
            error="Z3 theory construction failed",
        )

    solver, query_expr, _, used_premise_ids = built

    # Check entailment: theory ⊨ query  ↔  theory ∧ ¬query is UNSAT
    solver.push()
    solver.add(z3.Not(query_expr))
    check_yes = solver.check()
    solver.pop()

    # Check contradiction: theory ⊨ ¬query  ↔  theory ∧ query is UNSAT
    solver.push()
    solver.add(query_expr)
    check_no = solver.check()
    solver.pop()

    latency_ms = (time.perf_counter() - t0) * 1000

    entails_yes = (check_yes == z3.unsat)
    entails_no = (check_no == z3.unsat)

    if entails_yes and not entails_no:
        answer = "yes"
        status = "entailed"
        confidence = 0.92
        proof_steps = [
            f"Theory built from {len(used_premise_ids)} premises: {', '.join(used_premise_ids)}",
            f"FOL query: {translation.query_fol}",
            "Z3 check: theory ∧ ¬query is UNSAT",
            "Conclusion: premises entail the query",
            "Answer: yes",
        ]
    elif entails_no and not entails_yes:
        answer = "no"
        status = "contradicted"
        confidence = 0.92
        proof_steps = [
            f"Theory built from {len(used_premise_ids)} premises: {', '.join(used_premise_ids)}",
            f"FOL query: {translation.query_fol}",
            "Z3 check: theory ∧ query is UNSAT",
            "Conclusion: premises contradict the query",
            "Answer: no",
        ]
    elif entails_yes and entails_no:
        answer = "unknown"
        status = "inconsistent"
        confidence = 0.30
        proof_steps = [
            f"Theory built from {len(used_premise_ids)} premises",
            "Z3 detected inconsistency: both query and its negation are entailed",
            "Premises may be contradictory",
            "Answer: unknown (inconsistent theory)",
        ]
    else:
        answer = "unknown"
        status = "undetermined"
        confidence = 0.70
        proof_steps = [
            f"Theory built from {len(used_premise_ids)} premises: {', '.join(used_premise_ids)}",
            f"FOL query: {translation.query_fol}",
            "Z3 check: neither query nor its negation is entailed",
            "Premises do not provide sufficient information",
            "Answer: unknown",
        ]

    return Z3Result(
        answer=answer,
        z3_status=status,
        used_premise_ids=used_premise_ids,
        proof_steps=proof_steps,
        confidence=confidence,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Step 3: LLM → Explanation Writer
# ---------------------------------------------------------------------------

def fol_explain(
    answer: str,
    query: str,
    z3_result: Z3Result,
    used_premises: list[dict[str, str]],  # [{"id": "P1", "text": "..."}]
    llm_client: Any,
) -> str:
    """Generate a trace-grounded explanation from the Z3 proof trace.

    Task 13.1 merges what used to be two LLM-using explanation steps
    (``fol_explain`` here and ``runtime_trace.maybe_rewrite_explanation`` in the
    main pipeline) into a single deterministic explanation entry point. This
    function now:

      1. builds an :class:`ExplanationTrace` from the Z3 proof trace and the
         selected premises;
      2. produces a deterministic, trace-grounded explanation via
         :func:`app.explanation_worker.generate_explanation` (zero LLM calls);
      3. optionally tries one gated LLM rewrite, accepting it only when it
         passes :func:`app.explanation_worker.validate_explanation_rewrite`
         (Task 13.2 acceptance gate). On rejection, the deterministic
         template stands.

    The deterministic template guarantees the final answer is named
    (Req 10.6), at least one selected premise ID is cited literally
    (Req 10.1), and abstentions name the specific missing condition
    (Req 10.5, 11.3). Returning the deterministic template on rewrite
    failure preserves the no-hallucination invariant (Req 10.2, 10.3).

    Args:
        answer: The logical query answer ("yes", "no", "unknown").
        query: The original question string.
        z3_result: The Z3Result containing the proof trace.
        used_premises: The list of used premise dictionaries.
        llm_client: The LLM model client.

    Returns:
        The trace-grounded natural language explanation.
    """

    proof_steps_payload = [{"value": step} for step in (z3_result.proof_steps or [])]
    trace = build_explanation_trace(
        request_id=None,
        question=query,
        task_type="logic",
        answer=answer,
        explanation="",  # filled in by the deterministic template below
        fol=None,
        selected_premise_ids=[p["id"] for p in used_premises if p.get("id")],
        selected_premise_texts=[p["text"] for p in used_premises if p.get("text")],
        cot=list(z3_result.proof_steps or []),
        proof_steps=proof_steps_payload,
        physics_variables={},
        solver_used="fol_z3_pipeline",
        confidence=float(getattr(z3_result, "confidence", 0.0) or 0.0),
    )
    template = generate_explanation(trace)
    # Re-anchor the trace's solver_explanation against the deterministic
    # template so the LLM rewrite is judged against it, not an empty string.
    trace = build_explanation_trace(
        request_id=None,
        question=query,
        task_type="logic",
        answer=answer,
        explanation=template,
        fol=None,
        selected_premise_ids=[p["id"] for p in used_premises if p.get("id")],
        selected_premise_texts=[p["text"] for p in used_premises if p.get("text")],
        cot=list(z3_result.proof_steps or []),
        proof_steps=proof_steps_payload,
        physics_variables={},
        solver_used="fol_z3_pipeline",
        confidence=float(getattr(z3_result, "confidence", 0.0) or 0.0),
    )

    user_content = json.dumps({
        "answer": answer,
        "query": query,
        "proof_steps": z3_result.proof_steps,
        "used_premises": used_premises,
    }, ensure_ascii=False, indent=2)

    result = _call_llm_structured(llm_client, _EXPLAIN_SYSTEM_PROMPT, user_content, max_tokens=200)
    if result and isinstance(result.get("explanation"), str):
        rewritten = result["explanation"].strip()
        ok, _validation_errors = validate_explanation_rewrite(rewritten, trace)
        if ok:
            return rewritten

    return template


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def _parse_query_entities(question: str) -> tuple[str, str]:
    """Extract subject and predicate from a yes/no question.

    Returns (subject, predicate). Falls back to ("entity", "property") on failure.
    """
    q = re.sub(r"\s+", " ", question.lower().strip()).rstrip("?")
    # Strip "does it follow that" wrapper (produced by _option_yesno_query) to
    # expose the declarative core: "Alex needs a longer membership duration".
    q = re.sub(r"^does it follow that\s+", "", q)
    # "Does/Is/Can X <predicate>?"
    for pat in [
        r"^(?:does|can|is|did)\s+([a-z][a-z0-9 ]+?)\s+(.+)$",
        r"^(?:is|are)\s+([a-z][a-z0-9 ]+?)\s+(.+)$",
    ]:
        m = re.match(pat, q)
        if m:
            subject = m.group(1).strip().replace(" ", "_")
            predicate = m.group(2).strip().replace(" ", "_")
            return subject, predicate
    # Declarative fallback: "<Subject> <verb-phrase>" (e.g. "alex needs ...")
    m = re.match(r"^([a-z][a-z0-9 ]*?)\s+((?:is|are|has|have|can|must|needs?|should|will|would|shall|may|might|does|do|did|was|were|had)\b.+)$", q)
    if m:
        subject = m.group(1).strip().replace(" ", "_")
        predicate = m.group(2).strip().replace(" ", "_")
        return subject, predicate
    return "entity", "property"


def solve_fol_z3(
    question: str,
    premises: list[dict[str, str]],  # [{"id": "P1", "text": "..."}]
    llm_client: Any,
    translation_confidence_threshold: float = 0.65,
) -> FolZ3Solution:
    """Full FOL+Z3 pipeline: translate → prove → explain.

    Per Task 11.2 / Req 7.1, the LLM FOL translator runs as a gated **k=1**
    fallback only — it is invoked exclusively when both the deterministic
    fast path and the deterministic FOL compiler have abstained AND premises
    exist. The acceptance discipline (independent backend agreement) lives in
    ``app.logic.solver._with_fol_z3_pipeline`` (Task 2.1); the consensus
    (k=3) verdict is no longer the acceptance authority. Setting
    ``URA_FOL_Z3_TRANSLATION_K=1`` and ``URA_FOL_Z3_REQUIRE_CONSENSUS=0`` in
    the environment is also supported for reproducibility, but the new
    defaults below already enforce k=1 without consensus.

    Args:
        question: The yes/no (or binary) question.
        premises: List of {"id": ..., "text": ...} dicts.
        llm_client: OpenAI-compatible client or callable. The caller may pass
            a budget-gated wrapper (see ``app.runtime_workflow.BudgetGatedClient``)
            so the cap+deadline applies here automatically.
        translation_confidence_threshold: Minimum confidence to trust the FOL
            translation.

    Returns:
        FolZ3Solution with answer, explanation, proof trace, and metadata.
    """
    use_dsl = _env_bool("URA_FOL_Z3_USE_DSL", True)
    if use_dsl:
        result = _solve_fol_z3_dsl(question, premises, llm_client, translation_confidence_threshold)
        if result is not None and result.error is None:
            return result

    return _solve_fol_z3_legacy(question, premises, llm_client, translation_confidence_threshold)


def _solve_fol_z3_dsl(
    question: str,
    premises: list[dict[str, str]],
    llm_client: Any,
    translation_confidence_threshold: float = 0.65,
) -> FolZ3Solution | None:
    """DSL-based FOL+Z3 pipeline using typed JSON schema.

    Implements a Logic-LM-style self-refinement loop: when the first
    translation yields a non-decisive Z3 verdict (abstained / undetermined /
    error / low confidence) or leaves premises unsupported, the specific
    failure is fed back to the LLM and a re-translation is attempted (bounded
    by ``URA_FOL_Z3_REFINE_ROUNDS``, default 2). The backend Z3 solver remains
    the decision authority; refinement only improves the *translation* the
    solver reasons over, which generalizes far better than adding bespoke
    heuristics for each failing phrasing (AGENTS.md §13, §20).
    """
    from app.logic.dsl_compiler import (
        DSL_TRANSLATE_SYSTEM_PROMPT,
        parse_dsl_response,
        compile_dsl_to_z3,
    )

    t0 = time.perf_counter()
    subject, predicate = _parse_query_entities(question)

    base_payload = {
        "premises": premises,
        "query": question,
        "query_subject": subject,
        "query_predicate": predicate,
    }

    max_rounds = max(0, _env_int("URA_FOL_Z3_REFINE_ROUNDS", 2))
    # Wall-clock deadline for the whole refinement loop. On 20-36 premise
    # items each LLM translation of the huge prompt can take 60s+, and three
    # refinement rounds + explanation + downstream rescue stacked to 250s+,
    # blowing the ~30s budget (AGENTS.md §15.4) and making large items
    # un-evaluable. Cap the loop wall time (default 45s) so we stop refining
    # and let the caller fall through to the deterministic path. Sound: a
    # timed-out refinement returns None (abstain), never a guess.
    refine_deadline_s = float(_env_int("URA_FOL_Z3_REFINE_DEADLINE_S", 45))
    _loop_start = time.perf_counter()
    llm_calls = 0
    feedback: str | None = None
    last_translation = None
    last_compile = None

    for round_idx in range(max_rounds + 1):
        if round_idx > 0 and (time.perf_counter() - _loop_start) > refine_deadline_s:
            break  # deadline exceeded — stop refining, fall through to fallback
        user_content = json.dumps(base_payload, ensure_ascii=False, indent=2)
        if feedback:
            # Append structured feedback from the previous failed attempt so the
            # LLM can repair its own translation (self-refinement).
            user_content = (
                user_content
                + "\n\nThe previous translation FAILED for this reason:\n"
                + feedback
                + "\nProduce a corrected JSON translation that fixes the issue. "
                "Re-check clause `type`, `direction`, `polarity`, and that every "
                "premise needed to answer the query is translated (not left in "
                "`unsupported`)."
            )

        result = _call_llm_structured(
            llm_client, DSL_TRANSLATE_SYSTEM_PROMPT, user_content, max_tokens=1200
        )
        llm_calls += 1
        if result is None:
            feedback = "The translation output was empty or not valid JSON."
            continue

        translation = parse_dsl_response(result)
        last_translation = translation
        if not translation.success or translation.confidence < translation_confidence_threshold:
            feedback = (
                f"The translation confidence was too low ({translation.confidence:.2f}) "
                f"or no clauses parsed (error: {translation.error or 'none'}). "
                "Translate every premise into a clause with confidence >= 0.7."
            )
            continue

        compile_result = compile_dsl_to_z3(translation)
        last_compile = compile_result

        # AGENTS.md §24 / North Star Phase F2: faithfulness gate. Before
        # accepting a definite Z3 verdict, verify the translator did NOT
        # silently drop a premise. We use the cheap atom-coverage check
        # here (no LLM call) so it is free; the round-trip LLM check is
        # reserved for the final acceptance pass elsewhere.
        try:
            from app.methods.faithfulness import check_atom_coverage as _faith_atom_coverage
            # Build clause "blobs" from the translation: every clause's
            # subject_class / condition / conclusion / entity / predicate.
            blobs: list[str] = []
            for c in getattr(translation, "clauses", []) or []:
                for fld in ("subject_class", "condition", "conclusion", "entity", "predicate"):
                    val = getattr(c, fld, None) or ""
                    if val:
                        blobs.append(str(val))
            class _PremiseShim:
                def __init__(self, pid: str, txt: str) -> None:
                    self.id = pid
                    self.text = txt
            shim_inputs = [_PremiseShim(p["id"], p["text"]) for p in premises]
            faith_report = _faith_atom_coverage(shim_inputs, blobs, min_overlap=1)
        except Exception:
            faith_report = None

        # Decisive + gate-passing verdict → accept immediately.
        if compile_result.answer in {"yes", "no"} and compile_result.z3_status in {
            "entailed",
            "contradicted",
        }:
            # Faithfulness has VETO power on accept. A decisive verdict on
            # an INCOMPLETE translation is exactly the failure mode the gate
            # is designed to catch (e.g. "All birds can fly unless they are
            # penguins" → if the rule clause is dropped, Z3 entails 'yes'
            # from "Tweety is a bird" alone).
            if faith_report is not None and faith_report.has_issues:
                feedback = faith_report.feedback_text() or _build_refinement_feedback(
                    translation, compile_result
                )
                continue
            used_texts = [
                p["text"] for p in premises
                if p["id"] in set(compile_result.used_premise_ids)
            ]
            gate_ok, gate_reason = _directionality_gate_ok(
                compile_result.answer, used_texts, question
            )
            if gate_ok:
                used_premises = [
                    p for p in premises if p["id"] in set(compile_result.used_premise_ids)
                ]
                explanation = fol_explain(
                    compile_result.answer, question, compile_result, used_premises, llm_client
                )
                total_ms = (time.perf_counter() - t0) * 1000
                return FolZ3Solution(
                    answer=compile_result.answer,
                    explanation=explanation,
                    premises=compile_result.used_premise_ids,
                    fol_query=f"DSL:{translation.query.predicate}",
                    z3_status=compile_result.z3_status,
                    proof_steps=compile_result.proof_steps
                    + ([f"self_refinement_rounds={round_idx}"] if round_idx else []),
                    confidence=compile_result.confidence,
                    llm_calls=llm_calls + 1,  # +1 for the explanation call
                    latency_ms=total_ms,
                    method="fol_z3_dsl_pipeline",
                )
            feedback = (
                f"The Z3 verdict '{compile_result.answer}' was rejected by the "
                f"direction gate: {gate_reason}. This usually means a 'necessary' "
                "condition (requires / only if) was wrongly translated as "
                "'sufficient', or a rule direction was reversed. Re-check each "
                "clause's `direction`."
            )
            continue

        # Non-decisive verdict → build feedback for the next round.
        feedback = _build_refinement_feedback(translation, compile_result)

    # All rounds exhausted without a decisive, gated verdict → abstain (None)
    # so the caller falls through to the deterministic / legacy paths.
    return None


def _build_refinement_feedback(translation: Any, compile_result: Any) -> str:
    """Compose a concise, structural feedback string for translation refinement.

    Names the specific non-decisive condition (undetermined / unsupported
    premises / error) so the LLM can repair its translation. Never references
    question text verbatim — only the structural failure mode.
    """
    parts: list[str] = []
    status = getattr(compile_result, "z3_status", "")
    if status == "undetermined":
        parts.append(
            "Z3 found the theory did NOT entail the query or its negation "
            "(insufficient/disconnected clauses). A premise needed to chain "
            "from the query subject to the predicate may be missing, or a "
            "clause's subject_class/condition/conclusion atoms do not match "
            "across premises (use identical snake_case atom names so the chain "
            "connects)."
        )
        # Atom-connectivity diagnostic: the most common small-model failure is
        # a ground fact whose predicate does not match the condition atom that a
        # conditional rule needs to fire (e.g. fact `did_not_eat` vs query
        # `hungry` with rule condition `did_not_eat` -> conclusion `is_hungry`,
        # but the query predicate is `hungry` != `is_hungry`). Surface the atom
        # inventory so the LLM can align names.
        try:
            clause_atoms: list[str] = []
            for c in getattr(translation, "clauses", []) or []:
                for fld in ("condition", "conclusion", "predicate"):
                    val = getattr(c, fld, "") or ""
                    if val:
                        clause_atoms.append(val)
            q = getattr(translation, "query", None)
            q_pred = getattr(q, "predicate", "") if q is not None else ""
            if q_pred and clause_atoms and q_pred not in clause_atoms:
                parts.append(
                    f"The query predicate '{q_pred}' does not exactly match any "
                    f"clause atom {sorted(set(clause_atoms))}. Rename atoms so the "
                    "query predicate is IDENTICAL to the rule conclusion it should "
                    "follow from, and ensure every condition atom is established by "
                    "a ground_fact clause."
                )
        except Exception:
            pass
    elif status in {"error", "abstained"}:
        parts.append(
            f"The Z3 compilation did not run cleanly (status={status}, "
            f"error={getattr(compile_result, 'error', None)}). Ensure each "
            "clause has the correct `type` and the required fields for that type."
        )
    unsupported = list(getattr(translation, "unsupported", []) or [])
    if unsupported:
        parts.append(
            "These premises were left UNSUPPORTED and excluded from reasoning: "
            f"{', '.join(unsupported)}. Translate them too if they are relevant."
        )
    if not parts:
        parts.append(
            "The translation did not yield a decisive verdict. Re-examine clause "
            "directions and ensure atom names match across premises."
        )
    return " ".join(parts)



def _solve_fol_z3_legacy(
    question: str,
    premises: list[dict[str, str]],
    llm_client: Any,
    translation_confidence_threshold: float = 0.65,
) -> FolZ3Solution:
    """Full FOL+Z3 pipeline: translate → prove → explain.

    Args:
        question: The yes/no (or binary) question.
        premises: List of {"id": ..., "text": ...} dicts.
        llm_client: OpenAI-compatible client or callable.
        translation_confidence_threshold: Minimum confidence to trust the FOL translation.

    Returns:
        FolZ3Solution with answer, explanation, proof trace, and metadata.
    """
    t0 = time.perf_counter()
    llm_calls = 0

    # Per Task 11.2 / Req 7.1: gated k=1 fallback. The consensus (k=3) check is
    # NOT the acceptance authority any more — independent-backend agreement
    # lives in ``app.logic.solver._with_fol_z3_pipeline`` (Task 2.1). Default
    # to k=1 and no consensus so the LLM contributes at most one translation
    # per request; the env vars remain for reproducibility experiments only.
    k_translations = _env_int("URA_FOL_Z3_TRANSLATION_K", 1)
    require_consensus = _env_bool("URA_FOL_Z3_REQUIRE_CONSENSUS", False)

    # Step 1: Parse question entities
    subject, predicate = _parse_query_entities(question)

    # Step 2+3: LLM → FOL Translation(s) → Z3 checks
    candidates: list[tuple[FOLTranslation, Z3Result]] = []
    k = max(1, k_translations)
    for idx in range(k):
        # Deterministic diversity: rotate premise order.
        # This helps even with temperature=0 when the translator is order-sensitive.
        premises_variant = list(premises[idx:] + premises[:idx]) if premises else []
        translation = fol_translate(premises_variant, question, subject, predicate, llm_client)
        llm_calls += 1
        if not translation.success or translation.translation_confidence < translation_confidence_threshold:
            continue
        z3_result = z3_check(translation)
        if z3_result.z3_status in {"abstained", "error"}:
            continue
        # Apply semantic-direction gate before considering this candidate for rescue.
        used_texts = [p["text"] for p in premises_variant if p["id"] in set(z3_result.used_premise_ids)]
        gate_ok, gate_reason = _directionality_gate_ok(z3_result.answer, used_texts, question)
        if not gate_ok:
            continue
        candidates.append((translation, z3_result))

    if not candidates:
        total_ms = (time.perf_counter() - t0) * 1000
        return FolZ3Solution(
            answer="unknown",
            explanation="FOL translation/Z3 verification did not produce a gated decisive verdict. Falling back to symbolic solver.",
            premises=[],
            fol_query="",
            z3_status="abstained",
            proof_steps=[
                "No gated decisive Z3 verdict",
                f"k_translations={k_translations}",
            ],
            confidence=0.0,
            llm_calls=llm_calls,
            latency_ms=total_ms,
            method="fol_z3_pipeline_abstained",
            error="no_gated_candidate",
        )

    # Prefer decisive verdicts; if consensus is required, need >=2 matching yes/no.
    decisive = [(t, z) for (t, z) in candidates if z.answer in {"yes", "no"} and z.z3_status in {"entailed", "contradicted"}]
    if require_consensus:
        verdicts = {
            "yes": [c for c in decisive if c[1].answer == "yes"],
            "no": [c for c in decisive if c[1].answer == "no"],
        }
        agreed = verdicts["yes"] if len(verdicts["yes"]) >= 2 else verdicts["no"] if len(verdicts["no"]) >= 2 else []
        if not agreed:
            total_ms = (time.perf_counter() - t0) * 1000
            return FolZ3Solution(
                answer="unknown",
                explanation="Z3 candidates did not reach consensus under strict gates. Falling back to symbolic solver.",
                premises=[],
                fol_query="",
                z3_status="abstained",
                proof_steps=[
                    "No consensus across Z3 candidates",
                    f"k_translations={k_translations}",
                ],
                confidence=0.0,
                llm_calls=llm_calls,
                latency_ms=total_ms,
                method="fol_z3_pipeline_abstained",
                error="no_consensus",
            )
        decisive = agreed

    # Pick the best decisive candidate by combined confidence.
    best_translation, best_z3 = max(
        decisive or candidates,
        key=lambda pair: float(pair[1].confidence) * float(pair[0].translation_confidence),
    )

    # Step 4: LLM → Explanation from proof trace (for the selected candidate)
    used_premises = [p for p in premises if p["id"] in set(best_z3.used_premise_ids)]
    explanation = fol_explain(best_z3.answer, question, best_z3, used_premises, llm_client)
    llm_calls += 1

    total_ms = (time.perf_counter() - t0) * 1000
    return FolZ3Solution(
        answer=best_z3.answer,
        explanation=explanation,
        premises=best_z3.used_premise_ids,
        fol_query=best_translation.query_fol,
        z3_status=best_z3.z3_status,
        proof_steps=best_z3.proof_steps,
        confidence=best_z3.confidence * best_translation.translation_confidence,
        llm_calls=llm_calls,
        latency_ms=total_ms,
        method="fol_z3_pipeline" if k_translations <= 1 else "fol_z3_pipeline_k",
    )
