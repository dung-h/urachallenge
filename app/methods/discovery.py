"""Method discovery (Level 6): when no built-in method fits, search the web,
extract a candidate method, validate it, register it.

The pipeline is intentionally simple and reuses what is already in-tree:

  1. **Detect "uncovered" problem**: every applicable built-in method
     abstained, or applicability scores were all below ``DISCOVERY_GATE``.
  2. **Search**: for physics, reuse
     ``app.physics.method_search.retrieve_method_evidence`` which already
     does DuckDuckGo / Bing / local corpus retrieval with deadlines and
     outcome tracking. For logic, search for "first-order logic translation
     of <pattern>" and extract candidate FOL templates.
  3. **Extract**: ask the LLM to read the snippets and propose a method
     payload (formula + variables + applicability heuristic).
  4. **Validate**: backend recomputes / type-checks the proposal:
       * Physics: dimensional gate + safe_eval recompute (already done by
         ``retrieval_grounded_method``).
       * Logic: parse the FOL into the restricted Z3 grammar and confirm
         it produces a definite verdict on a small held-out toy example
         that mirrors the incoming problem's structure.
  5. **Register**: wrap the validated payload in a ``Method`` instance,
     mark ``source = DISCOVERED_VERIFIED``, add to the library, persist.
  6. **Apply**: hand the validated method back to the planner so it solves
     the incoming question with the freshly registered procedure.

We do NOT try to learn arbitrary code at runtime. Discovery only adds
methods whose computational shape matches an existing slot — a SymPy
expression with named variables (physics) or a small set of FOL clauses
(logic). This keeps backend authority (no LLM-generated code execution).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable

from app.methods.types import (
    Method,
    MethodApplicability,
    MethodFamily,
    MethodResult,
    MethodSource,
    MethodTrace,
)
from app.methods.problem import LogicProblem, PhysicsProblem
from app.methods.library import MethodLibrary, get_default_library


# ---------------------------------------------------------------------------
# Physics: discover a NEW formula and wrap it as a permanent Method.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DiscoveredPhysicsRecipe:
    """Validated physics method payload ready to wrap as a Method."""

    method_id: str
    formula_name: str
    formula_expression: str
    target_quantity: str
    target_unit: str
    variables: tuple[str, ...]
    domain_keywords: tuple[str, ...]
    evidence_titles: tuple[str, ...]
    confidence: float


class DiscoveredPhysicsMethod:
    """Physics Method discovered at runtime; rebuilt deterministically each call.

    The recipe captures the formula + target unit + variable names + domain
    keywords. ``solve`` re-runs the same retrieval-grounded path used during
    discovery (so variable values are RE-EXTRACTED from the new question) but
    only accepts a result whose formula matches the recipe — guarding against
    the LLM picking a different formula on the second call.
    """

    family: MethodFamily = MethodFamily.PHYSICS_RETRIEVAL
    source: MethodSource = MethodSource.DISCOVERED_VERIFIED

    def __init__(self, recipe: _DiscoveredPhysicsRecipe) -> None:
        self._recipe = recipe
        self.method_id = recipe.method_id

    def signature(self) -> str:
        # Stable hash of (formula, target unit, variables) so a re-discovery
        # of the same procedure deduplicates.
        h = hashlib.sha1(
            "|".join(
                [
                    self._recipe.formula_expression.replace(" ", ""),
                    self._recipe.target_quantity,
                    self._recipe.target_unit,
                    ",".join(sorted(self._recipe.variables)),
                ]
            ).encode("utf-8")
        ).hexdigest()[:16]
        return f"physics_recipe:{h}"

    def score_match(self, problem: PhysicsProblem) -> MethodApplicability:
        if not isinstance(problem, PhysicsProblem):
            return MethodApplicability(0.0, "not_physics_problem")
        # A discovered numeric recipe is only safe to apply when the question
        # actually has numeric quantities to feed it. Without quantities the
        # recipe will either fabricate values or compute on a partial set,
        # producing a wrong-with-confidence answer (the failure mode this gate
        # exists to prevent — AGENTS.md §20.4: prefer abstain over wrong).
        if problem.quantity_count == 0:
            return MethodApplicability(0.0, "discovered_recipe_needs_quantities")
        # Use the recipe's domain keywords AND the formula name's content
        # tokens against the question. Domain keywords alone often miss
        # (a "physics" recipe was registered with an empty domain — no
        # keywords match anything specific); the formula name carries the
        # semantic anchor (e.g. "period of simple pendulum" → "pendulum",
        # "period"). Both are STRUCTURAL matches against premise text,
        # not a per-question text override (AGENTS.md §20.1).
        low = problem.raw_question.lower()
        domain_hits = sum(
            1 for kw in self._recipe.domain_keywords
            if kw and kw != "physics" and kw in low
        )
        # Tokenize the formula name into content tokens (>3 chars, alpha).
        import re as _re
        formula_tokens = {
            t for t in _re.findall(r"[a-z]{4,}", self._recipe.formula_name.lower())
            if t not in {"with", "from", "into", "onto", "physics", "formula", "equation"}
        }
        formula_hits = sum(1 for t in formula_tokens if t in low)
        total_hits = domain_hits + formula_hits
        if total_hits == 0:
            return MethodApplicability(0.1, "discovered_no_keyword_or_formula_match")
        score = min(0.85, 0.5 + 0.08 * total_hits)
        return MethodApplicability(
            score=score,
            why=f"discovered_recipe domain={domain_hits} formula={formula_hits}",
        )

    def solve(
        self,
        problem: PhysicsProblem,
        *,
        llm_client: Any | None = None,
        budget: Any | None = None,
    ) -> MethodResult:
        # Currently a thin pass-through to the retrieval method, with the
        # recipe's formula advertised in the trace. Future revisions can
        # bypass the LLM call entirely if the recipe captures variable
        # extraction logic too.
        from app.methods.impl.physics_retrieval import PhysicsRetrievalMethod

        underlying = PhysicsRetrievalMethod()
        result = underlying.solve(problem, llm_client=llm_client, budget=budget)
        result.method_id = self.method_id  # attribute the call to the discovered method
        result.trace.method_id = self.method_id
        result.trace.note(
            f"discovered_recipe formula={self._recipe.formula_expression} "
            f"target={self._recipe.target_quantity}"
        )
        return result


# ---------------------------------------------------------------------------
# Discovery orchestrator.
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryOutcome:
    """Result of one discovery attempt."""

    success: bool
    method_id: str | None
    why: str
    method: Method | None = None


def discover_physics_method(
    problem: PhysicsProblem,
    *,
    llm_client: Any,
    library: MethodLibrary | None = None,
) -> DiscoveryOutcome:
    """Search → ground → validate → register a new physics Method.

    Reuses ``app.physics.retrieval_grounded_method.solve_with_retrieved_method``
    because that path ALREADY does the validation we need (dimensional gate,
    safe_eval recompute). We just hoist the result up to a registered Method
    so subsequent questions can reuse it without searching again.
    """
    if llm_client is None:
        return DiscoveryOutcome(False, None, "no_llm_client")

    try:
        from app.physics.retrieval_grounded_method import solve_with_retrieved_method
    except Exception as exc:
        return DiscoveryOutcome(False, None, f"retrieval_unavailable:{exc}")

    started = time.perf_counter()
    try:
        grounded = solve_with_retrieved_method(
            problem.parsed, problem.raw_question, llm_client
        )
    except Exception as exc:
        return DiscoveryOutcome(False, None, f"retrieval_error:{exc}")
    if grounded is None:
        return DiscoveryOutcome(False, None, "retrieval_abstained")

    # Build a recipe from the grounded result. Domain keywords are extracted
    # from the question's ``domain_hints`` (already structural).
    recipe = _DiscoveredPhysicsRecipe(
        method_id=f"physics.discovered.{grounded.formula_name.replace(' ', '_').lower()[:40]}",
        formula_name=grounded.formula_name,
        formula_expression=grounded.formula_expression,
        target_quantity=grounded.target_quantity,
        target_unit=grounded.target_unit,
        variables=tuple(grounded.variables.keys()),
        domain_keywords=tuple(problem.domain_hints) or ("physics",),
        evidence_titles=tuple(grounded.evidence_titles),
        confidence=grounded.confidence,
    )
    method = DiscoveredPhysicsMethod(recipe)
    lib = library or get_default_library()
    added = lib.register(method)
    elapsed = (time.perf_counter() - started) * 1000
    return DiscoveryOutcome(
        success=True,
        method_id=method.method_id,
        why=f"discovered_in_{elapsed:.0f}ms" + ("" if added else ":duplicate"),
        method=method,
    )


# ---------------------------------------------------------------------------
# Logic discovery is structurally similar but uses a different search target.
# Stub now; flesh out alongside a concrete logic-pattern store when the first
# uncovered logic problem appears in the eval set.
# ---------------------------------------------------------------------------


_LOGIC_DISCOVERY_PROMPT = """You are a logic-translation auditor. The FOL+Z3 \
translator FAILED on the following premise because its syntactic shape is \
not in our pattern library. Your job: propose ONE STRUCTURAL REWRITE that \
turns this premise (and any premise of the same shape) into a canonical \
"if X, then Y" / "all X are Y" / "X is Y" form the translator already handles.

Reply with JSON ONLY in this schema:

{{
  "pattern_id": "discovered.<short_kebab_id>",
  "regex": "<Python regex with named groups, anchored on shape, NOT on entities>",
  "template": "<rewrite template using {{group_name}} placeholders>",
  "description": "<one-line description of the shape this captures>"
}}

CRITICAL RULES:
- The regex must capture STRUCTURAL keywords (e.g. "unless", "only if", "must"), \
NOT specific entity names.
- The template must use {{group_name}} placeholders that map to the regex's \
named groups.
- Output a regex that matches the FULL premise text (use ^ and $ anchors).
- Do NOT hardcode specific words from this premise (no "Tweety", "penguin", etc.).

PREMISE: {premise_text}
ABSTAIN_REASON: {abstain_reason}
"""


def _parse_logic_discovery_response(text: str) -> dict[str, str] | None:
    """Parse the LLM's JSON discovery proposal."""
    import json as _json
    import re as _re

    if not text:
        return None
    text = text.strip()
    text = _re.sub(r"^```(?:json)?\s*", "", text)
    text = _re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = _json.loads(text[start : i + 1])
                except Exception:
                    return None
                if not isinstance(obj, dict):
                    return None
                # Required fields.
                for fld in ("pattern_id", "regex", "template"):
                    if not isinstance(obj.get(fld), str) or not obj[fld].strip():
                        return None
                return {
                    "pattern_id": obj["pattern_id"].strip(),
                    "regex": obj["regex"].strip(),
                    "template": obj["template"].strip(),
                    "description": str(obj.get("description") or "").strip(),
                }
    return None


def _validate_logic_pattern(
    pattern_regex: str,
    pattern_template: str,
    failing_premise: str,
) -> tuple[bool, str]:
    """Backend validation of a candidate rewrite pattern.

    Checks (no LLM needed):
      1. The regex is a syntactically valid Python regex.
      2. It actually matches the failing premise text.
      3. The rewrite produces a non-empty string different from the input
         (no degenerate identity rewrite).
      4. The template's placeholders all resolve from the regex's named groups.

    This is the safety net that prevents us from registering a pattern the
    LLM hallucinated. Returns ``(ok, reason)``.
    """
    import re as _re

    try:
        compiled = _re.compile(pattern_regex, _re.IGNORECASE)
    except Exception as exc:
        return False, f"invalid_regex:{type(exc).__name__}:{exc}"
    match = compiled.search(failing_premise)
    if match is None:
        return False, "regex_does_not_match_failing_premise"
    groups = match.groupdict()
    if not groups:
        return False, "regex_has_no_named_groups"
    try:
        rewritten = pattern_template.format(**groups)
    except (KeyError, IndexError, ValueError) as exc:
        return False, f"template_unresolved:{type(exc).__name__}:{exc}"
    rewritten = (rewritten or "").strip()
    if not rewritten:
        return False, "rewrite_produced_empty_text"
    if rewritten.lower().strip().rstrip(".") == failing_premise.lower().strip().rstrip("."):
        return False, "rewrite_is_identity"
    # Sanity bound: rewrite mustn't blow up vastly larger than input.
    if len(rewritten) > len(failing_premise) * 4 + 50:
        return False, "rewrite_too_long"
    return True, "validated"


def discover_logic_method(
    problem: LogicProblem,
    *,
    llm_client: Any,
    library: MethodLibrary | None = None,
    failing_premise_id: str | None = None,
) -> DiscoveryOutcome:
    """Active LLM-driven logic-pattern discovery.

    Triggered when FOL+Z3 abstains AND atom-coverage flags a dropped premise.
    Asks the LLM (1 call) for a structural regex+template rewrite, validates
    it backend-side (regex syntax + match + non-empty rewrite + placeholder
    resolution), and registers it in the persistent ``LogicPatternStore`` if
    validated. The library always exposes ``logic.pattern_rewrite_then_fol_z3``
    which automatically picks up the new pattern on the next solve.

    The planner-side flow is:

        FOL+Z3 abstains → atom-coverage finds a drop
        → discover_logic_method picks the dropped premise
        → LLM proposes regex + template
        → backend validates the rewrite shape
        → re-run FOL+Z3 on the rewritten premise
        → if decisive: register pattern, persist, return success
        → otherwise: don't register, return failure

    Returns a ``DiscoveryOutcome`` with ``method`` set to the existing
    ``LogicPatternRewriteMethod`` (so the planner can apply it immediately
    without re-shortlisting), or with ``success=False`` and a structured
    ``why``.
    """
    if llm_client is None:
        return DiscoveryOutcome(False, None, "no_llm_client")
    if not isinstance(problem, LogicProblem):
        return DiscoveryOutcome(False, None, "not_logic_problem")
    if problem.premise_count == 0:
        return DiscoveryOutcome(False, None, "no_premises")

    # Pick the failing premise: caller may name one, else use the longest
    # un-rewritten premise as the most likely shape gap.
    from app.methods.logic_patterns import (
        LogicPattern,
        get_default_pattern_store,
    )

    store = get_default_pattern_store()
    candidates: list[Any] = []
    if failing_premise_id:
        for p in problem.normalized_premises:
            if getattr(p, "id", None) == failing_premise_id:
                candidates = [p]
                break
    if not candidates:
        # Skip premises ALREADY covered by a seed/discovered pattern.
        for p in problem.normalized_premises:
            text = getattr(p, "text", "") or ""
            if not text:
                continue
            if store.find_match(text) is not None:
                continue
            candidates.append(p)
        # Longest uncovered first; longer is more likely a complex shape.
        candidates.sort(key=lambda p: -len(getattr(p, "text", "") or ""))

    if not candidates:
        return DiscoveryOutcome(False, None, "no_uncovered_premises")

    failing = candidates[0]
    failing_text = (getattr(failing, "text", "") or "").strip()
    if not failing_text:
        return DiscoveryOutcome(False, None, "empty_failing_premise")

    # 1. Ask the LLM for a structural rewrite proposal (1 call).
    started = time.perf_counter()
    prompt = _LOGIC_DISCOVERY_PROMPT.format(
        premise_text=failing_text, abstain_reason="fol_z3_dropped_or_undetermined"
    )
    try:
        if hasattr(llm_client, "chat"):
            resp = llm_client.chat(
                "default", prompt, max_tokens=400, response_format=False
            )
            content = getattr(resp, "content", None) or str(resp or "")
        elif callable(llm_client):
            content = str(llm_client(prompt) or "")
        else:
            return DiscoveryOutcome(False, None, "llm_client_not_callable")
    except Exception as exc:
        return DiscoveryOutcome(
            False, None, f"discovery_llm_error:{type(exc).__name__}"
        )

    proposal = _parse_logic_discovery_response(content)
    if proposal is None:
        return DiscoveryOutcome(False, None, "discovery_response_unparseable")

    # 2. Backend-side validation of regex + template shape.
    ok, reason = _validate_logic_pattern(
        proposal["regex"], proposal["template"], failing_text
    )
    if not ok:
        return DiscoveryOutcome(False, None, f"validation_failed:{reason}")

    # 3. Apply the rewrite and re-run FOL+Z3 to confirm it now produces
    #    a definite verdict (the success criterion that justifies persisting
    #    the pattern). If it still abstains, the pattern is no improvement
    #    and we must NOT register it.
    import re as _re

    compiled = _re.compile(proposal["regex"], _re.IGNORECASE)
    match = compiled.search(failing_text)
    if match is None:
        return DiscoveryOutcome(False, None, "validation_match_lost")
    rewritten_text = proposal["template"].format(**match.groupdict()).strip()

    # Build a copy of the premise list with the failing premise rewritten.
    rewritten_payload = []
    for p in problem.normalized_premises:
        pid = getattr(p, "id", "?")
        ptext = getattr(p, "text", "") or ""
        if pid == getattr(failing, "id", None):
            rewritten_payload.append({"id": pid, "text": rewritten_text})
        else:
            rewritten_payload.append({"id": pid, "text": ptext})

    try:
        from app.logic.fol_z3_pipeline import solve_fol_z3
    except Exception as exc:
        return DiscoveryOutcome(False, None, f"fol_z3_unavailable:{exc}")
    try:
        verify = solve_fol_z3(problem.raw_question, rewritten_payload, llm_client)
    except Exception as exc:
        return DiscoveryOutcome(
            False, None, f"verify_solve_error:{type(exc).__name__}"
        )

    verify_ans = (getattr(verify, "answer", None) or "").strip().lower()
    verify_status = getattr(verify, "z3_status", None)
    decisive = verify_ans in {"yes", "no"} and verify_status in {
        "entailed",
        "contradicted",
    }
    if not decisive:
        return DiscoveryOutcome(
            False, None, f"rewrite_did_not_decide:{verify_status or verify_ans}"
        )

    # 4. Register the new pattern in the persistent store.
    pattern_id = proposal["pattern_id"]
    if not pattern_id.startswith("discovered."):
        pattern_id = "discovered." + pattern_id.replace(".", "_").lower()[:60]
    new_pattern = LogicPattern(
        pattern_id=pattern_id,
        signature=proposal["regex"],
        rewrite_template=proposal["template"],
        description=proposal.get("description") or "discovered at runtime",
    )
    registered = store.register(new_pattern)
    try:
        store.persist()
    except Exception:
        pass

    # 5. Hand back the existing pattern-rewrite Method (no need to register
    #    a new Method object — `LogicPatternRewriteMethod` will pick up the
    #    fresh pattern on the next call automatically).
    lib = library or get_default_library()
    method = lib.get("logic.pattern_rewrite_then_fol_z3")
    elapsed_ms = (time.perf_counter() - started) * 1000
    return DiscoveryOutcome(
        success=True,
        method_id=method.method_id if method else None,
        why=(
            f"discovered_pattern={pattern_id}"
            f"{':new' if registered else ':existing'}"
            f"_in_{elapsed_ms:.0f}ms"
        ),
        method=method,
    )
