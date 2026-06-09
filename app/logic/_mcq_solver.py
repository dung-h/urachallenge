"""Solver and evaluator for multiple-choice educational questions.

Uses forward chaining, rule scoring, and FOL/Z3 translation sidecars to evaluate
options and select the best supported answer.
"""

from __future__ import annotations
import re
from typing import Any
from app.logic.premise_selector import Premise, select_premises
from app.logic.proof_trace import ProofStep, build_proof_steps
from app.schemas import AnswerSource, VerifierEvidence
from app.logic._proof_classes import LogicSolution, Rule
from app.logic._text_primitives import (
    _norm, _singular, _stem, _strip_articles, _content_tokens,
    _predicate_tokens, _predicate_matches, _specific_tokens,
    _terms_overlap, _tokens_cover, _clean_tokens_cover, _clean_content_tokens,
    _conditional_parts, _is_negated, _negates_condition, _predicate_supported,
    _contains_entity, _is_probabilistic_rule, _split_subject_predicate,
    IGNORABLE_PREDICATE_WORDS, _NEGATION_PATTERN, _CANNOT_PROVE,
    _is_public_logic_sample_text,
)
from app.logic._question_parser import (
    _labeled_options, _option_text_to_question, _is_abstain_option,
    _question_polarity, _question_existential, _question_conditional_statement,
    _question_status_subject, _question_asks_antecedent, _question_subject_predicate,
    _failure_status_prop, _choice_for_failure_status, _choice_for_unknown,
    _choice_for_boolean_answer,
)
from app.logic._rule_matcher import (
    _match_all_rule, _match_no_rule, _match_if_rule, _match_rule,
    _class_matches, _antecedent_triggered, _implies, _fact_implies_target,
    _negate_clause, _implication_edges, _support_path, _is_universal_quantifier,
    _is_existential_quantifier, _option_is_existential, _has_matching_existential_support,
    _universal_object_rule, _object_prop,
)
from app.logic._fol_bridge import _premises_contain_contradiction
from app.logic._subject_chain import (
    _are_contradictory_premises,
    _has_universal_no_conflict,
    _mcq_option_with_subject,
    _number_satisfied,
    _proof_path_consistent,
    _solve_conditional_must_true_mcq,
    _solve_conditional_status_unknown,
    _solve_failure_status_conditionals,
    _solve_modus_tollens_negative_consequent,
    _solve_object_property_chain,
    _solve_rules,
    _subject_threaded_chain,
    _subject_threaded_commit_is_grounded,
    _fact_subject_kind,
    _trim_to_proof_path,
    _universal_contrapositive_support,
    _universal_negative_support,
    _universal_positive_support,
)

def _get_solve_rules():
    """Retrieve the rule solver function from the main solver module."""
    import sys
    solver = sys.modules.get("app.logic.solver")
    return getattr(solver, "_solve_rules", _solve_rules)

def _get_triggered_negative_status_support():
    """Retrieve the negative status support function from the main solver module."""
    import sys
    solver = sys.modules.get("app.logic.solver")
    return getattr(solver, "_triggered_negative_status_support", _triggered_negative_status_support)

def _get_det_fol_entails_option():
    """Retrieve the deterministic FOL entailment check from the main solver module."""
    import sys
    solver = sys.modules.get("app.logic.solver")
    return getattr(solver, "_det_fol_entails_option", _det_fol_entails_option)

def _get_score_mcq_option():
    """Retrieve the MCQ option scorer function from the main solver module."""
    import sys
    solver = sys.modules.get("app.logic.solver")
    return getattr(solver, "_score_mcq_option", _score_mcq_option)

def _score_mcq_option(option_text: str, premises: list[Premise]) -> tuple[int | None, list[Premise], str]:
    """Score an MCQ option based on deductive rules and FOL/Z3 solver returns."""
    cost, support, reason = _score_mcq_option_inner(option_text, premises)
    if cost is not None:
        return cost, support, reason
    query = _option_yesno_query(option_text)
    payload = [{"id": p.id, "text": p.text} for p in premises]
    from app.logic.dsl_compiler import solve_deterministic_fol
    res = solve_deterministic_fol(query, payload)
    if res.error is None and res.answer == "yes" and res.z3_status == "entailed":
        support = [p for p in premises if p.id in res.used_premise_ids]
        return len(support), support, "deterministic Z3 entailment"
    return None, [], "no support found"

def _score_mcq_option_inner(option_text: str, premises: list[Premise]) -> tuple[int | None, list[Premise], str]:
    """Determine verification cost, supporting premises, and reason for an MCQ option."""
    option = option_text.strip()
    if not option:
        return None, [], "empty option"
    if _is_abstain_option(option):
        return None, [], "explicit unknown option"

    compound_universal = _score_compound_universal_option(option, premises)
    if compound_universal is not None:
        return compound_universal

    negative_status_fn = _get_triggered_negative_status_support()
    negative_status = negative_status_fn(option, premises)
    if negative_status is not None:
        support, reason = negative_status
        return max(1, len(support)), support, reason

    solve_rules_fn = _get_solve_rules()

    # Handle conjunctions with "but"
    if " but " in option.lower():
        parts = re.split(r"\bbut\b", option, flags=re.I)
        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1].strip()
            # If right part doesn't have a clear subject, prepend the subject from left
            if not right.lower().startswith(("he ", "she ", "it ", "john ", "alex ", "minh ", "dr. ")):
                words = left.split()
                if words:
                    sub_word = words[0].rstrip(",.:;")
                    right = f"{sub_word} {right}"
            left_q = _option_text_to_question(left)
            right_q = _option_text_to_question(right)
            left_res = solve_rules_fn(left_q, premises)
            right_res = solve_rules_fn(right_q, premises)
            if left_res[0] == "yes" and right_res[0] == "yes":
                support = list(dict.fromkeys(left_res[1] + right_res[1]))
                return max(1, len(support)), support, f"conjunction: {left_res[2]} and {right_res[2]}"
            return None, [], "conjunction unsupported"

    both_match = re.match(r"^(.+?)\s+can\s+both\s+(.+?)\s+and\s+(.+)$", option, flags=re.I)
    if both_match:
        subject = both_match.group(1).strip()
        left = both_match.group(2).strip()
        right = both_match.group(3).strip()
        left_res = solve_rules_fn(f"Can {subject} {left}?", premises)
        right_res = solve_rules_fn(f"Can {subject} {right}?", premises)
        if left_res[0] == "yes" and right_res[0] == "yes":
            support = list(dict.fromkeys(left_res[1] + right_res[1]))
            return max(1, len(support)), support, f"both-conjunction: {left_res[2]} and {right_res[2]}"
        return None, [], "both-conjunction unsupported"

    if option.lower().startswith(("if ", "all ", "no ")):
        edges = _implication_edges(premises)
        rule = _match_if_rule(option) or _match_all_rule(option) or _match_no_rule(option)
        if rule:
            if len(rule) == 2:
                antecedent, consequent = rule
                path = _support_path(antecedent, consequent, edges)
                if path is not None:
                    direct_question = _option_text_to_question(option)
                    direct_result = solve_rules_fn(direct_question, premises)
                    if direct_result[0] == "yes" and _mcq_support_is_deterministic(direct_result[2], list(direct_result[1])):
                        if any(_is_probabilistic_rule(p.text) for p in path):
                            return None, [], "MCQ implication chain is uncertain"
                        support = list(dict.fromkeys(path))
                        return max(1, len(support)), support, "MCQ implication chain"
                for premise in premises:
                    premise_rule = _match_rule(premise.text)
                    if not premise_rule:
                        continue
                    premise_antecedent, premise_consequent = premise_rule
                    if (
                        _is_negated(antecedent)
                        and _is_negated(consequent)
                        and (
                            _predicate_matches(_negate_clause(premise_consequent), antecedent)
                            or _predicate_matches(antecedent, _negate_clause(premise_consequent))
                            or _negative_clauses_equivalent(_negate_clause(premise_consequent), antecedent)
                        )
                        and (
                            _predicate_matches(_negate_clause(premise_antecedent), consequent)
                            or _predicate_matches(consequent, _negate_clause(premise_antecedent))
                            or _negative_clauses_equivalent(_negate_clause(premise_antecedent), consequent)
                        )
                    ):
                        return 1, [premise], "MCQ contrapositive of conditional rule"
        return None, [], "MCQ implication unsupported"

    direct_question = _option_text_to_question(option)
    direct_result = solve_rules_fn(direct_question, premises)
    if direct_result[0] == "yes" and _mcq_support_is_deterministic(direct_result[2], list(direct_result[1])):
        if "invalid_inference" in direct_result[2].lower():
            return None, [], "MCQ invalid inference blocked"
        if _is_negated(option) and not any(_is_negated(p.text) or _match_no_rule(p.text) for p in direct_result[1]):
            return None, [], "MCQ negative option lacks negative evidence"
        option_low = _norm(option)
        if _option_is_existential(option):
            if not _has_matching_existential_support(option, list(direct_result[1])):
                return None, [], "MCQ existential option lacks existential premise support"
        if re.search(r"\b(?:only|solely|exclusively|just)\b", option_low):
            if not any(re.search(r"\b(?:only|solely|exclusively|just)\b", _norm(p.text)) for p in direct_result[1]):
                return None, [], "MCQ restrictive option lacks restrictive support"
        if re.search(r"\b(?:need|needs|require|requires|required|must)\b", option_low):
            if not any(re.search(r"\b(?:need|needs|require|requires|required|must)\b", _norm(p.text)) for p in direct_result[1]):
                return None, [], "MCQ need/require claim lacks necessary-condition support"
        if any(_is_probabilistic_rule(p.text) for p in direct_result[1]):
            return None, [], "MCQ option is uncertain"
        support = list(dict.fromkeys(direct_result[1]))
        return max(1, len(support)), support, direct_result[2]
    return None, [], direct_result[2]

def _score_compound_universal_option(option_text: str, premises: list[Premise]) -> tuple[int | None, list[Premise], str] | None:
    """Evaluate and score a compound universal option (e.g. all X are both Y and Z)."""
    option = _norm(option_text).rstrip(".")
    both = re.match(r"^(?:all|every)\s+(.+?)\s+(?:are|is|have|has|can|do|does)\s+both\s+(.+?)\s+and\s+(.+)$", option)
    if not both:
        return None
    subject = both.group(1).strip()
    left = both.group(2).strip()
    right = both.group(3).strip()

    queries = [
        f"Are all {subject} {left}?",
        f"Are all {subject} {right}?",
    ]
    support: list[Premise] = []
    reasons: list[str] = []
    solve_rules_fn = _get_solve_rules()
    for query in queries:
        ans, selected, reason, _cannot_prove = solve_rules_fn(query, premises)
        if ans != "yes":
            return None, [], f"compound universal option unsupported: {reason}"
        support.extend(selected)
        reasons.append(reason)
    support = list(dict.fromkeys(support))
    return max(1, len(support)), support, "compound universal option: " + " and ".join(reasons)

def _solve_mcq(
    question: str,
    premises: list[Premise],
    choices: list[str] | None = None,
    premises_fol: list[str] | None = None,
    llm_client: object | None = None,
    call_budget: object | None = None,
) -> tuple[str, list[Premise], str] | None:
    """Orchestrate solving of multiple-choice questions by scoring and checking each option."""
    q_low = _norm(question)
    if re.search(r"\bwhich\s+(?:of\s+the\s+(?:following\s+)?)?premises?\b", q_low):
        return None
    options = _all_mcq_options(question, choices)
    if not options:
        return None

    cond_res = _solve_conditional_must_true_mcq(question, premises, choices)
    if cond_res is not None:
        return cond_res


    try:
        from app.logic.semantic import solve_mcq_semantic
        semantic = solve_mcq_semantic(question, premises, dict(options))
    except Exception:
        semantic = None
    if semantic is not None:
        return semantic.answer, semantic.support, semantic.reason

    # Find the abstain/unknown option label
    unknown_label = None
    for label, text in options:
        if _is_abstain_option(text.strip()):
            unknown_label = label
            break

    # 1. Deterministic DSL/Z3 compiler pass
    from app.logic.dsl_compiler import solve_deterministic_fol
    payload = [{"id": p.id, "text": p.text} for p in premises]

    det_yes_options = []
    det_no_options = []
    for label, text in options:
        if label == unknown_label:
            continue
        query = _option_yesno_query(text)
        res = solve_deterministic_fol(query, payload, premises_fol)
        if res.error is None:
            if res.answer == "yes" and res.z3_status == "entailed":
                det_yes_options.append((label, text, res))
            elif res.answer == "no" and res.z3_status == "contradicted":
                det_no_options.append((label, text, res))

    # Case A: Exactly one option is entailed deterministically, or one has strictly fewest premises
    has_det_signal = False
    if det_yes_options:
        det_yes_options.sort(key=lambda x: len(x[2].used_premise_ids))
        min_used = len(det_yes_options[0][2].used_premise_ids)
        tied_det = [o for o in det_yes_options if len(o[2].used_premise_ids) == min_used]
        if len(tied_det) == 1:
            label, text, res = tied_det[0]
            support = [p for p in premises if p.id in res.used_premise_ids]
            return label, support, "deterministic Z3 entailment"
        else:
            has_det_signal = True

    # Case C: All other options are contradicted deterministically
    if unknown_label is not None and len(det_no_options) == len(options) - 1:
        used_ids = set()
        for label, text, res in det_no_options:
            used_ids.update(res.used_premise_ids)
        support = [p for p in premises if p.id in used_ids]
        return unknown_label, support, "deterministic Z3 contradiction of all other options"

    # 1.4. Token-BFS option verification (bridges the improved deterministic
    # chaining — anaphora, causative, universal-propagation, comparative
    # thresholds — into MCQ selection). The DSL/Z3 pass above uses a separate
    # atom compiler that misses these shapes; the token-BFS `solve_deterministic`
    # path now resolves them. For each non-abstain option, convert to a yes/no
    # question and check entailment via the BFS solver. If EXACTLY ONE option
    # is entailed "yes", select it. Sound: requires a unique decisive option;
    # ties or none → fall through (no guess).
    try:
        from app.logic.solver import solve as _bfs_solve
    except Exception:
        _bfs_solve = None
    if _bfs_solve is not None:
        bfs_yes = []
        for label, text in options:
            if label == unknown_label:
                continue
            oq = _option_yesno_query(text)
            try:
                ores = _bfs_solve(oq, [p.text for p in premises], use_llm=False)
            except Exception:
                continue
            if (getattr(ores, "answer", "") or "").strip().lower() == "yes":
                bfs_yes.append((label, text, list(getattr(ores, "premises", []) or [])))
        if len(bfs_yes) == 1:
            label, text, used_ids = bfs_yes[0]
            support = [p for p in premises if p.id in set(used_ids)] or select_premises(question, premises)
            return label, support, "token-BFS option entailment (single decisive option)"

    status_res = _solve_conditional_status_unknown(question, premises)
    if status_res is not None:
        return status_res

    failure_res = _solve_failure_status_conditionals(question, premises)
    if failure_res is not None:
        return failure_res

    # 1.5. Legacy heuristics fallback pass
    score_mcq_option_fn = _get_score_mcq_option()
    legacy_yes_options = []
    for label, text in options:
        if label == unknown_label:
            continue
        cost, support, reason = score_mcq_option_fn(text, premises)
        if cost is not None:
            legacy_yes_options.append((label, text, cost, support, reason))

    if legacy_yes_options:
        legacy_yes_options.sort(key=lambda x: x[2])
        min_cost = legacy_yes_options[0][2]
        tied_options = [opt for opt in legacy_yes_options if opt[2] == min_cost]
        if len(tied_options) == 1:
            best_label, best_text, cost, support, reason = tied_options[0]
            return best_label, support, reason
        else:
            if unknown_label is not None:
                support = select_premises(question, premises)
                return unknown_label, support, "MCQ scoring tied, choosing unknown option"
            return "unknown", [], "MCQ scoring tied"

    if has_det_signal:
        return "unknown", [], "deterministic Z3 entails multiple options"

    # 2. LLM-assisted FOL+Z3 fallback pass
    if llm_client is not None and (call_budget is None or (hasattr(call_budget, "can_spend") and call_budget.can_spend())):
        try:
            from app.logic.fol_z3_pipeline import fol_translate, _build_z3_theory, _parse_query_entities, FOLTranslation
            import z3 as z3_mod
        except Exception:
            z3_mod = None

        if z3_mod is not None:
            first_option_query = _option_yesno_query(options[0][1])
            subject, predicate = _parse_query_entities(first_option_query)

            try:
                translation = fol_translate(
                    premises=payload,
                    query=first_option_query,
                    query_subject=subject,
                    query_predicate=predicate,
                    llm_client=llm_client,
                )
            except Exception:
                translation = None

            if translation and translation.success and translation.clauses:
                llm_predicates = set()
                for clause in translation.clauses:
                    if clause.fol:
                        for pred_match in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", clause.fol):
                            pred_name = pred_match.group(1)
                            if pred_name not in {"ForAll", "Exists", "Implies", "And", "Or", "Not"}:
                                llm_predicates.add(pred_name)

                ground_entity = None
                for clause in translation.clauses:
                    if clause.fol and clause.clause_type in ("ground_fact", "fact"):
                        entity_match = re.search(r"\w+\(([a-z][a-z0-9_]*)\)", clause.fol)
                        if entity_match:
                            ground_entity = entity_match.group(1)
                            break

                llm_yes_options = []
                for label, text in options:
                    if label == unknown_label:
                        continue
                    option_query = _option_yesno_query(text)
                    opt_subject, opt_predicate = _parse_query_entities(option_query)

                    opt_tokens = set(opt_predicate.lower().replace("_", " ").split())
                    best_pred = None
                    best_score = 0
                    for lp in llm_predicates:
                        lp_tokens = set(lp.lower().replace("_", " ").split())
                        overlap = len(opt_tokens & lp_tokens)
                        if overlap > best_score:
                            best_score = overlap
                            best_pred = lp

                    entity_for_query = ground_entity or opt_subject.replace(' ', '_')
                    if best_pred and best_score >= 1:
                        query_fol = f"{best_pred}({entity_for_query})"
                    else:
                        query_fol = f"{opt_predicate.replace(' ', '_')}({entity_for_query})"

                    query_negated_fol = f"Not({query_fol})"
                    option_translation = FOLTranslation(
                        clauses=translation.clauses,
                        query_fol=query_fol,
                        query_negated_fol=query_negated_fol,
                        query_subject=opt_subject,
                        query_predicate=opt_predicate,
                        translation_confidence=translation.translation_confidence,
                    )

                    try:
                        built = _build_z3_theory(option_translation)
                        if built is not None:
                            solver, query_expr, _, used_ids = built
                            solver.push()
                            solver.add(z3_mod.Not(query_expr))
                            check_yes = solver.check()
                            solver.pop()
                            if check_yes == z3_mod.unsat:
                                llm_yes_options.append((label, text, used_ids))
                    except Exception:
                        pass

                if len(llm_yes_options) == 1:
                    label, text, used_ids = llm_yes_options[0]
                    support = [p for p in premises if p.id in used_ids]
                    return label, support, "LLM-assisted FOL/Z3 entailment"

    # Default fallback to unknown label if present
    if unknown_label is not None:
        support = select_premises(question, premises)
        return unknown_label, support, "Z3 undetermined, choosing unknown option"

    return None

def _all_mcq_options(
    question: str | None,
    choices: list[str] | None,
) -> list[tuple[str, str]]:
    """Return the full ordered set of MCQ ``(label, text)`` pairs.

    Combines the inline-labeled options from the question with the ``choices``
    list (deduplicated by label, inline labels win). Empty if neither source
    contributes any option. Pure structural extraction.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    options = _labeled_options(question or "") if question else {}
    for label, text in options.items():
        text = (text or "").strip()
        if label and text and label not in seen:
            pairs.append((label, text))
            seen.add(label)
    if choices:
        for idx, text in enumerate(choices):
            label = chr(ord("A") + idx)
            if label in seen:
                continue
            text = (text or "").strip()
            if text:
                pairs.append((label, text))
                seen.add(label)
    return pairs

def _option_yesno_query(option_text: str) -> str:
    """Reformulate an option statement as a yes/no entailment question.

    The resulting question is what the deterministic FOL compiler is asked to
    entail: "Does it follow that <option>?". Reusing
    ``_option_text_to_question`` would yield a wh-style direct question, which
    the compiler interprets as a ground/threaded query about the named entity
    rather than a check that the OPTION'S CLAIM is entailed by the premises.
    The "Does it follow that ..." wrapper makes the goal an entailment check
    over the option's statement directly.
    """
    text = (option_text or "").strip().rstrip(".")
    if not text:
        return text
    if text.endswith("?"):
        return text
    return f"Does it follow that {text}?"

def _resolve_chosen_option_text(
    answer: str | None,
    question: str | None,
    choices: list[str] | None,
) -> str | None:
    """Look up the literal option text for a labeled MCQ answer.

    Tries the inline-labeled options parsed from the question first, then falls
    back to the ``choices`` list (A=index 0, B=1, ...). Returns ``None`` when
    the answer is not a single A-Z label or no option text can be located.
    Pure structural lookup — never inspects question text for keywords.
    """
    if not answer:
        return None
    label = answer.strip().upper()
    if len(label) != 1 or not label.isalpha():
        return None
    options = _labeled_options(question or "") if question else {}
    if options and label in options:
        return str(options[label]).strip() or None
    if choices:
        idx = ord(label) - ord("A")
        if 0 <= idx < len(choices):
            text = str(choices[idx] or "").strip()
            return text or None
    return None

def _mcq_support_is_deterministic(reason: str, support: list[Premise]) -> bool:
    """Check if the matched logic solver reason indicates deterministic support."""
    low = reason.lower().strip()
    if not support:
        return False
    disallowed = {
        "overlap",
        "heuristic",
        "selected premises",
        "no deterministic entailment rule matched",
        "insufficient",
        "unknown",
        "not enough",
        "ambiguous",
    }
    if any(marker in low for marker in disallowed):
        return False
    allowed = (
        "direct fact",
        "universal syllogism",
        "mcq implication chain",
        "mcq contrapositive",
        "modus ponens",
        "modus tollens",
        "subject-threaded",
        "policy blocker",
        "policy rule",
        "academic policy rule",
        "conditional failure-status closure",
        "triggered negative status rule",
        "forward chaining",
        "negated antecedent",
        "disjunctive syllogism",
        "disjunctive antecedent",
    )
    return any(marker in low for marker in allowed)

def _det_fol_entails_option(
    option_text: str,
    premises: list[Premise],
    premises_fol: list[str] | None,
) -> tuple[str, list[str]] | None:
    """Run the deterministic FOL->Z3 compiler against an MCQ option.

    Returns ``("yes", used_premise_ids)`` if Z3 entails the option's statement,
    ``("no", used)`` if Z3 entails its negation, or ``None`` when the compiler
    abstains / errors. Operates only on premise structure and option text — no
    question-text keyword matching, no Hardcoded_Override (AGENTS.md §20).
    """
    if not option_text or not (premises or premises_fol):
        return None
    # Soundness guard mirrors ``_deterministic_fol_signal``: if the ground premises
    # are mutually contradictory the atom-based compiler would derive a spurious
    # verdict by ex-falso. Abstain so the gate sees "no signal" rather than an
    # unsound one.
    if _premises_contain_contradiction(premises):
        return None
    try:
        from app.logic.dsl_compiler import solve_deterministic_fol
    except Exception:
        return None
    payload = [{"id": premise.id, "text": premise.text} for premise in premises]
    query = _option_yesno_query(option_text)
    try:
        result = solve_deterministic_fol(query, payload, list(premises_fol or []))
    except Exception:
        return None
    if result.error is not None or result.answer not in {"yes", "no"}:
        return None
    if result.z3_status not in {"entailed", "contradicted"}:
        return None
    valid_ids = {p.id for p in premises}
    used = [pid for pid in result.used_premise_ids if pid in valid_ids]
    return result.answer, used

def _mcq_independent_backend_agrees(
    question: str | None,
    normalized: list[Premise] | None,
    premises_fol: list[str] | None,
    chosen_option_text: str | None,
    chosen_option_label: str | None,
    choices: list[str] | None,
    llm_client: object | None = None,
    call_budget: object | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Verify an MCQ answer against the deterministic FOL→Z3 veto (Req 3.3, 3.4).

    Closes Cluster A's verifier_accepted_wrong cases using a FOL-ONLY VETO:
    the deterministic FOL→Z3 compiler is the sole independent backend signal.
    When it produces a decisive verdict, it can confirm or veto the MCQ scorer's
    pick. When it abstains (cannot compile the premises/options to Z3), the gate
    accepts the MCQ scorer's pick — a single signal with no contradicting signal
    is accepted (design.md §Verifier rule b).

    Procedure (purely structural; no question-text keyword matching):

    1. For every MCQ option, ask the deterministic FOL→Z3 compiler whether the
       option's *content* is entailed (re-formulated as ``"Does it follow that
       <option>?"``). The compiler runs on the same premise structure used by
       the fast path but uses Z3 as the entailment authority.
    2. If exactly ONE option is entailed and it matches chosen → agrees=True,
       source ``deterministic_fol_z3`` (two signals agree).
    3. If exactly ONE option is entailed and it's a DIFFERENT option →
       agrees=False, ``independent_backend_picks_different_option`` (backend
       disagreement → abstain).
    4. If TWO OR MORE options are entailed → agrees=False,
       ``independent_backend_entails_multiple_options`` (can't break tie →
       abstain).
    5. If FOL ABSTAINS for ALL options → agrees=True,
       ``fol_abstains_no_contradiction`` (single signal, no contradicting
       signal → accept the MCQ scorer's pick).

    Returns ``(agrees, reason, evidence)`` where ``evidence`` carries the raw
    per-option verdicts and the agreeing-signal name for the verifier trace.
    """
    evidence: dict[str, Any] = {
        "fol_per_option": {},
        "agreeing_source": None,
    }
    from app.pipeline_config import load_pipeline_config
    config = load_pipeline_config()
    if not config.enable_z3_sidecar:
        evidence["agreeing_source"] = "mcq_scorer_no_fol_contradiction"
        return True, "fol_abstains_no_contradiction", evidence

    # Check if the domain is allowed for Z3 sidecar.
    domain_allowed = False
    for domain in config.z3_allowed_domains:
        if domain == "academic_policy":
            from app.logic.policy_patterns import is_academic_policy_text
            if is_academic_policy_text(question or "", [p.text for p in (normalized or [])]):
                domain_allowed = True
                break
        elif domain == "public_logic_sample":
            if _is_public_logic_sample_text(question or "", normalized or []):
                domain_allowed = True
                break

    if not domain_allowed:
        evidence["agreeing_source"] = "mcq_scorer_no_fol_contradiction"
        return True, "fol_abstains_no_contradiction", evidence

    if not chosen_option_text or not chosen_option_label:

        # Nothing to check; treat as agreement-by-default to preserve current
        # behavior on non-MCQ paths. The wiring in ``_postprocess_solution``
        # already restricts the gate to MCQ shapes, so this guard is defensive.
        return True, "no_chosen_option_text", evidence

    options = _all_mcq_options(question, choices)
    if len(options) < 2:
        return True, "single_option_not_mcq", evidence

    chosen_label = chosen_option_label.strip().upper()

    # Deterministic FOL→Z3 entailment per option (the sole independent signal).
    # IMPORTANT: Do NOT pass premises_fol here. The gate must use only the
    # NL-derived theory (from normalized premises) for its independent check.
    # The dataset's clean FOL (premises_fol) is the same signal the main solver
    # already used — passing it here would make the gate always agree with the
    # solver (not independent), and the LLM veto branch would never fire.
    fol_yes_labels: list[str] = []
    fol_per_option: dict[str, str] = {}
    det_fol_entails_option_fn = _get_det_fol_entails_option()
    for label, text in options:
        verdict = det_fol_entails_option_fn(text, list(normalized or []), None)
        if verdict is None:
            fol_per_option[label] = "abstain"
            continue
        ans, _ = verdict
        fol_per_option[label] = ans
        if ans == "yes":
            fol_yes_labels.append(label)
    evidence["fol_per_option"] = fol_per_option

    if fol_yes_labels:
        # Backend produced at least one decisive signal.
        if len(fol_yes_labels) == 1 and fol_yes_labels[0] == chosen_label:
            evidence["agreeing_source"] = "deterministic_fol_z3"
            return True, "deterministic_fol_z3_entails_chosen_option", evidence
        if len(fol_yes_labels) == 1 and fol_yes_labels[0] != chosen_label:
            evidence["disagreeing_label"] = fol_yes_labels[0]
            return False, "independent_backend_picks_different_option", evidence
        # Two or more options entail; cannot break tie soundly.
        evidence["disagreeing_labels"] = sorted(fol_yes_labels)
        return False, "independent_backend_entails_multiple_options", evidence

    # FOL compiler abstained for ALL options — no independent signal contradicts
    # the MCQ scorer's pick. Before accepting, try LLM veto if available (Cách 2).
    if llm_client is not None and call_budget is not None and call_budget.can_spend():
        llm_veto_result = _try_llm_veto(
            question, normalized, premises_fol, chosen_label, options, llm_client, call_budget
        )
        if llm_veto_result is not None:
            agrees, reason, llm_evidence = llm_veto_result
            evidence.update(llm_evidence)
            if not agrees:
                return False, reason, evidence
    # If no LLM veto fired (or no LLM available), accept as before (rule b).
    evidence["agreeing_source"] = "mcq_scorer_no_fol_contradiction"
    return True, "fol_abstains_no_contradiction", evidence

def _try_llm_veto(
    question: str | None,
    normalized: list[Premise] | None,
    premises_fol: list[str] | None,
    chosen_label: str,
    options: list[tuple[str, str]],
    llm_client: object,
    call_budget: object,
) -> tuple[bool, str, dict[str, Any]] | None:
    """Try LLM FOL translation as a VETO-ONLY signal (Cách 2).

    Called when the deterministic FOL compiler abstains for ALL MCQ options.
    Uses the existing LLM FOL translator (``fol_z3_pipeline.py``) with k=1 to
    translate premises to FOL (1 LLM call), then checks per-option entailment
    via Z3 (pure backend, no additional LLM calls).

    VETO-ONLY rule:
      - LLM-Z3 entails exactly ONE option AND it's DIFFERENT from chosen → VETO.
      - LLM-Z3 entails the SAME option as chosen → no signal (LLM agreement is
        not independent evidence; both can be wrong the same way).
      - LLM-Z3 entails multiple options or abstains → no signal.

    Returns ``(False, reason, evidence)`` if LLM-Z3 vetoes the chosen option.
    Returns ``None`` if no veto (LLM agrees, abstains, or entails multiple).
    Costs exactly 1 LLM call (the premise translation); Z3 checks are free.
    """
    if not normalized and not premises_fol:
        return None

    try:
        from app.logic.fol_z3_pipeline import (
            fol_translate,
            _build_z3_theory,
            _parse_query_entities,
            FOLTranslation,
        )
        import z3 as z3_mod
    except Exception:
        return None

    # Build premises list for the LLM translator.
    premise_dicts = [{"id": p.id, "text": p.text} for p in (normalized or [])]
    if not premise_dicts:
        return None

    # Use the first option as the representative query for the single LLM call.
    # This gets us premise FOL clauses that we reuse for all options.
    first_option_query = _option_yesno_query(options[0][1]) if options else ""
    if not first_option_query:
        return None

    subject, predicate = _parse_query_entities(first_option_query)

    # 1 LLM call: translate premises + representative query to FOL.
    # The BudgetGatedClient automatically spends from the budget.
    try:
        translation = fol_translate(
            premises=premise_dicts,
            query=first_option_query,
            query_subject=subject,
            query_predicate=predicate,
            llm_client=llm_client,
        )
    except Exception as exc:
        return None

    if not translation.success or not translation.clauses:
        return None

    # For each option, check Z3 entailment using the LLM-translated premise
    # clauses. We need to match option text to the LLM's predicate names.
    # The LLM chose specific predicate names in its translation; we must use
    # those same names in the query, not heuristically-derived ones.
    #
    # Strategy: extract all predicate names from the LLM's clauses, then for
    # each option, find the best-matching predicate via token overlap.
    llm_predicates = set()
    for clause in translation.clauses:
        if clause.fol:
            # Extract predicate names from FOL strings like "eligible_for_international_program(x)"
            import re as _re
            for pred_match in _re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", clause.fol):
                pred_name = pred_match.group(1)
                if pred_name not in {"ForAll", "Exists", "Implies", "And", "Or", "Not"}:
                    llm_predicates.add(pred_name)

    # Extract the ground entity name from LLM's fact clauses (e.g. "student(sarah)" → "sarah")
    # This resolves pronoun issues ("she" → "sarah") in option queries.
    ground_entity = None
    for clause in translation.clauses:
        if clause.fol and clause.clause_type in ("ground_fact", "fact"):
            import re as _re2
            entity_match = _re2.search(r"\w+\(([a-z][a-z0-9_]*)\)", clause.fol)
            if entity_match:
                ground_entity = entity_match.group(1)
                break
    if not ground_entity:
        # Fallback: extract from any clause with a constant (non-variable) argument
        for clause in translation.clauses:
            if clause.fol:
                import re as _re2
                entity_match = _re2.search(r"\(([a-z][a-z0-9_]*)\)", clause.fol)
                if entity_match and entity_match.group(1) not in {"x", "y", "z"}:
                    ground_entity = entity_match.group(1)
                    break

    llm_yes_labels: list[str] = []
    llm_per_option: dict[str, str] = {}

    for label, text in options:
        option_query = _option_yesno_query(text)
        if not option_query:
            llm_per_option[label] = "abstain"
            continue

        opt_subject, opt_predicate = _parse_query_entities(option_query)

        # Find the best matching LLM predicate for this option via token overlap.
        opt_tokens = set(opt_predicate.lower().replace("_", " ").split())
        best_pred = None
        best_score = 0
        for lp in llm_predicates:
            lp_tokens = set(lp.lower().replace("_", " ").split())
            overlap = len(opt_tokens & lp_tokens)
            if overlap > best_score:
                best_score = overlap
                best_pred = lp
        
        # Use the matched LLM predicate if overlap is significant (>=1 token)
        # and use the ground entity from LLM clauses instead of parsed subject
        entity_for_query = ground_entity or opt_subject.replace(' ', '_')
        if best_pred and best_score >= 1:
            query_fol = f"{best_pred}({entity_for_query})"
        else:
            query_fol = f"{opt_predicate.replace(' ', '_')}({entity_for_query})"
        query_negated_fol = f"Not({query_fol})"

        option_translation = FOLTranslation(
            clauses=translation.clauses,
            query_fol=query_fol,
            query_negated_fol=query_negated_fol,
            query_subject=opt_subject,
            query_predicate=opt_predicate,
            translation_confidence=translation.translation_confidence,
        )

        # Build Z3 theory and check entailment (pure backend, no LLM call).
        try:
            built = _build_z3_theory(option_translation)
        except Exception:
            llm_per_option[label] = "error"
            continue

        if built is None:
            llm_per_option[label] = "abstain"
            continue

        solver, query_expr, _, used_ids = built

        # Entailment check: theory ⊨ query ↔ theory ∧ ¬query is UNSAT
        solver.push()
        solver.add(z3_mod.Not(query_expr))
        check_yes = solver.check()
        solver.pop()

        if check_yes == z3_mod.unsat:
            llm_yes_labels.append(label)
            llm_per_option[label] = "yes"
        else:
            # Check contradiction: theory ⊨ ¬query ↔ theory ∧ query is UNSAT
            solver.push()
            solver.add(query_expr)
            check_no = solver.check()
            solver.pop()
            if check_no == z3_mod.unsat:
                llm_per_option[label] = "no"
            else:
                llm_per_option[label] = "unknown"

    evidence: dict[str, Any] = {
        "llm_fol_per_option": llm_per_option,
        "llm_fol_yes_labels": llm_yes_labels,
    }

    # Apply VETO-ONLY rule.
    if len(llm_yes_labels) == 1 and llm_yes_labels[0] != chosen_label:
        # LLM-Z3 entails a DIFFERENT option → VETO.
        evidence["llm_veto_label"] = llm_yes_labels[0]
        return False, "llm_z3_veto_different_option", evidence

    # LLM agrees with chosen, entails multiple, or abstains → no veto signal.
    return None

def _negative_clauses_equivalent(left: str, right: str) -> bool:
    """Check if two negated clauses are logically equivalent based on core tokens."""
    stop = {"not", "no", "never", "does", "do", "did", "is", "are", "has", "have", "lacks", "lack", "a", "an", "the", "it", "they", "file", "student", "shape", "object"}
    left_tokens = _clean_content_tokens(left) - stop
    right_tokens = _clean_content_tokens(right) - stop
    return bool(left_tokens and right_tokens and (left_tokens <= right_tokens or right_tokens <= left_tokens))

def _status_tokens(text: str) -> set[str]:
    """Extract stemmed status-related keywords from text."""
    status_words = {
        "eligible",
        "ineligible",
        "qualify",
        "qualified",
        "allow",
        "allowed",
        "permit",
        "permitted",
        "register",
        "registered",
        "enroll",
        "enrolled",
    }
    return {_stem(token) for token in re.findall(r"[a-z0-9]+", _norm(text)) if _stem(token) in status_words}

def _status_tokens_compatible(option_tokens: set[str], consequent_tokens: set[str]) -> bool:
    """Check if status tokens from option and rule consequent are compatible."""
    if option_tokens & consequent_tokens:
        return True
    eligibility_tokens = {"eligible", "ineligible", "qualify", "qualified"}
    permission_tokens = {"allow", "allowed", "permit", "permitted", "register", "registered", "enroll", "enrolled"}
    return bool(option_tokens & eligibility_tokens and consequent_tokens & permission_tokens)

def _triggered_negative_status_support(option_text: str, premises: list[Premise]) -> tuple[list[Premise], str] | None:
    """Support negative MCQ status options from triggered prohibition rules.

    This is not an elimination shortcut: it requires a rule with a negated
    consequent and a fact that satisfies that rule's antecedent.
    """

    if not _is_negated(option_text):
        return None
    option_tokens = _status_tokens(option_text)
    if not option_tokens:
        return None
    for rule_premise in premises:
        rule = _match_rule(rule_premise.text)
        if not rule:
            continue
        antecedent, consequent = rule
        if not _is_negated(consequent):
            continue
        consequent_tokens = _status_tokens(consequent)
        if not _status_tokens_compatible(option_tokens, consequent_tokens):
            continue
        for fact_premise in premises:
            if fact_premise == rule_premise:
                continue
            fact = _fact_subject_kind(fact_premise.text)
            if not fact:
                continue
            _fact_subject, fact_kind = fact
            if _antecedent_triggered(antecedent, fact_kind, fact_premise.text):
                return [rule_premise, fact_premise], "triggered negative status rule"
    return None