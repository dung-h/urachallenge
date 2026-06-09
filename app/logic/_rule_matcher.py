"""Parser and matcher for logical rules and facts extracted from premise texts.

This module provides functions to parse natural language rules (universal, conditional,
and negative), match them against facts, and construct inference edges for chaining.
"""

from __future__ import annotations
import re
from typing import Any
from app.logic.premise_selector import Premise
from app.logic._proof_classes import Fact, Rule

from app.logic._text_primitives import (
    _norm, _singular, _stem, _strip_articles, _content_tokens,
    _predicate_tokens, _predicate_matches, _specific_tokens,
    _terms_overlap, _tokens_cover, _clean_tokens_cover, _clean_content_tokens,
    _conditional_parts, _is_negated, _negates_condition, _predicate_supported,
    _contains_entity, _is_probabilistic_rule, _split_subject_predicate,
    IGNORABLE_PREDICATE_WORDS, _NEGATION_PATTERN,
)

def _match_all_rule(premise: str) -> tuple[str, str] | None:
    """Match a universal-affirmative rule ("All X are Y") and return (X, Y)."""
    # Defensive: a universal that contains an "unless" / "except" exception
    # clause MUST be parsed by the conditional path (``_match_if_rule`` /
    # ``_conditional_parts``), which negates the exception clause and yields
    # a sound antecedent. The shallow "all X are Y" regexes below are greedy
    # over copulas and would match the COPULA INSIDE the unless-clause
    # ("they are penguins"), e.g. "All birds can fly unless they are penguins"
    # would parse to ("birds can fly unless they", "penguins"), wiping out the
    # exception. Skip here so the if-rule branch handles it. Generalizes over
    # every "All X ... unless Y." formulation, not a specific phrasing.
    norm_text = _norm(premise)
    if re.search(r"\bunless\b", norm_text):
        return None
    if re.search(r"\bexcept\s+(?:if|when)\b", norm_text):
        return None
    match = re.match(r"(?:all|every|each) (.+?) who (.+?) receive (.+?)[.]?$", norm_text)
    if match:
        return _singular(match.group(2).strip()), "receive " + _singular(match.group(3).strip())
    match = re.match(r"(?:all|every|each) (.+?) (?:are|is) (.+?)[.]?$", _norm(premise))
    if match:
        # Strip ONLY a leading indefinite/definite article from each side so
        # universal-affirmative rules with the singular copula ("Every A is a
        # B.") parse to the same bare-class tokens as the plural form
        # ("All As are Bs."). A blanket _strip_articles would erase single-
        # letter class names ("A" -> "" after normalization), so we use a
        # leading-only strip pattern that targets the indefinite article slot.
        # Component-level fix in the matcher; generalizes to any "Every X is a
        # Y" formulation regardless of whether X/Y are words or letters.
        def _strip_leading_article(s: str) -> str:
            return re.sub(r"^(?:a|an|the)\s+", "", s.strip(), flags=re.I)
        left = _singular(_strip_leading_article(match.group(1).strip()))
        right = _singular(_strip_leading_article(match.group(2).strip()))
        return left, right
    match = re.match(r"(?:all|every|each) (.+?) have (.+?)[.]?$", _norm(premise))
    if match:
        return _singular(match.group(1).strip()), "have " + _singular(match.group(2).strip())
    match = re.match(r"(?:all|every|each) (.+?) need (.+?)[.]?$", _norm(premise))
    if match:
        return _singular(match.group(1).strip()), "need " + _singular(match.group(2).strip())
    match = re.match(r"(?:all|every|each) (.+?) require (.+?)[.]?$", _norm(premise))
    if match:
        return _singular(match.group(1).strip()), "require " + _singular(match.group(2).strip())
    match = re.match(r"(?:all|every|each) (.+?) can (.+?)[.]?$", _norm(premise))
    if match:
        return _singular(match.group(1).strip()), "can " + _singular(match.group(2).strip())
    match = re.match(r"(?:all|every|each) (.+?) receive (.+?)[.]?$", _norm(premise))
    if match:
        return _singular(match.group(1).strip()), "receive " + _singular(match.group(2).strip())
    match = re.match(
        r"(?:all|every|each) (.+?)\s+((?:turn|turns|pass|passes|fail|fails|qualify|qualifies|"
        r"receive|receives|get|gets|need|needs|require|requires|have|has|submit|submits|"
        r"attend|attends|study|studies|complete|completes|meet|meets|can|must|will)\b.+)[.]?$",
        _norm(premise),
    )
    if match:
        return _singular(match.group(1).strip()), _singular(match.group(2).strip())
    return None

def _match_no_rule(premise: str) -> tuple[str, str] | None:
    """Match a universal-negative rule "No X is/are Y" -> (X, Y).

    Supports both the plural ("No corporations are natural persons.") and the
    singular ("No corporation is a natural person.") surface forms. The
    singular form occurs naturally for non-pluralizable subjects and for
    formal definitional language (e.g. "No corporation is a natural
    person.", "No square is a circle."). Both compile to the same universal-
    negative ``ForAll x. Implies(X(x), Not(Y(x)))`` rule, so the surface
    distinction must NOT change the parsed structure (component-level fix:
    one rule recognized regardless of copula number; never a per-question
    override).
    """
    norm = _norm(premise)
    # Plural copula: "No As are Bs."
    match = re.match(r"no (.+?) are (.+?)[.]?$", norm)
    if match:
        return _singular(match.group(1).strip()), _singular(match.group(2).strip())
    # Singular copula: "No A is (a/an) B." — the optional indefinite article on
    # the predicate is stripped via _singular's existing handling not being
    # required here; remove a leading article in the predicate clause so the
    # parsed B token is consistent across forms.
    match = re.match(r"no (.+?) is (.+?)[.]?$", norm)
    if match:
        subject = _singular(match.group(1).strip())
        predicate = match.group(2).strip()
        predicate = re.sub(r"^(?:a|an|the)\s+", "", predicate)
        return subject, _singular(predicate)
    return None

def _match_if_rule(premise: str) -> tuple[str, str] | None:
    """Match a conditional rule ("If X then Y") and return (X, Y)."""
    return _conditional_parts(premise)

def _match_rule(premise: str) -> tuple[str, str] | None:
    """Identify and parse any rule pattern (if, all, no, or prohibition) from text."""
    if_res = _match_if_rule(premise)
    if if_res:
        return if_res
    all_res = _match_all_rule(premise)
    if all_res:
        return all_res
    no_res = _match_no_rule(premise)
    if no_res:
        return no_res

    low = _norm(premise)

    prohibition_match = re.match(
        r"^(?:students|student|faculty\s+members|faculty\s+member|lecturers|lecturer|anyone|drivers|driver|people|person|all\s+students|all\s+faculty\s+members|all\s+lecturers|lecturers\s+with|faculty\s+members\s+with)\s+(?:who|with|that)\s+(.+?)\s+(may\s+not|cannot|can't)\s+(.+)$",
        low,
    )
    if prohibition_match:
        condition = prohibition_match.group(1).strip()
        modal = prohibition_match.group(2).strip()
        action = prohibition_match.group(3).strip()
        return condition, f"{modal} {action}".strip()
    
    # Generic natural-language conditional/universal rule patterns
    rule_subject_patterns = [
        # "students who completed A are qualified for B"
        # "faculty members with at least 5 publications can serve on C"
        r"^(?:students|student|faculty\s+members|faculty\s+member|lecturers|lecturer|anyone|drivers|driver|people|person|all\s+students|all\s+faculty\s+members|all\s+lecturers|lecturers\s+with|faculty\s+members\s+with)\s+(?:who|with|that)\s+(.+?)\s+(?:are|is|can|may\s+not|cannot|can't|qualify\s+for|qualifies\s+for|receive|receives|must\s+be|need|needs|require|requires|holds|grant|grants)\s+(.+)$",
        # "completing 500 clinical hours grants Advanced Practice"
        # "Enrollment in Course C makes a student eligible for ..."
        # Keep the nominalized ACTION word in the antecedent (don't strip
        # "enrollment in"/"completing"), otherwise the antecedent can reduce
        # to a token-empty fragment ("course c" -> {}) that matches nothing
        # and blocks the chain. We map the nominal action to its verb stem so
        # it aligns with a derived fact like "can enroll in Course C".
        r"^(completing|achieving|having|maintaining|prioritizing|using|understanding|enrollment\s+in|enrolling\s+in|passing|attending|submitting|finishing)\s+(.+?)\s+(?:grants|grant|leads\s+to|lead\s+to|makes|make|results\s+in|enhances|enhance|provides|provide|qualifies|qualify)\s+(.+)$",
        # "if a student is registered, they will achieve success" (sometimes has commas/pronouns)
        r"^if\s+(.+?),?\s+(?:they|it|she|he)\s+(?:will|can|are|is|must|receive|receives|qualify|qualifies|earn|earns|succeed|succeeds)\s+(.+)$"
    ]
    for pattern in rule_subject_patterns:
        match = re.match(pattern, low)
        if match:
            groups = match.groups()
            if len(groups) == 3:
                # Nominalized causative: (action, object, consequent).
                # Build the antecedent as "<verb-stem> <object>" so it carries
                # content tokens and aligns with a derived "<verb> <object>"
                # fact (e.g. action="enrollment in", object="course c" ->
                # antecedent "enroll in course c").
                action, obj, conc = groups
                _ACTION_VERB = {
                    "enrollment in": "enroll in", "enrolling in": "enroll in",
                    "completing": "complete", "achieving": "achieve",
                    "having": "have", "maintaining": "maintain",
                    "prioritizing": "prioritize", "using": "use",
                    "understanding": "understand", "passing": "pass",
                    "attending": "attend", "submitting": "submit",
                    "finishing": "finish",
                }
                verb = _ACTION_VERB.get(action.strip().lower(), action.strip())
                antecedent = f"{verb} {obj.strip()}".strip()
                return antecedent, conc.strip()
            return match.group(1).strip(), match.group(2).strip()
            
    return None

def _class_matches(rule_class: str, fact_class: str) -> bool:
    """Check if the rule's subject/class matches the fact's class."""
    if _tokens_cover(rule_class, fact_class):
        return True
    generic_class_tokens = {"file", "form", "object", "item", "thing"}
    required_tokens = _content_tokens(rule_class) - generic_class_tokens
    actual_tokens = _content_tokens(fact_class)
    if bool(required_tokens) and required_tokens <= actual_tokens:
        return True
    # Single-letter class fallback: when the rule's class is a single short
    # identifier (e.g. "a", "b", "x"), the content-token filter strips it
    # because of the >=3-char rule. Match by direct word presence in the
    # fact's content tokens. Generic structural fix — generalizes to every
    # short-symbol classification problem (logic puzzles, set-membership).
    rc = (rule_class or "").strip().lower()
    if rc and len(rc) <= 2 and rc.isalnum():
        import re as _re
        # Tokenize fact_class and check if any whole-word matches.
        fact_words = _re.findall(r"\b[a-z0-9]+\b", (fact_class or "").lower())
        if rc in fact_words:
            return True
    return False

def _antecedent_triggered(antecedent: str, fact_kind: str, fact_text: str) -> bool:
    """Check if the fact text triggers or supports the rule antecedent."""
    # Comparative modifier guardrail:
    comparatives = {"higher", "above", "more", "greater", "exceeds", "longer", "older", "larger", "less", "below", "fewer", "shorter", "younger", "smaller"}
    ant_low = antecedent.lower()
    fact_low = fact_text.lower()
    fact_comps = {w for w in comparatives if w in fact_low}
    ant_comps = {w for w in comparatives if w in ant_low}
    if fact_comps and not ant_comps:
        return False

    # Include pronouns so that consequents like "it is safe" match "is safe"
    generic_actor_tokens = {
        "student", "person", "learner", "candidate", "shape", "object", "item", "thing",
        "is", "are", "has", "have", "it", "they", "he", "she", "the",
    }
    antecedent_tokens = _clean_content_tokens(antecedent) - generic_actor_tokens
    fact_tokens = _clean_content_tokens(fact_kind)
    return bool(antecedent_tokens) and (antecedent_tokens <= fact_tokens or antecedent in _norm(fact_text))

def _implies(actual: str, expected: str, context_subject: set[str] | None = None) -> bool:
    """Determine if a source clause implies a target clause, considering context."""
    actual_low = _norm(actual)
    expected_low = _norm(expected)
    act_neg = _is_negated(actual_low)
    exp_neg = _is_negated(expected_low)
    if act_neg != exp_neg:
        return False
    if expected_low in actual_low and not act_neg:
        return True
    if actual_low in expected_low and act_neg:
        return True
    expected_tokens = _clean_content_tokens(expected_low)
    actual_tokens = _clean_content_tokens(actual_low)
    if not expected_tokens or not actual_tokens:
        return False
        
    act_sub, act_pred = _split_subject_predicate(actual_low)
    exp_sub, exp_pred = _split_subject_predicate(expected_low)
    
    pronouns = {"it", "he", "she", "they", "them", "their", "its", "him", "her"}
    
    if context_subject:
        if not act_sub or act_sub <= pronouns:
            act_sub = context_subject
        if not exp_sub or exp_sub <= pronouns:
            exp_sub = context_subject

    if not exp_sub or exp_sub <= pronouns:
        sub_ok = True
    else:
        sub_ok = not act_sub or act_sub <= pronouns or bool(exp_sub & act_sub)
        
    if not act_neg:
        pred_ok = exp_pred <= act_pred
    else:
        pred_ok = act_pred <= exp_pred
        
    return sub_ok and pred_ok

def _fact_implies_target(fact_tokens: set[str], fact_positive: bool, target_tokens: set[str], q_negative: bool) -> bool:
    """Check if the fact's tokens and polarity imply the target's tokens and polarity."""
    if not fact_tokens or not target_tokens:
        return False
    fact_neg = not fact_positive
    if fact_neg != q_negative:
        return False
    if not fact_neg:
        return target_tokens <= fact_tokens
    else:
        return fact_tokens <= target_tokens

def _negate_clause(clause: str) -> str:
    """Return a surface-level negated version of the clause."""
    low = _norm(clause)
    if low.startswith("not "):
        return low[4:].strip()
    if low.startswith("no "):
        return low[3:].strip()
    if " not " in low:
        return low.replace(" not ", " ", 1).strip()
    return f"not {low}".strip()

def _implication_edges(premises: list[Premise]) -> list[tuple[str, str, Premise]]:
    """Build directed implication edges (antecedent -> consequent) from premises."""
    edges: list[tuple[str, str, Premise]] = []
    for premise in premises:
        if_rule = _match_if_rule(premise.text)
        if if_rule:
            antecedent, consequent = if_rule
            low_text = premise.text.lower()
            idx = low_text.find(antecedent.lower())
            orig_ant = premise.text[idx : idx + len(antecedent)] if idx >= 0 else antecedent
            idx_cons = low_text.find(consequent.lower())
            orig_cons = premise.text[idx_cons : idx_cons + len(consequent)] if idx_cons >= 0 else consequent
            
            edges.append((orig_ant, orig_cons, premise))
            if not _is_negated(orig_cons):
                edges.append((_negate_clause(orig_cons), _negate_clause(orig_ant), premise))
            continue
        all_rule = _match_all_rule(premise.text)
        if all_rule:
            antecedent, consequent = all_rule
            low_text = premise.text.lower()
            idx = low_text.find(antecedent.lower())
            orig_ant = premise.text[idx : idx + len(antecedent)] if idx >= 0 else antecedent
            idx_cons = low_text.find(consequent.lower())
            orig_cons = premise.text[idx_cons : idx_cons + len(consequent)] if idx_cons >= 0 else consequent
            
            edges.append((orig_ant, orig_cons, premise))
            if not _is_negated(orig_cons):
                edges.append((_negate_clause(orig_cons), _negate_clause(orig_ant), premise))
            continue
        no_rule = _match_no_rule(premise.text)
        if no_rule:
            left, right = no_rule
            low_text = premise.text.lower()
            idx = low_text.find(left.lower())
            orig_left = premise.text[idx : idx + len(left)] if idx >= 0 else left
            idx_right = low_text.find(right.lower())
            orig_right = premise.text[idx_right : idx_right + len(right)] if idx_right >= 0 else right
            
            edges.append((orig_left, _negate_clause(orig_right), premise))
            edges.append((orig_right, _negate_clause(orig_left), premise))
    return edges

def _support_path(start: str, target: str, edges: list[tuple[str, str, Premise]]) -> list[Premise] | None:
    """Find a chain of premises supporting the target statement starting from start."""
    start_norm = _norm(start)
    target_norm = _norm(target)
    start_sub, _ = _split_subject_predicate(start_norm)
    queue: list[tuple[str, list[Premise]]] = [(start_norm, [])]
    seen: set[str] = set()
    while queue:
        current, support = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        if _implies(current, target_norm, context_subject=start_sub):
            return list(dict.fromkeys(support))
        for source, destination, premise in edges:
            if _implies(current, source, context_subject=start_sub):
                next_support = list(dict.fromkeys([*support, premise]))
                queue.append((destination, next_support))
    return None

def _is_universal_quantifier(text: str) -> bool:
    """Return True if the TEXT's SUBJECT is a universal quantifier.

    This checks grammatical structure: the universal word must appear at the
    start of the sentence (subject position), not buried in the predicate.

    Correct detections:
        'Everyone is trained.'          → True  (subject = everyone)
        'All employees must attend.'    → True  (subject = all employees)
        'Anyone who passes qualifies.'  → True  (subject = anyone)

    Correct non-detections:
        'John trains everyone.'         → False (everyone is object, not subject)
        'The rule applies to anyone.'   → False (anyone is prepositional object)
    """
    low = _norm(text)
    # Universal subject patterns: word appears at sentence start (optionally after 'if')
    universal_subject_patterns = [
        r"^(?:if\s+)?(?:every(?:one|body)?|any(?:one|body)?|all\s+\w+)\b",
        r"^(?:if\s+)?(?:no\s+\w+)\b",  # "No student" is universal negative
        r"^(?:if\s+)?(?:each\s+\w+)\b",
    ]
    return any(re.search(pat, low) for pat in universal_subject_patterns)

def _is_existential_quantifier(text: str) -> bool:
    """Return True if the text subject is an existential quantifier."""
    low = _norm(text)
    existential_subject_patterns = [
        r"^(?:there\s+exists?|there\s+is|there\s+are)\s+(?:at\s+least\s+one|some|an?|one)\b",
        r"^(?:at\s+least\s+one|some)\s+\w+",
        r"^(?:if\s+)?(?:there\s+exists?|there\s+is|there\s+are)\s+(?:at\s+least\s+one|some|an?|one)\b",
    ]
    return any(re.search(pat, low) for pat in existential_subject_patterns)

def _option_is_existential(option_text: str) -> bool:
    """True if an MCQ option makes an existentially-quantified claim.

    Matches the same surface forms recognized by the ``_postprocess_solution``
    existential safety gate so both layers agree on what counts as an
    existential option ("Some ...", "There exists ...", "At least one ...",
    "Not all ..." — the last being ∃x¬, existential in nature).
    """
    low = _norm(option_text)
    return bool(re.match(r"^(?:some|there exists?|there is|there are|at least one|not all)\b", low))

def _has_matching_existential_support(option_text: str, support: list[Premise]) -> bool:
    """True if ``support`` contains an existential premise matching the option.

    A "matching" existential premise is one that is itself an existential
    quantifier ("Some ...", "There exists ...") AND shares content tokens with
    the option, so a generic existential premise cannot license an unrelated
    existential option. This mirrors the authoritative gate in
    ``_postprocess_solution`` (Task 7.9, Req 4.5).
    """
    option_tokens = _clean_content_tokens(_norm(option_text))
    for premise in support:
        if _is_existential_quantifier(_norm(premise.text)) and (
            option_tokens & _clean_content_tokens(_norm(premise.text))
        ):
            return True
    return False

def _universal_object_rule(text: str) -> tuple[tuple[str, ...], str] | None:
    """Match universal claims about object classes and return properties/consequents."""
    low = _norm(text).rstrip(".")
    match = re.match(r"all\s+(.+?)\s+objects?\s+are\s+(.+)$", low)
    if match:
        return (_singular(_strip_articles(match.group(1))),), _singular(_strip_articles(match.group(2)))
    match = re.match(r"everything\s+that\s+is\s+(.+?)\s+is\s+(.+)$", low)
    if match:
        left = match.group(1).strip()
        requirements = tuple(_singular(_strip_articles(part.strip())) for part in re.split(r"\s+and\s+", left) if part.strip())
        consequent = _singular(_strip_articles(match.group(2).strip()))
        return requirements, consequent
    match = re.match(r"if\s+(?:an?\s+)?object\s+is\s+(.+?),?\s+then\s+it\s+(?:will\s+)?(.+)$", low)
    if not match:
        match = re.match(r"if\s+(?:an?\s+)?object\s+is\s+(.+?),\s+it\s+(?:will\s+)?(.+)$", low)
    if match:
        antecedent = _singular(_strip_articles(match.group(1).strip()))
        consequent = _singular(_strip_articles(match.group(2).strip()))
        return (antecedent,), consequent
    return None

def _object_prop(text: str) -> tuple[str, str] | None:
    """Parse object property facts (e.g. 'X is Y') and return (subject, property)."""
    low = _norm(text).rstrip(".")
    if low.startswith(("all ", "everything ", "if ", "no ")):
        return None
    match = re.match(r"(.+?)\s+is\s+(?:a |an )?(.+)$", low)
    if not match:
        return None
    subject = _strip_articles(match.group(1).strip())
    prop = _singular(_strip_articles(match.group(2).strip()))
    return subject, prop


def parse_fact(premise: Premise) -> Fact:
    """Parse a Premise object into a structured Fact.

    Args:
        premise: The Premise containing the text to be parsed.

    Returns:
        A Fact object with normalized text, content tokens, and polarity.
    """
    text = premise.text
    low = _norm(text)
    positive = not _is_negated(low)
    tokens = _clean_content_tokens(text)
    return Fact(text=text, tokens=tokens, positive=positive, premises=[premise])


def parse_rule(premise: Premise, antecedent: str, consequent: str, consequent_positive: bool = True) -> Rule:
    """Parse antecedent and consequent components into a structured Rule.

    Args:
        premise: The source Premise object.
        antecedent: The antecedent clause string.
        consequent: The consequent clause string.
        consequent_positive: The positive/negative polarity of the consequent.

    Returns:
        A structured Rule representation.
    """
    low_text = premise.text.lower()
    idx = low_text.find(antecedent.lower())
    orig_ant = premise.text[idx : idx + len(antecedent)] if idx >= 0 else antecedent
    idx_cons = low_text.find(consequent.lower())
    orig_cons = premise.text[idx_cons : idx_cons + len(consequent)] if idx_cons >= 0 else consequent

    ant_low = _norm(orig_ant)
    cons_low = _norm(orig_cons)
    antecedent_tokens = _clean_content_tokens(orig_ant)
    consequent_tokens = _clean_content_tokens(orig_cons)
    antecedent_positive = not _is_negated(ant_low)
    final_consequent_positive = consequent_positive if not _is_negated(cons_low) else not consequent_positive
    return Rule(
        premise=premise,
        antecedent_tokens=antecedent_tokens,
        consequent_tokens=consequent_tokens,
        antecedent_positive=antecedent_positive,
        consequent_positive=final_consequent_positive
    )

