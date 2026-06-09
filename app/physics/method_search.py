from __future__ import annotations

import html as html_lib
import math
import os
import re
from dataclasses import dataclass, field
from typing import Protocol
from typing import Any

import httpx
from urllib.parse import parse_qs, unquote, urljoin, urlparse

from app.physics.expression_eval import normalize_equation_expression, safe_eval_expression
from app.physics.formulas import K_COULOMB
from app.physics.parser import ParsedPhysicsProblem


@dataclass(frozen=True)
class MethodSearchObjective:
    question: str
    target_quantity: str | None
    quantity_units: list[str]
    structural_terms: list[str]
    query_plan: list[str]


@dataclass(frozen=True)
class MethodEvidenceSnippet:
    source: str
    title: str
    text: str
    url: str | None = None


@dataclass(frozen=True)
class MethodEquationProposal:
    method_id: str
    method_family: str
    expression: str
    target_unit: str
    variables: tuple[str, ...]
    assumptions: tuple[str, ...]
    evidence: MethodEvidenceSnippet
    blocked_formula_families: tuple[str, ...] = ()
    confidence: float = 0.0


@dataclass(frozen=True)
class VerifiedMethod:
    proposal: MethodEquationProposal
    variables: dict[str, float]
    value: float
    verification_notes: list[str] = field(default_factory=list)


class MethodSearchProvider(Protocol):
    def search(self, objective: MethodSearchObjective) -> list[MethodEvidenceSnippet]: ...


_STOP_TERMS = {
    "what", "from", "with", "that", "this", "then", "than", "into", "onto", "over", "under",
    "find", "used", "total", "value", "center", "carries", "applied", "question", "calculate",
    "radius", "length", "charge", "field", "electric", "its", "the", "and", "for", "axis",
    "formula", "problem", "parsedphysicsproblem", "quantity", "quantities", "variables", "raw",
}


_LOCAL_CORPUS = (
    MethodEvidenceSnippet(
        source="local_physics_method_corpus",
        title="Electric field on the axis of a uniformly charged ring",
        text=(
            "For a uniformly charged ring, the field at a point on the symmetry axis is found by symmetry "
            "and integration. The transverse components cancel and the axial component remains. "
            "E = k*q*x/(x^2 + R^2)^(3/2), where q is total charge, R is ring radius, and x is the axial distance."
        ),
    ),
    MethodEvidenceSnippet(
        source="local_physics_method_corpus",
        title="Electric field on the axis of a uniformly charged disk",
        text=(
            "For a uniformly charged disk, the field on its symmetry axis is found by summing rings. "
            "The standard result is E = (sigma / (2*eps0)) * (1 - z / sqrt(z^2 + R^2)), where sigma is surface "
            "charge density, R is disk radius, and z is the axial distance from the center."
        ),
    ),
    MethodEvidenceSnippet(
        source="local_physics_method_corpus",
        title="Electric field due to an infinite line charge",
        text=(
            "For an infinitely long line charge, Gauss' law gives E = lambda / (2*pi*eps0*r), where lambda is "
            "linear charge density, r is the radial distance from the line, and eps0 is the vacuum permittivity."
        ),
    ),
    MethodEvidenceSnippet(
        source="local_physics_method_corpus",
        title="Thin uniformly charged spherical shell field and potential",
        text=(
            "For a uniformly charged thin spherical shell, the electric field is zero for points inside the shell "
            "and outside it is equivalent to a point charge at the center. The potential is constant inside, V = kQ/R, "
            "and outside it is V = kQ/r."
        ),
    ),
    MethodEvidenceSnippet(
        source="local_physics_method_corpus",
        title="Complex resistor networks require topology analysis",
        text=(
            "Bridge, ladder, mesh, and diamond resistor networks are not generally reducible by a single "
            "series or parallel shortcut. Use nodal analysis, mesh analysis, or a fully specified topology."
        ),
    ),
)


def structural_terms(question: str) -> list[str]:
    low = question.lower().replace("ε", "epsilon")
    terms: set[str] = set()
    for match in re.finditer(r"[a-zA-Z_][a-zA-Z_/-]{2,}", low):
        token = match.group(0).strip("-_/")
        if token and token not in _STOP_TERMS and not token.isdigit():
            terms.add(token)
    for phrase in (
        "open switch", "switch is open", "uniformly charged", "uniformly distributed", "linear charge density",
        "surface charge density", "perpendicular bisector", "cross resistor", "resistor network", "network of resistors",
        "relative permittivity", "series rlc", "ac circuit", "alternating current", "turns ratio", "magnetic field",
        "magnetic flux", "symmetry axis", "capacitive reactance", "inductive reactance", "impedance",
        "charged disk", "uniformly charged disk", "infinite line charge", "line charge", "dipole moment",
        "axial line", "equatorial line", "electric dipole", "spherical shell", "thin spherical shell", "shell",
    ):
        if phrase in low:
            terms.add(phrase)
    return sorted(terms)


def _web_method_search_enabled() -> bool:
    raw = os.environ.get("URA_ENABLE_WEB_METHOD_SEARCH")
    if raw is None:
        return False
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _infer_search_target(question: str, parsed: ParsedPhysicsProblem) -> str | None:
    if parsed.target_quantity:
        return parsed.target_quantity
    low = question.lower()
    if any(term in low for term in ["capacitive reactance", "inductive reactance", "reactance of the capacitor", "reactance of the inductor"]):
        return "reactance"
    if "impedance" in low:
        return "impedance"
    if any(term in low for term in ["charged disk", "uniformly charged disk", "infinite line charge", "line charge", "dipole"]):
        return "electric_field"
    if any(term in low for term in ["electric field", "field", "force", "voltage", "current", "power", "capacitance", "energy"]):
        return parsed.target_quantity or "electric_field"
    return parsed.target_quantity


def build_objective(parsed: ParsedPhysicsProblem, question: str) -> MethodSearchObjective:
    quantity_units = sorted({quantity.si_unit for quantity in parsed.quantities})
    terms = structural_terms(question)
    priority_terms = _priority_query_terms(terms)
    compact_terms = " ".join(priority_terms[:6]) or "physics"
    target = _infer_search_target(question, parsed) or "physics"
    return MethodSearchObjective(
        question=question,
        target_quantity=parsed.target_quantity,
        quantity_units=quantity_units,
        structural_terms=terms,
        query_plan=[
            f"{target} method selection with quantities {' '.join(quantity_units) or 'unknown units'}",
            f"{target} derivation for {compact_terms}",
            *_specialized_queries(target, priority_terms),
        ],
    )


def _priority_query_terms(terms: list[str]) -> list[str]:
    priority = [
        "uniformly charged",
        "uniformly distributed",
        "open switch",
        "switch is open",
        "bridge",
        "ladder",
        "network",
        "ring",
        "axis",
        "transformer",
        "turns ratio",
        "reactance",
        "impedance",
        "dielectric",
        "permittivity",
        "solenoid",
        "inductor",
        "magnetic field",
        "perpendicular bisector",
        "midpoint",
        "superposition",
        "charged disk",
        "uniformly charged disk",
        "infinite line charge",
        "line charge",
        "dipole moment",
        "axial line",
        "capacitive reactance",
        "inductive reactance",
        "impedance",
        "spherical shell",
        "thin spherical shell",
        "shell",
    ]
    selected: list[str] = []
    for phrase in priority:
        if phrase in terms and phrase not in selected:
            selected.append(phrase)
    for term in terms:
        if term not in selected and len(term) > 2:
            selected.append(term)
    return selected


def _specialized_queries(target: str, terms: list[str]) -> list[str]:
    joined = " ".join(terms[:6]) or "physics"
    term_set = set(terms)
    queries = [f"{target} method derivation {joined}"]
    if "uniformly charged" in term_set and "ring" in term_set:
        queries.extend([
            f"{target} uniformly charged ring axis derivation",
            f"electric field on the axis of a uniformly charged ring derivation",
            f"uniformly charged ring electric field on axis formula",
        ])
    if "open switch" in term_set or "switch is open" in term_set:
        queries.extend([
            f"{target} open switch incomplete circuit reasoning",
            f"{target} no current open circuit method",
        ])
    if term_set.intersection({"bridge", "ladder", "network", "diamond"}):
        queries.extend([
            f"{target} bridge network equivalent resistance method",
            f"{target} network nodal analysis derivation",
        ])
    if term_set.intersection({"capacitive reactance", "inductive reactance", "impedance", "reactance"}):
        queries.extend([
            f"capacitive reactance formula",
            f"inductive reactance formula",
            f"reactance and impedance formulas",
            f"ac circuit reactance formula derivation",
        ])
    if term_set.intersection({"charged disk", "uniformly charged disk", "disk", "disc"}):
        queries.extend([
            f"electric field charged disk axis formula",
            f"uniformly charged disk electric field axis derivation",
            f"charged disk physicsbook electric field axis",
        ])
    if term_set.intersection({"spherical shell", "thin spherical shell", "shell"}):
        queries.extend([
            f"uniformly charged thin spherical shell electric field derivation",
            f"electric field thin spherical shell inside outside derivation",
            f"spherical shell electric field inside outside gauss law",
            f"uniformly charged spherical shell electric field at a point inside outside",
        ])
    if term_set.intersection({"infinite line charge", "line charge"}):
        queries.extend([
            f"infinite line of charge electric field formula",
            f"uniformly charged infinite line electric field",
            f"electric field due to infinite line charge derivation",
        ])
    if term_set.intersection({"dipole moment", "electric dipole", "axial line", "equatorial line"}):
        queries.extend([
            f"electric dipole axial field formula",
            f"electric dipole equatorial field formula",
            f"electric field of dipole derivation",
        ])
    if term_set.intersection({"reactance", "impedance", "series rlc"}):
        queries.extend([
            f"{target} ac circuit reactance impedance method",
            f"{target} resonance derivation",
        ])
    if term_set.intersection({"transformer", "turns ratio"}):
        queries.extend([
            f"transformer secondary voltage primary voltage turns ratio",
            f"ideal transformer primary secondary voltage ratio",
            f"transformer voltage ratio V_primary V_secondary N_primary N_secondary",
            f"{target} transformer turns ratio method",
            f"{target} ideal transformer relation",
        ])
    if term_set.intersection({"dielectric", "permittivity"}):
        queries.extend([
            f"{target} dielectric capacitor state change method",
            f"{target} dielectric derivation",
        ])
    if term_set.intersection({"solenoid", "inductor", "magnetic"}):
        queries.extend([
            f"{target} magnetic geometry method",
            f"{target} solenoid inductor derivation",
        ])
    if term_set.intersection({"perpendicular bisector", "midpoint", "superposition", "vector"}):
        queries.extend([
            f"{target} vector superposition method",
            f"{target} component symmetry derivation",
        ])
    # Deduplicate while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for query in queries:
        normalized = query.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def retrieve_method_evidence(objective: MethodSearchObjective, max_search_calls: int | None = None) -> list[MethodEvidenceSnippet]:
    snippets: list[MethodEvidenceSnippet] = []
    if _web_method_search_enabled():
        snippets.extend(_retrieve_web_evidence(objective, max_search_calls=max_search_calls))
    if not snippets:
        snippets.extend(_retrieve_local_evidence(objective))
    else:
        snippets.extend(_retrieve_local_evidence(objective))
    snippets = _dedupe_snippets(snippets)
    return snippets


def _retrieve_local_evidence(objective: MethodSearchObjective) -> list[MethodEvidenceSnippet]:
    terms = set(objective.structural_terms)
    results: list[MethodEvidenceSnippet] = []
    for snippet in _LOCAL_CORPUS:
        text = snippet.text.lower() + " " + snippet.title.lower()
        score = sum(1 for term in terms if term in text)
        target_ok = not objective.target_quantity or objective.target_quantity.replace("_", " ") in text or "field" in text
        if score >= 2 and target_ok:
            results.append(snippet)
    return results


def _retrieve_web_evidence(objective: MethodSearchObjective, max_search_calls: int | None = None) -> list[MethodEvidenceSnippet]:
    snippets: list[MethodEvidenceSnippet] = []
    max_search_calls = max_search_calls if max_search_calls is not None else int(os.environ.get("URA_MAX_SEARCH_CALLS") or "3")
    for query in objective.query_plan[:max_search_calls]:
        snippets.extend(_duckduckgo_search(query))
    snippets = _rank_snippets(objective, snippets)
    if not snippets:
        for query in objective.query_plan[:max_search_calls]:
            snippets.extend(_bing_search(query))
        snippets = _rank_snippets(objective, snippets)
    if snippets:
        snippets.extend(_fetch_result_pages(snippets[:3]))
    return snippets


def _duckduckgo_search(query: str) -> list[MethodEvidenceSnippet]:
    html = _fetch_search_html(query)
    if not html:
        return []
    titles, urls, snippets = _parse_duckduckgo_search(html)
    results: list[MethodEvidenceSnippet] = []
    for index, title in enumerate(titles[:8]):
        snippet_text = snippets[index] if index < len(snippets) else ""
        url = urls[index] if index < len(urls) else None
        results.append(
            MethodEvidenceSnippet(
                source="web_search",
                title=title,
                text=f"{title}. {snippet_text}".strip(),
                url=url,
            )
        )
    return results


def _bing_search(query: str) -> list[MethodEvidenceSnippet]:
    html = _fetch_bing_search_html(query)
    if not html:
        return []
    return _parse_bing_search(html)


def _fetch_bing_search_html(query: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; URA-Challenge/1.0; +https://example.com)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True, headers=headers) as client:
            response = client.get("https://www.bing.com/search", params={"q": query, "setlang": "en-US"})
            response.raise_for_status()
            if response.text.strip():
                return response.text
    except Exception:
        return ""
    return ""


def _parse_bing_search(html_text: str) -> list[MethodEvidenceSnippet]:
    clean = html_text.replace("\n", " ")
    matches = list(re.finditer(r'<li[^>]*class="[^"]*\bb_algo\b[^"]*"[^>]*>(.*?)</li>', clean, re.I | re.S))
    results: list[MethodEvidenceSnippet] = []
    for match in matches[:10]:
        block = match.group(1)
        link = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.I | re.S)
        if not link:
            link = re.search(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.I | re.S)
        if not link:
            continue
        url = html_lib.unescape(link.group(1))
        title = _strip_html(link.group(2))
        snippet_match = re.search(r'<p[^>]*>(.*?)</p>', block, re.I | re.S)
        if not snippet_match:
            snippet_match = re.search(r'<div[^>]*class="[^"]*\bb_caption\b[^"]*"[^>]*>(.*?)</div>', block, re.I | re.S)
        snippet_text = _strip_html(snippet_match.group(1)) if snippet_match else ""
        if not title:
            continue
        results.append(
            MethodEvidenceSnippet(
                source="bing_search",
                title=title,
                text=f"{title}. {snippet_text}".strip(),
                url=url,
            )
        )
    return results


def _fetch_search_html(query: str) -> str:
    endpoints = [
        "https://html.duckduckgo.com/html/",
        "https://lite.duckduckgo.com/lite/",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; URA-Challenge/1.0; +https://example.com)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    for endpoint in endpoints:
        try:
            with httpx.Client(timeout=8.0, follow_redirects=True, headers=headers) as client:
                response = client.get(endpoint, params={"q": query, "kl": "us-en"})
                response.raise_for_status()
                if response.text.strip():
                    return response.text
        except Exception:
            continue
    return ""


def _parse_duckduckgo_search(html_text: str) -> tuple[list[str], list[str], list[str]]:
    clean = html_text.replace("\n", " ")
    title_matches = list(re.finditer(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', clean, re.I | re.S))
    if not title_matches:
        title_matches = list(re.finditer(r'<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', clean, re.I | re.S))
    snippet_matches = list(
        re.finditer(
            r'<(?:a|div)[^>]*class="result__snippet"[^>]*>(.*?)</(?:a|div)>',
            clean,
            re.I | re.S,
        )
    )
    titles = [_strip_html(match.group(2)) for match in title_matches]
    urls = [unquote(_normalize_ddg_url(match.group(1))) for match in title_matches]
    snippets = [_strip_html(match.group(1)) for match in snippet_matches]
    return titles, urls, snippets


def _normalize_ddg_url(url: str) -> str:
    parsed = urlparse(html_lib.unescape(url))
    if parsed.path.endswith("/l/") and parsed.query:
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return qs["uddg"][0]
    if "duckduckgo.com/l/" in url and "uddg=" in url:
        qs = parse_qs(parsed.query)
        if "uddg" in qs:
            return qs["uddg"][0]
    return html_lib.unescape(url)


def _fetch_result_pages(snippets: list[MethodEvidenceSnippet]) -> list[MethodEvidenceSnippet]:
    results: list[MethodEvidenceSnippet] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; URA-Challenge/1.0; +https://example.com)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    for snippet in snippets:
        if not snippet.url:
            continue
        try:
            with httpx.Client(timeout=6.0, follow_redirects=True, headers=headers) as client:
                response = client.get(snippet.url)
                response.raise_for_status()
        except Exception:
            continue
        page_text = _html_to_text(response.text)
        if not page_text.strip():
            continue
        results.append(
            MethodEvidenceSnippet(
                source="web_page",
                title=snippet.title,
                text=f"{snippet.title}. {page_text[:3500]}",
                url=snippet.url,
            )
        )
    return results


def _html_to_text(text: str) -> str:
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_html(text: str) -> str:
    return _html_to_text(text)


def _dedupe_snippets(snippets: list[MethodEvidenceSnippet]) -> list[MethodEvidenceSnippet]:
    seen: set[tuple[str, str]] = set()
    results: list[MethodEvidenceSnippet] = []
    for snippet in snippets:
        key = (snippet.source, snippet.title.lower())
        if key in seen:
            continue
        seen.add(key)
        results.append(snippet)
    return results


def _rank_snippets(objective: MethodSearchObjective, snippets: list[MethodEvidenceSnippet]) -> list[MethodEvidenceSnippet]:
    if not snippets:
        return []
    terms = [term.lower() for term in objective.structural_terms if term]
    query_terms = {token for query in objective.query_plan for token in re.findall(r"[a-zA-Z]+", query.lower()) if len(token) > 2}

    def score(snippet: MethodEvidenceSnippet) -> tuple[int, int, int]:
        text = f"{snippet.title} {snippet.text}".lower()
        term_hits = sum(1 for term in terms if term in text)
        query_hits = sum(1 for term in query_terms if term in text)
        url_bonus = 1 if snippet.url else 0
        return (term_hits, query_hits, url_bonus)

    scored = sorted(snippets, key=score, reverse=True)
    return scored[:12]


_TARGET_UNIT_BY_LHS = {
    "e": "N/C",
    "v": "V",
    "v_primary": "V",
    "v_secondary": "V",
    "primary_voltage": "V",
    "secondary_voltage": "V",
    "voltage": "V",
    "i": "A",
    "p": "W",
    "q": "C",
    "c": "F",
    "r": "ohm",
    "f": "N",
    "b": "T",
    "x_l": "ohm",
    "x_c": "ohm",
    "z": "ohm",
    "u": "J",
    "l": "H",
    "omega": "rad/s",
}


def _extract_equation_candidates(text: str) -> list[tuple[str, str]]:
    clean = _html_to_text(text)
    matches: list[tuple[str, str]] = []
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]*(?:\s*[_][A-Za-z0-9_]+)?)\s*=\s*([^.;\n]+)", clean):
        lhs = match.group(1).strip()
        rhs = match.group(2).strip().rstrip(",;:")
        rhs = re.split(r"\b(?:where|with|for|given that|assuming)\b", rhs, maxsplit=1, flags=re.I)[0].strip()
        if not rhs:
            continue
        rhs = rhs.rstrip(",;:")
        if not re.search(r"[0-9A-Za-z]", rhs):
            continue
        if not re.search(r"[+\-*/^()]", rhs) and not re.search(r"\b(?:sqrt|pi|sin|cos)\b", rhs, re.I):
            continue
        matches.append((lhs, rhs))
    return matches


def _infer_method_family_from_text(text: str, lhs: str) -> str:
    low = text.lower()
    lhs_norm = lhs.lower().replace(" ", "_")
    if any(term in low for term in ["wave", "wavelength", "frequency", "wave speed"]):
        return "wave_relation"
    if any(term in low for term in ["open switch", "switch is open", "incomplete circuit", "no current"]):
        return "circuit_open_state"
    if any(term in low for term in ["bridge", "ladder", "mesh", "diamond", "network of resistors", "nodal analysis", "mesh analysis"]):
        return "network_reduction_or_symbolic"
    if any(term in low for term in ["uniformly charged", "distributed charge", "linear charge density", "surface charge density", "ring", "wire", "rod", "arc", "plate", "semicircle", "axis", "disk", "disc", "sheet", "line charge"]):
        return "distributed_charge_integration"
    if any(term in low for term in ["perpendicular bisector", "midpoint", "equidistant", "superposition", "vector sum", "components", "resultant", "dipole", "axial line", "equatorial line"]):
        return "vector_superposition"
    if any(term in low for term in ["dielectric", "relative permittivity", "epsilon_r", "connected", "disconnected"]):
        return "dielectric_transform"
    if any(term in low for term in ["resonance", "impedance", "reactance", "series rlc", "ac circuit", "alternating current"]):
        return "ac_circuit_method"
    if any(term in low for term in ["transformer", "primary", "secondary", "turns ratio"]):
        return "transformer_relation"
    if any(term in low for term in ["solenoid", "inductor", "magnetic field", "magnetic flux", "coil"]):
        return "magnetics_geometry"
    if lhs_norm in {"e", "v", "i", "p", "q", "c", "r", "f", "b", "x_l", "x_c", "z", "u", "l", "omega"}:
        return "direct_physics_relation"
    return "direct_physics_relation"


def _variables_from_equation(expression: str) -> tuple[str, ...]:
    normalized = normalize_equation_expression(expression)
    names = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", normalized):
        lowered = token.lower()
        if lowered in {"sqrt", "pi", "sin", "cos", "tan"}:
            continue
        if token not in names:
            names.append(token)
    return tuple(names)


def _looks_like_point_charge_expression(expression: str) -> bool:
    compact = re.sub(r"\s+", "", expression.lower())
    compact = compact.replace("·", "*")
    patterns = (
        r"k\*?q/?r\^?2$",
        r"kq/?r\^?2$",
        r"k\*?q/\(r\^?2\)",
        r"kq/\(r\^?2\)",
        r"k\*?q/r\^?2",
        r"kq/r\^?2",
    )
    return any(re.search(pattern, compact) for pattern in patterns)


def _looks_like_ring_axis_expression(expression: str) -> bool:
    compact = normalize_equation_expression(expression).lower().replace(" ", "")
    return any(
        pattern in compact
        for pattern in [
            "k*q*x/(x**2+r**2)**(3/2)",
            "k*q*x/(r**2+x**2)**(3/2)",
            "k*q*x/(x^2+r^2)^(3/2)",
            "k*q*x/(r^2+x^2)^(3/2)",
        ]
    )


def _objective_has_any(objective: MethodSearchObjective, terms: list[str]) -> bool:
    term_set = set(objective.structural_terms)
    return any(term in term_set for term in terms)


def _method_assumptions(method_family: str, snippet: MethodEvidenceSnippet, objective: MethodSearchObjective) -> tuple[str, ...]:
    low = f"{snippet.title} {snippet.text}".lower()
    if method_family == "distributed_charge_integration":
        assumptions = []
        if "ring" in low:
            assumptions.extend(["ring", "axis"])
        elif "wire" in low or "rod" in low or "arc" in low or "plate" in low:
            assumptions.append("distributed charge")
        if "uniform" in low:
            assumptions.append("uniformly charged")
        return tuple(dict.fromkeys(assumptions))
    if method_family == "network_reduction_or_symbolic":
        assumptions = [term for term in ["bridge", "ladder", "network", "diamond"] if term in low]
        return tuple(dict.fromkeys(assumptions))
    if method_family == "vector_superposition":
        assumptions = [term for term in ["midpoint", "perpendicular bisector", "superposition", "vector"] if term in low]
        return tuple(dict.fromkeys(assumptions))
    if method_family == "dielectric_transform":
        assumptions = [term for term in ["dielectric", "permittivity"] if term in low]
        return tuple(dict.fromkeys(assumptions))
    if method_family == "ac_circuit_method":
        assumptions = [term for term in ["ac circuit", "reactance", "impedance", "resonance"] if term in low]
        return tuple(dict.fromkeys(assumptions))
    if method_family == "transformer_relation":
        assumptions = [term for term in ["transformer", "primary", "secondary", "turns ratio"] if term in low]
        return tuple(dict.fromkeys(assumptions))
    if method_family == "magnetics_geometry":
        assumptions = [term for term in ["solenoid", "inductor", "magnetic field", "magnetic flux", "coil"] if term in low]
        return tuple(dict.fromkeys(assumptions))
    return ()


def _proposal_from_equation(snippet: MethodEvidenceSnippet, lhs: str, rhs: str, objective: MethodSearchObjective) -> MethodEquationProposal | None:
    method_family = _infer_method_family_from_text(f"{snippet.title} {snippet.text}", lhs)
    target_unit = _TARGET_UNIT_BY_LHS.get(lhs.lower().replace(" ", "_"))
    if lhs.lower().replace(" ", "_") == "f" and objective.target_quantity == "frequency":
        target_unit = "Hz"
    if not target_unit:
        return None
    variables = _variables_from_equation(rhs)
    if not variables:
        return None
    low = f"{snippet.title} {snippet.text}".lower()
    blocked: list[str] = []
    if method_family == "distributed_charge_integration":
        blocked.append("point_charge")
    if method_family == "network_reduction_or_symbolic":
        blocked.append("simple_network_reduction")
    if objective.target_quantity == "electric_field" and lhs.lower() not in {"e"}:
        return None
    if objective.target_quantity == "current" and lhs.lower() not in {"i"}:
        return None
    if objective.target_quantity == "voltage" and lhs.lower() not in {"v"}:
        return None
    if objective.target_quantity == "power" and lhs.lower() not in {"p"}:
        return None
    if objective.target_quantity == "charge" and lhs.lower() not in {"q"}:
        return None
    if objective.target_quantity == "capacitance" and lhs.lower() not in {"c"}:
        return None
    if objective.target_quantity == "resistance" and lhs.lower() not in {"r", "z"}:
        return None
    if objective.target_quantity in {"reactance", "impedance"} and lhs.lower() not in {"x_l", "x_c", "z", "omega"}:
        return None
    if objective.target_quantity == "frequency" and lhs.lower() not in {"f"}:
        return None
    if _objective_has_any(objective, ["ring", "wire", "rod", "arc", "plate", "uniformly charged", "distributed charge"]) and _looks_like_point_charge_expression(rhs):
        return None
    if _objective_has_any(objective, ["disk", "disc", "line charge", "line", "wire"]) and _looks_like_ring_axis_expression(rhs):
        return None
    if method_family == "distributed_charge_integration" and not any(term in low for term in ["ring", "wire", "rod", "arc", "plate", "axis", "distributed charge"]):
        return None
    if _objective_has_any(objective, ["bridge", "ladder", "network", "diamond"]) and method_family == "direct_physics_relation" and objective.target_quantity in {"resistance", "capacitance"}:
        return None
    if method_family == "network_reduction_or_symbolic" and not any(term in low for term in ["bridge", "ladder", "mesh", "diamond", "network"]):
        return None
    if method_family == "vector_superposition" and not any(term in low for term in ["midpoint", "perpendicular bisector", "superposition", "vector", "dipole", "axial line", "equatorial line"]):
        return None
    if method_family == "dielectric_transform" and not any(term in low for term in ["dielectric", "permittivity"]):
        return None
    if method_family == "ac_circuit_method" and not any(term in low for term in ["resonance", "impedance", "reactance", "ac circuit"]):
        return None
    if method_family == "transformer_relation" and not any(term in low for term in ["transformer", "primary", "secondary", "turns ratio"]):
        return None
    if method_family == "magnetics_geometry" and not any(term in low for term in ["solenoid", "inductor", "magnetic"]):
        return None
    if method_family == "transformer_relation":
        normalized_rhs = rhs.lower().replace(" ", "")
        if "voltage" not in low and "turn" not in low:
            return None
        if "v_secondary" in normalized_rhs or "vprimary" in normalized_rhs or "v_primary" in normalized_rhs:
            pass
        elif "v_secondary" not in low and "secondary" not in low and "primary" not in low:
            return None
    return MethodEquationProposal(
        method_id=f"retrieved_{lhs.lower()}_equation",
        method_family=method_family,
        expression=rhs,
        target_unit=target_unit,
        variables=variables,
        assumptions=_method_assumptions(method_family, snippet, objective),
        evidence=snippet,
        blocked_formula_families=tuple(blocked),
        confidence=0.9 if method_family in {"distributed_charge_integration", "direct_physics_relation"} else 0.8,
    )


def extract_equation_proposals(objective: MethodSearchObjective, snippets: list[MethodEvidenceSnippet]) -> list[MethodEquationProposal]:
    proposals: list[MethodEquationProposal] = []
    for snippet in snippets:
        candidates = _extract_equation_candidates(f"{snippet.title}. {snippet.text}")
        for lhs, rhs in candidates:
            proposal = _proposal_from_equation(snippet, lhs, rhs, objective)
            if proposal is not None:
                proposals.append(proposal)
        if not candidates:
            proposal = _proposal_from_ratio_relation(snippet, objective)
            if proposal is not None:
                proposals.append(proposal)
    return proposals


def _proposal_from_ratio_relation(snippet: MethodEvidenceSnippet, objective: MethodSearchObjective) -> MethodEquationProposal | None:
    low = f"{snippet.title} {snippet.text}".lower()
    question_low = objective.question.lower()
    if "transformer" not in low:
        return None
    if not any(term in low for term in ["primary", "secondary", "turns", "voltage"]):
        return None
    if objective.target_quantity not in {"voltage", "current"}:
        return None
    if "secondary voltage" in question_low and "primary voltage" in question_low:
        expression = "V_primary * N_secondary / N_primary"
        lhs = "V_secondary"
    elif "primary voltage" in question_low and "secondary voltage" in question_low:
        expression = "V_secondary * N_primary / N_secondary"
        lhs = "V_primary"
    elif "voltage ratio" in low or "turns ratio" in low or "turns ratio" in question_low:
        expression = "V_primary * N_secondary / N_primary"
        lhs = "V_secondary"
    else:
        return None
    return MethodEquationProposal(
        method_id="retrieved_transformer_ratio_equation",
        method_family="transformer_relation",
        expression=expression,
        target_unit="V",
        variables=_variables_from_equation(expression),
        assumptions=_method_assumptions("transformer_relation", snippet, objective),
        evidence=snippet,
        blocked_formula_families=(),
        confidence=0.88,
    )


def verify_and_compute_method(
    parsed: ParsedPhysicsProblem,
    question: str,
    proposal: MethodEquationProposal,
    reject_reasons: list[str] | None = None,
) -> VerifiedMethod | None:
    low = question.lower()
    for assumption in proposal.assumptions:
        if assumption and assumption not in low:
            # Keep verification conservative: extracted evidence must map back to the question.
            if reject_reasons is not None:
                reject_reasons.append(f"assumption mismatch: '{assumption}' not in question")
            return None

    # Cross-validate target units against requested target quantity
    _TARGET_UNIT_HINTS = {
        "voltage": "V",
        "electric_potential": "V",
        "current": "A",
        "power": "W",
        "resistance": "ohm",
        "impedance": "ohm",
        "reactance": "ohm",
        "capacitance": "F",
        "charge": "C",
        "energy": "J",
        "potential_energy": "J",
        "force": "N",
        "electric_field": "N/C",
        "frequency": "Hz",
        "angular_frequency": "rad/s",
        "inductance": "H",
        "magnetic_field": "T",
        "distance": "m",
        "dielectric_constant": "dimensionless",
    }
    if parsed.target_quantity:
        expected_unit = _TARGET_UNIT_HINTS.get(parsed.target_quantity)
        if expected_unit:
            prop_unit_norm = proposal.target_unit.lower().replace(" ", "").replace("·", "*")
            expected_unit_norm = expected_unit.lower().replace(" ", "").replace("·", "*")
            equivalent = False
            if prop_unit_norm == expected_unit_norm:
                equivalent = True
            elif expected_unit_norm == "n/c" and prop_unit_norm in {"n/c", "v/m"}:
                equivalent = True
            elif expected_unit_norm == "v/m" and prop_unit_norm in {"n/c", "v/m"}:
                equivalent = True
            elif expected_unit_norm == "ohm" and prop_unit_norm in {"ohm", "ω", "Ω", "reactance", "impedance"}:
                equivalent = True

            if not equivalent:
                # Target unit mismatch - reject the proposal
                if reject_reasons is not None:
                    reject_reasons.append(f"target unit mismatch: expected {expected_unit_norm}, got {prop_unit_norm}")
                return None

    variables = _extract_method_variables(parsed, question, proposal)
    missing = set(proposal.variables) - set(variables)
    if missing:
        if reject_reasons is not None:
            reject_reasons.append(f"missing variables: {', '.join(missing)}")
        return None
    value = safe_eval_expression(proposal.expression, variables)
    return VerifiedMethod(
        proposal=proposal,
        variables=variables,
        value=value,
        verification_notes=[
            "assumptions matched question",
            "variables extracted from parsed quantities",
            "target unit verified and matched expected quantity",
            "equation evaluated in safe math evaluator"
        ],
    )


def _extract_method_variables(parsed: ParsedPhysicsProblem, question: str, proposal: MethodEquationProposal) -> dict[str, float]:
    values: dict[str, float] = {}
    charges = [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == "C"]
    charge_density_line = [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == "C/m"]
    charge_density_area = [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == "C/m²"]
    dipole_moments = [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == "C·m"]
    distances = [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == "m"]
    voltages = [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == "V"]
    currents = [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == "A"]
    resistances = [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == "ohm"]
    capacitances = [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == "F"]
    frequencies = [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == "Hz"]
    inductances = [quantity.si_value for quantity in parsed.quantities if quantity.si_unit == "H"]

    def first_or_none(values: list[float]) -> float | None:
        return values[0] if values else None

    for symbol in proposal.variables:
        key = symbol.lower()
        if key == "k":
            values[symbol] = K_COULOMB
            continue
        if key in {"v_secondary", "v_primary"}:
            picked = _extract_labeled_voltage(question, symbol) or first_or_none(voltages)
            if picked is not None:
                values[symbol] = picked
            continue
        if key in {"q"} and charges:
            values[symbol] = charges[0]
            continue
        if key in {"sigma"} and charge_density_area:
            values[symbol] = charge_density_area[0]
            continue
        if key in {"lambda", "lam"} and charge_density_line:
            values[symbol] = charge_density_line[0]
            continue
        if key in {"p"} and dipole_moments:
            values[symbol] = dipole_moments[0]
            continue
        if key in {"v", "v1", "v2", "v_primary", "v_source"}:
            picked = _extract_labeled_voltage(question, symbol) or first_or_none(voltages)
            if picked is not None:
                values[symbol] = picked
            continue
        if key in {"i"} and currents:
            values[symbol] = currents[0]
            continue
        if key in {"c"} and capacitances:
            values[symbol] = capacitances[0]
            continue
        if key in {"f"} and frequencies:
            values[symbol] = frequencies[0]
            continue
        if key in {"l"} and inductances:
            values[symbol] = inductances[0]
            continue
        if key in {"r"}:
            if proposal.method_family == "distributed_charge_integration":
                radius = _extract_labeled_distance(question, ["radius", "ring radius"])
                if radius is None:
                    radius = first_or_none(distances)
                if radius is not None:
                    values[symbol] = radius
            elif resistances:
                values[symbol] = resistances[0]
            continue
        if key in {"x", "d"}:
            axis_distance = _extract_labeled_distance(question, ["axis", "from the center", "from center", "distance"])
            if axis_distance is None:
                axis_distance = first_or_none(distances)
            if axis_distance is not None:
                values[symbol] = axis_distance
            continue
        if key in {"z"} and proposal.method_family == "distributed_charge_integration":
            axis_distance = _extract_labeled_distance(question, ["axis", "from the center", "from center", "distance", "above the disk", "away"])
            if axis_distance is None:
                axis_distance = first_or_none(distances)
            if axis_distance is not None:
                values[symbol] = axis_distance
            continue
        if key in {"n", "n1", "n2"}:
            turns = _extract_turns_count(question, symbol)
            if turns is not None:
                values[symbol] = turns
            continue
        if key in {"n_primary", "n_secondary", "np", "ns"}:
            turns = _extract_turns_count(question, symbol)
            if turns is not None:
                values[symbol] = turns
            continue
        if key in {"theta", "theta_deg", "theta_rad"}:
            angle = _extract_angle_value(question)
            if angle is not None:
                values[symbol] = math.radians(angle) if key.endswith("rad") else angle
            continue
        if key in {"z"} and resistances:
            values[symbol] = resistances[0]
            continue
        if key in {"omega"}:
            omega = _extract_angular_frequency(question)
            if omega is not None:
                values[symbol] = omega
            continue
        if key in {"eps0", "epsilon_0", "epsilon0"}:
            values[symbol] = 8.854e-12
            continue
    return values


def _extract_labeled_voltage(question: str, symbol: str) -> float | None:
    low = question.lower()
    if symbol.lower() in {"v_primary", "v1"}:
        match = re.search(r"(?:primary\s+voltage|v1|v_primary)\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(mv|v|kv)\b", low, re.I)
        if match:
            return _convert_scalar(match.group(1), match.group(2))
    if symbol.lower() in {"v2", "v_secondary"}:
        match = re.search(r"(?:secondary\s+voltage|v2|v_secondary)\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(mv|v|kv)\b", low, re.I)
        if match:
            return _convert_scalar(match.group(1), match.group(2))
    match = re.search(r"(?:voltage|potential difference)\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\s*(mv|v|kv)\b", low, re.I)
    if match:
        return _convert_scalar(match.group(1), match.group(2))
    return None

def _extract_turns_count(question: str, symbol: str) -> float | None:
    low = question.lower()
    if symbol.lower() in {"n1", "n_primary"}:
        patterns = [
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:turns?|coils?)\s*(?:on\s+the\s+)?primary\b",
            r"(?:primary\s+turns|turns\s+on\s+the\s+primary|primary\s+has\s+)([0-9]+(?:\.[0-9]+)?)\s*(?:turns?|coils?)\b",
            r"(?:primary\s+turns|n1|n_primary)\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, low, re.I)
            if match:
                return float(match.group(1))
    if symbol.lower() in {"n2", "n_secondary"}:
        patterns = [
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:turns?|coils?)\s*(?:on\s+the\s+)?secondary\b",
            r"(?:secondary\s+turns|turns\s+on\s+the\s+secondary|secondary\s+has\s+)([0-9]+(?:\.[0-9]+)?)\s*(?:turns?|coils?)\b",
            r"(?:secondary\s+turns|n2|n_secondary)\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, low, re.I)
            if match:
                return float(match.group(1))
    if symbol.lower() in {"np"}:
        patterns = [
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:turns?|coils?)\s*(?:on\s+the\s+)?primary\b",
            r"(?:primary\s+turns|turns\s+on\s+the\s+primary|primary\s+has\s+)([0-9]+(?:\.[0-9]+)?)\s*(?:turns?|coils?)\b",
            r"(?:primary\s+turns|np)\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, low, re.I)
            if match:
                return float(match.group(1))
    if symbol.lower() in {"ns"}:
        patterns = [
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:turns?|coils?)\s*(?:on\s+the\s+)?secondary\b",
            r"(?:secondary\s+turns|turns\s+on\s+the\s+secondary|secondary\s+has\s+)([0-9]+(?:\.[0-9]+)?)\s*(?:turns?|coils?)\b",
            r"(?:secondary\s+turns|ns)\s*(?:=|is|of)?\s*([0-9]+(?:\.[0-9]+)?)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, low, re.I)
            if match:
                return float(match.group(1))
    match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:turns?|coils?)\b", low, re.I)
    if match:
        return float(match.group(1))
    return None


def _extract_angular_frequency(question: str) -> float | None:
    match = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*(?:rad/s|rads|angular frequency)\b", question, re.I)
    if match:
        return float(match.group(1))
    return None


def _extract_angle_value(question: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:degrees?|°)\b", question, re.I)
    if match:
        return float(match.group(1))
    return None


def _convert_scalar(value: str, unit: str) -> float:
    numeric = float(value)
    unit = unit.lower()
    if unit == "kv":
        return numeric * 1e3
    if unit == "mv":
        return numeric * 1e-3
    return numeric


def _extract_labeled_distance(question: str, labels: list[str]) -> float | None:
    for label in labels:
        if label in {"from the center", "from center"}:
            pattern = r"([0-9]+(?:\.[0-9]+)?)\s*(?:m|meter|meters|cm|mm)\s+from\s+(?:the\s+)?center"
        else:
            pattern = rf"{re.escape(label)}(?:\s+of)?\s+([0-9]+(?:\.[0-9]+)?)\s*(m|meter|meters|cm|mm)\b"
        match = re.search(pattern, question, re.I)
        if not match:
            continue
        value = float(match.group(1))
        unit = match.group(2).lower() if len(match.groups()) > 1 else "m"
        if unit == "cm":
            return value * 1e-2
        if unit == "mm":
            return value * 1e-3
        return value
    return None
