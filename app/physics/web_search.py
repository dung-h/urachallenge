from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from app.physics.formula_cache import CachedFormulaContext, FormulaCache


SERPER_SEARCH_URL = "https://google.serper.dev/search"


@dataclass(frozen=True)
class FormulaSearchResult:
    context: str
    sources: list[dict[str, Any]]
    search_query: str
    cache_hit: bool
    cache_key: str | None = None


def build_formula_search_query(question: str) -> str:
    compact = re.sub(r"\s+", " ", question).strip()
    return f"physics formula {compact}"


def _organic_results(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    results = payload.get("organic") or []
    cleaned: list[dict[str, Any]] = []
    for item in results[:limit]:
        title = str(item.get("title") or "").strip()
        link = str(item.get("link") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if not (title or snippet):
            continue
        cleaned.append({"title": title, "link": link, "snippet": snippet})
    return cleaned


def _context_from_sources(sources: list[dict[str, Any]]) -> str:
    lines = []
    for index, source in enumerate(sources, start=1):
        title = source.get("title") or "Untitled source"
        snippet = source.get("snippet") or ""
        link = source.get("link") or ""
        lines.append(f"[{index}] {title}\nSnippet: {snippet}\nURL: {link}")
    return "\n\n".join(lines)


def search_formula_context(
    question: str,
    *,
    cache: FormulaCache | None = None,
    api_key: str | None = None,
    limit: int = 5,
    timeout: float = 12.0,
) -> FormulaSearchResult | None:
    cache = cache or FormulaCache()
    cached: CachedFormulaContext | None = cache.get(question)
    if cached is not None:
        return FormulaSearchResult(
            context=cached.context,
            sources=cached.sources,
            search_query=cached.search_query,
            cache_hit=True,
            cache_key=cached.key,
        )

    key = api_key or os.environ.get("SERPER_API_KEY")
    if not key:
        return None

    search_query = build_formula_search_query(question)
    payload = {"q": search_query, "num": limit}
    headers = {"X-API-KEY": key, "Content-Type": "application/json"}
    with httpx.Client(timeout=timeout) as client:
        response = client.post(SERPER_SEARCH_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    sources = _organic_results(data, limit=limit)
    if not sources:
        return None
    context = _context_from_sources(sources)
    cached = cache.set(question, search_query, context, sources)
    return FormulaSearchResult(
        context=cached.context,
        sources=cached.sources,
        search_query=cached.search_query,
        cache_hit=False,
        cache_key=cached.key,
    )
