from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_WRAPPER_RE = re.compile(r"^[`\"'“”‘’\s]+|[`\"'“”‘’\s]+$")
_LIST_PREFIX_RE = re.compile(r"^(?:[>\-*•]\s*|\d+[\).]\s*)+")
_NOISE_PREFIXES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ignore_instruction",
        re.compile(
            r"^\s*(?:please\s+)?(?:ignore|forget)\b.*?(?:previous|prior|above|earlier|the previous sentence|instructions?)\b(?:[,:;.-]\s*|\s+)*",
            re.I,
        ),
    ),
    (
        "tool_directive",
        re.compile(r"^\s*(?:do not|don't|never|avoid)\s+(?:use\s+)?tools?\b(?:[,:;.-]\s*|\s+)*", re.I),
    ),
    (
        "answer_only",
        re.compile(r"^\s*(?:answer|reply|respond|output|return)\s+(?:only|just)\b.*?(?:[,:;.-]\s*|\s+)*", re.I),
    ),
    (
        "web_search",
        re.compile(r"^\s*(?:web search|search the web|browse the web|use the web|use web search)\b(?:[,:;.-]\s*|\s+)*", re.I),
    ),
)
_NOISE_SENTENCE_RE = re.compile(
    r"^\s*(?:"
    r"ignore\b.*?(?:previous|prior|above|earlier|instructions?|sentence)\b"
    r"|(?:do not|don't|never|avoid)\s+(?:use\s+)?tools?\b"
    r"|(?:answer|reply|respond|output|return)\s+(?:only|just)\b"
    r"|(?:web search|search the web|browse the web|use the web|use web search)\b"
    r")\b.*$",
    re.I,
)


@dataclass(frozen=True)
class GuardrailResult:
    original_text: str
    normalized_text: str
    noise_detected: bool
    noise_markers: tuple[str, ...] = ()
    removed_segments: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "noise_detected": self.noise_detected,
            "noise_markers": list(self.noise_markers),
            "removed_segments": list(self.removed_segments),
        }


def _strip_wrappers(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.count("```") >= 2:
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    value = re.sub(r"^```(?:\w+)?\s*", "", value)
    value = re.sub(r"\s*```$", "", value)
    value = _WRAPPER_RE.sub("", value)
    return value.strip()


def _strip_list_prefix(text: str) -> str:
    return _LIST_PREFIX_RE.sub("", text).strip()


def _strip_noise_prefixes(text: str) -> tuple[str, list[str]]:
    cleaned = text
    removed: list[str] = []
    changed = True
    while changed and cleaned:
        changed = False
        for marker, pattern in _NOISE_PREFIXES:
            match = pattern.match(cleaned)
            if not match:
                continue
            removed.append(marker)
            cleaned = cleaned[match.end():].lstrip(" \t\r\n-–—:;,.")
            changed = True
            break
    return cleaned, removed


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part and part.strip()]


def _clean_segment(segment: str) -> tuple[str, list[str]]:
    stripped = _strip_wrappers(segment)
    stripped = _strip_list_prefix(stripped)
    stripped, removed = _strip_noise_prefixes(stripped)
    if not stripped:
        return "", removed

    cleaned_sentences: list[str] = []
    for sentence in _split_sentences(stripped):
        candidate = sentence.strip()
        if not candidate:
            continue
        if _NOISE_SENTENCE_RE.match(candidate):
            removed.append("noise_sentence")
            continue
        cleaned_sentences.append(candidate)

    if not cleaned_sentences and not removed:
        return stripped, removed
    return " ".join(cleaned_sentences).strip(), removed


def guardrail_prompt_text(text: str) -> GuardrailResult:
    original = str(text or "")
    segments = original.splitlines() if "\n" in original else [original]
    cleaned_segments: list[str] = []
    markers: list[str] = []
    removed_segments: list[str] = []

    for raw_segment in segments:
        cleaned_segment, removed = _clean_segment(raw_segment)
        if removed:
            markers.extend(removed)
            removed_segments.append(raw_segment.strip())
        if cleaned_segment:
            cleaned_segments.append(cleaned_segment)

    normalized_lines = [re.sub(r"[ \t]+", " ", segment).strip() for segment in cleaned_segments]
    normalized_text = "\n".join(segment for segment in normalized_lines if segment).strip()
    if not normalized_text:
        normalized_text = _strip_wrappers(original).strip()
        normalized_text = "\n".join(
            re.sub(r"[ \t]+", " ", segment).strip()
            for segment in normalized_text.splitlines()
            if segment.strip()
        ).strip()

    deduped_markers = tuple(dict.fromkeys(markers))
    deduped_removed = tuple(segment for segment in removed_segments if segment)
    return GuardrailResult(
        original_text=original,
        normalized_text=normalized_text,
        noise_detected=bool(deduped_markers or deduped_removed),
        noise_markers=deduped_markers,
        removed_segments=deduped_removed,
    )
