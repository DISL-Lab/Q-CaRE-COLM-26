"""Parsers for the backbone's raw responses.

Two behaviours are available for the chunk-fact step, selected by
``--parsing``:

``paper`` (default)
    Reproduces the published results. Whatever the model returns in
    ``verified_facts`` is kept verbatim, so a fact named once by its key and
    once by its text counts twice.

``strict``
    Maps fact text back to its key first, so a fact is never counted twice.
    Cleaner, but it does not reproduce the published Verifiableness numbers.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

PARSING_MODES = ("paper", "strict")


def load_json_block(response: str) -> dict:
    """Parse a JSON object from a response, tolerating a ```json fence."""
    if not isinstance(response, str):
        return {}
    content = response.strip()
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    content = content.replace("\\n", "\n")
    try:
        parsed = json.loads(content)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_subqueries(response: str) -> Dict[str, str]:
    """Read the ``query_decomposition`` object, keys unchanged."""
    decomposition = load_json_block(response).get("query_decomposition")
    if isinstance(decomposition, dict):
        return {str(k): str(v) for k, v in decomposition.items()}
    return {}


def parse_atomic_facts(response: str) -> Dict[str, str]:
    """Read the ``atomic_facts`` object and renumber the keys in order."""
    facts = load_json_block(response).get("atomic_facts")
    if isinstance(facts, dict):
        return {f"Atomic fact{i}": str(v) for i, (_, v) in enumerate(facts.items(), 1)}
    if isinstance(facts, list):
        return {f"Atomic fact{i}": str(v) for i, v in enumerate(facts, 1)}
    return {}


def parse_verified_facts(
    response: str,
    atomic_facts: Dict[str, str] | None = None,
    mode: str = "paper",
) -> List[str]:
    """Read the ``verified_facts`` list from a chunk-fact judgement."""
    facts = load_json_block(response).get("verified_facts")
    if not isinstance(facts, list):
        return []
    if mode == "paper":
        return facts
    return map_facts_to_keys(facts, atomic_facts or {})


def map_facts_to_keys(raw_facts: List, atomic_facts: Dict[str, str]) -> List[str]:
    """Normalise fact references to their ``Atomic factN`` key (strict mode)."""
    by_text = {text.strip().lower(): key for key, text in atomic_facts.items()}
    resolved: List[str] = []
    for item in raw_facts:
        if not isinstance(item, str) or not item.strip():
            continue
        item = item.strip()

        key_match = re.search(r"atomic[_\s]*fact[_\s]*(\d+)", item, re.IGNORECASE)
        if key_match:
            resolved.append(f"Atomic fact{key_match.group(1)}")
            continue

        lowered = item.lower()
        if lowered in by_text:
            resolved.append(by_text[lowered])
            continue

        for text, key in by_text.items():
            if lowered in text or text in lowered:
                resolved.append(key)
                break
    # preserve order, drop duplicates
    return list(dict.fromkeys(resolved))


def is_yes(response: str) -> bool:
    """Yes/No judgement, matching the published implementation."""
    return isinstance(response, str) and "yes" in response.lower()


def parse_answer(response: str) -> str:
    """Read the ``{"Answer": ...}`` object a generator returns.

    Used when building a benchmark (see ``scripts/generate_answers.py``), not
    during evaluation. Falls back to the raw text if no JSON is found.
    """
    if not isinstance(response, str) or not response.strip():
        return ""

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    candidates = list(fenced)
    candidates += sorted(
        re.findall(r"\{(?:[^{}]|\{[^{}]*\})*\}", response, re.DOTALL), key=len, reverse=True
    )
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and "Answer" in parsed:
            return str(parsed["Answer"])
    return response.strip()
