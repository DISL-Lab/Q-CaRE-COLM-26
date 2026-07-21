"""Prompt loading.

All prompts live in a YAML file (``configs/prompts.yaml`` by default) so they can
be edited, versioned, or swapped for a different backbone without touching the
pipeline code.
"""

from __future__ import annotations

import os
from typing import Dict, List

import yaml

DEFAULT_PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "prompts.yaml",
)


class PromptLibrary:
    """Loads prompt templates and renders them into chat messages.

    Example
    -------
    >>> prompts = PromptLibrary()
    >>> prompts.build("query_chunk_coverage", subquery="Who wrote it?", chunk="...")
    [{'role': 'system', ...}, {'role': 'user', ...}]
    """

    def __init__(self, path: str = DEFAULT_PROMPT_FILE):
        self.path = path
        with open(path, "r", encoding="utf-8") as f:
            self._templates: Dict[str, Dict[str, str]] = yaml.safe_load(f)

    def __contains__(self, name: str) -> bool:
        return name in self._templates

    def names(self) -> List[str]:
        return list(self._templates)

    def build(self, name: str, **fields) -> List[Dict[str, str]]:
        """Render a task's prompt into a chat-message list."""
        try:
            template = self._templates[name]
        except KeyError:
            raise KeyError(
                f"No prompt named {name!r} in {self.path}. "
                f"Available: {', '.join(self.names())}"
            ) from None
        return [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": _fill(template["user"], fields)},
        ]


def _fill(template: str, fields: Dict[str, str]) -> str:
    """Substitute ``{name}`` placeholders.

    Only the named fields are replaced, so literal braces in the template — the
    JSON examples the prompts show the model — are left untouched.
    """
    rendered = template
    for name, value in fields.items():
        rendered = rendered.replace("{" + name + "}", str(value))
    return rendered
