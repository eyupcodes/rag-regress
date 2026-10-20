"""Golden set: the fixed questions a pipeline is checked against.

One JSON object per line::

    {"id": "q1", "query": "...", "relevant_ids": ["doc-3"], "reference_answer": "..."}

``reference_answer`` and ``tags`` are optional. Blank lines are ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class GoldenSetError(ValueError):
    """The golden set file cannot be used; the message names file and line."""


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    query: str
    relevant_ids: tuple[str, ...] = Field(min_length=1)
    reference_answer: str | None = None
    tags: tuple[str, ...] = ()

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("relevant_ids")
    @classmethod
    def _ids_unique_and_not_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not doc_id.strip() for doc_id in value):
            raise ValueError("relevant_ids must not contain blank ids")
        if len(set(value)) != len(value):
            raise ValueError("relevant_ids must not repeat an id")
        return value


def load_golden_set(path: str | Path) -> list[GoldenItem]:
    """Read and validate a golden set; fail on the first bad line."""
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise GoldenSetError(f"{path}: file not found") from None
    except UnicodeDecodeError:
        raise GoldenSetError(f"{path}: not valid UTF-8 text") from None
    except OSError as exc:
        raise GoldenSetError(f"{path}: cannot read ({exc.strerror or exc})") from None

    items: list[GoldenItem] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenSetError(f"{where}: invalid JSON ({exc.msg})") from None
        if not isinstance(raw, dict):
            raise GoldenSetError(f"{where}: expected a JSON object")
        try:
            item = GoldenItem.model_validate(raw)
        except ValidationError as exc:
            first = exc.errors()[0]
            field = ".".join(str(part) for part in first["loc"]) or "item"
            raise GoldenSetError(f"{where}: {field}: {first['msg']}") from None
        if item.id in seen:
            first_line = seen[item.id]
            raise GoldenSetError(
                f"{where}: duplicate id {item.id!r} (first seen on line {first_line})"
            )
        seen[item.id] = number
        items.append(item)

    if not items:
        raise GoldenSetError(f"{path}: golden set is empty")
    return items
