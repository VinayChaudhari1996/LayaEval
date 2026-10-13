"""Dataset loader and container for evaluation suites."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalExample:
    """A single evaluation example."""

    state: dict[str, Any]
    metrics: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    id: str | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> EvalExample:
        state = row.get("state")
        if not isinstance(state, dict):
            # If row doesn't have a nested "state", the row itself can serve as state
            state = {
                k: v for k, v in row.items() if k not in ("metrics", "tags", "thresholds", "id")
            }
        metrics = row.get("metrics") or []
        if isinstance(metrics, str):
            metrics = [metrics]
        return cls(
            state=state,
            metrics=list(metrics),
            tags=list(row.get("tags") or []),
            thresholds=dict(row.get("thresholds") or {}),
            id=row.get("id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "metrics": self.metrics,
            "tags": self.tags,
            "thresholds": self.thresholds,
        }


class EvalDataset:
    """An ordered dataset of evaluation examples."""

    def __init__(self, examples: Iterable[EvalExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[EvalExample]:
        return iter(self.examples)

    def __getitem__(self, index: int) -> EvalExample:
        return self.examples[index]

    @classmethod
    def from_jsonl(cls, path: str) -> EvalDataset:
        """Load evaluation examples from a JSON Lines file."""
        examples: list[EvalExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Line {line_no} in {path} is not valid JSON: {exc}")
                examples.append(EvalExample.from_dict(data))
        return cls(examples)

    def to_jsonl(self, path: str) -> None:
        """Write examples out to a JSON Lines file."""
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(
                json.dumps(ex.to_dict(), ensure_ascii=False) + "\n" for ex in self.examples
            )
