
"""Структуроване JSON-логування траєкторії ReAct-агента."""

import json
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    """Збирає інформацію про виконання одного запиту агента.

    Зберігає:
    - номер кроку;
    - назву вузла LangGraph;
    - вхідні та вихідні дані;
    - виклики інструментів;
    - часові показники.
    """

    def __init__(self, query: str):
        self.query = query
        self.steps: list[dict] = []
        self.started_at = time.monotonic()

    @staticmethod
    def serialize(value: Any) -> str:
        """Перетворити значення на текст для JSON-логу."""

        if isinstance(value, str):
            return value[:500]

        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )[:500]

    def log_step(
        self,
        node_name: str,
        input_data: Any,
        output_data: Any,
        duration_ms: float,
        tool_calls: list[dict] | None = None,
    ) -> None:
        """Записати завершений крок виконання графа."""

        step = {
            "step_number": len(self.steps) + 1,
            "node_name": node_name,
            "input": self.serialize(input_data),
            "output": self.serialize(output_data),
            "tool_calls": tool_calls or [],
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "duration_ms": round(duration_ms, 2),
            "elapsed_ms": round(
                (time.monotonic() - self.started_at) * 1000,
                2,
            ),
        }

        self.steps.append(step)

    def to_dict(self) -> dict:
        """Повернути повну траєкторію як словник."""

        return {
            "query": self.query,
            "total_steps": len(self.steps),
            "total_time_ms": round(
                (time.monotonic() - self.started_at) * 1000,
                2,
            ),
            "trajectory": self.steps,
        }

    def save(self, filepath: str = "trajectory.json") -> None:
        """Зберегти траєкторію у JSON-файл."""

        path = Path(filepath)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                self.to_dict(),
                file,
                ensure_ascii=False,
                indent=2,
            )
