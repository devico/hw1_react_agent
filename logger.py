
"""Структуроване JSON-логування траєкторії ReAct-агента.

Логер зберігає:
- повний вхід і вихід кожного вузла;
- виклики інструментів та їхні аргументи;
- часові мітки;
- тривалість виконання вузлів;
- загальну тривалість роботи агента.

Вкладені словники та списки залишаються JSON-структурами.
Текст не обрізається.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    """Збирає та зберігає структуровану траєкторію агента."""

    def __init__(self, query: str):
        self.query = query
        self.steps: list[dict[str, Any]] = []
        self.started_at = time.monotonic()

    # ========================================================
    # 1. SERIALIZATION
    # ========================================================

    @staticmethod
    def serialize(value: Any) -> Any:
        """Перетворити значення на JSON-сумісну структуру.

        Важливо:
        - рядки зберігаються повністю;
        - словники залишаються словниками;
        - списки залишаються списками;
        - вкладені значення не обрізаються;
        - Unicode зберігається без ASCII-екранування.

        Для нестандартних Python-об'єктів використовується
        рядкове представлення через default=str.
        """

        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        )

    # ========================================================
    # 2. RECORD STEP
    # ========================================================

    def log_step(
        self,
        node_name: str,
        input_data: Any,
        output_data: Any,
        duration_ms: float,
        tool_calls: list[dict] | None = None,
    ) -> None:
        """Записати один завершений крок виконання графа."""

        elapsed_ms = (
            time.monotonic() - self.started_at
        ) * 1000

        step = {
            "step_number": len(self.steps) + 1,
            "node_name": node_name,
            "input": self.serialize(input_data),
            "output": self.serialize(output_data),
            "tool_calls": self.serialize(
                tool_calls if tool_calls is not None else []
            ),
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "duration_ms": round(
                duration_ms,
                2,
            ),
            "elapsed_ms": round(
                elapsed_ms,
                2,
            ),
        }

        self.steps.append(step)

    # ========================================================
    # 3. BUILD TRAJECTORY
    # ========================================================

    def to_dict(self) -> dict[str, Any]:
        """Повернути повну траєкторію як словник."""

        total_time_ms = (
            time.monotonic() - self.started_at
        ) * 1000

        return {
            "query": self.query,
            "total_steps": len(self.steps),
            "total_time_ms": round(
                total_time_ms,
                2,
            ),
            "trajectory": self.steps,
        }

    # ========================================================
    # 4. SAVE JSON
    # ========================================================

    def save(
        self,
        filepath: str | Path = "trajectory.json",
    ) -> None:
        """Зберегти траєкторію у файл UTF-8 JSON.

        За потреби автоматично створює директорії.
        """

        path = Path(filepath)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = self.to_dict()

        with path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )
