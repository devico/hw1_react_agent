
"""Захисні механізми ReAct-агента.

Реалізує:
- обмеження кількості ітерацій агента;
- контроль часу через monotonic clock;
- детекцію повторюваних tool calls.
"""

import json
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SafetyConfig:
    """Конфігурація захисних механізмів."""

    max_steps: int = 10
    timeout_seconds: float = 120.0
    max_repeats: int = 3

    def __post_init__(self):
        """Перевірити правильність конфігурації."""
        if self.max_steps < 1:
            raise ValueError("max_steps має бути >= 1")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds має бути > 0")

        if self.max_repeats < 2:
            raise ValueError("max_repeats має бути >= 2")


class SafetyStop(Exception):
    """Сигнал про необхідність безпечної зупинки агента."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        super().__init__(message)


class LoopDetector:
    """Виявляє однакові послідовні виклики інструментів."""

    def __init__(self, max_repeats: int = 3):
        if max_repeats < 2:
            raise ValueError("max_repeats має бути >= 2")

        self.max_repeats = max_repeats
        self.last_call = None
        self.repeat_count = 0

    def check(self, tool_name: str, args: dict) -> bool:
        """Повернути True, якщо виявлено зациклення.

        Аргументи нормалізуються, щоб порядок ключів
        JSON не впливав на порівняння викликів.
        """

        normalized_args = json.dumps(
            args,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        current_call = (tool_name, normalized_args)

        if current_call == self.last_call:
            self.repeat_count += 1
        else:
            self.last_call = current_call
            self.repeat_count = 1

        return self.repeat_count >= self.max_repeats


class SafetyController:
    """Контролює обмеження одного запуску агента.

    Для кожного нового запиту створюється окремий
    екземпляр SafetyController.
    """

    def __init__(
        self,
        config: SafetyConfig | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.config = config or SafetyConfig()

        # Можливість підмінити годинник спрощує unit-тести
        self.clock = clock or time.monotonic
        self.started_at = self.clock()

        self.loop_detector = LoopDetector(
            max_repeats=self.config.max_repeats
        )

    @property
    def elapsed_seconds(self) -> float:
        """Час від початку виконання агента."""
        return self.clock() - self.started_at

    def check_timeout(self) -> None:
        """Перевірити перевищення загального часу."""

        if self.elapsed_seconds >= self.config.timeout_seconds:
            raise SafetyStop(
                reason="timeout",
                message=(
                    "Перевищено допустимий час виконання "
                    f"({self.config.timeout_seconds} с)."
                ),
            )

    def before_agent(self, completed_steps: int) -> None:
        """Перевірити обмеження перед наступним викликом LLM."""

        self.check_timeout()

        if completed_steps >= self.config.max_steps:
            raise SafetyStop(
                reason="max_steps",
                message=(
                    "Досягнуто ліміт ітерацій агента "
                    f"({self.config.max_steps})."
                ),
            )

    def before_tool(self, name: str, args: dict) -> None:
        """Перевірити виклик інструмента перед виконанням."""

        self.check_timeout()

        if self.loop_detector.check(name, args):
            raise SafetyStop(
                reason="loop_detected",
                message=(
                    "Виявлено зациклення: інструмент "
                    f"{name} викликається з однаковими "
                    "аргументами повторно."
                ),
            )
