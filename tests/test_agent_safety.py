
"""Інтеграційні тести захисту асинхронного LangGraph."""

import asyncio
import json

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

import agent
from safety import SafetyConfig


# ============================================================
# 1. MAX STEPS
# ============================================================

def test_graph_max_steps_returns_partial_answer():
    """Ліміт ітерацій зупиняє граф без LLM-виклику."""

    graph = agent.build_graph(
        SafetyConfig(max_steps=1)
    )

    result = asyncio.run(
        graph.ainvoke({
            "messages": [
                HumanMessage(
                    content="Перевір реліз api 1.4.0"
                )
            ],
            "step_count": 1,
        })
    )

    assert result["safety_stop"] == "max_steps"
    assert result["step_count"] == 1
    assert result["final_answer"]["confidence"] == 0.0


# ============================================================
# 2. LOOP DETECTION
# ============================================================

class RepeatingLLM:
    """Тестова модель, яка повторює один tool call."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1

        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_service_status",
                    "args": {"service": "api"},
                    "id": f"repeat-{self.calls}",
                    "type": "tool_call",
                }
            ],
        )


def test_graph_loop_detection(monkeypatch):
    """Третій однаковий виклик зупиняє граф."""

    fake_llm = RepeatingLLM()

    monkeypatch.setattr(
        agent,
        "llm_with_tools",
        fake_llm,
    )

    graph = agent.build_graph(
        SafetyConfig(
            max_steps=10,
            max_repeats=3,
        )
    )

    result = asyncio.run(
        graph.ainvoke({
            "messages": [
                HumanMessage(
                    content="Перевір статус api"
                )
            ],
            "step_count": 0,
        })
    )

    assert result["safety_stop"] == "loop_detected"
    assert fake_llm.calls == 3
    assert result["step_count"] == 3

    tool_messages = [
        msg
        for msg in result["messages"]
        if isinstance(msg, ToolMessage)
    ]

    assert len(tool_messages) == 3

    assert tool_messages[0].status == "success"
    assert tool_messages[1].status == "success"
    assert tool_messages[2].status == "error"

    assert result["final_answer"]["confidence"] == 0.0


# ============================================================
# 3. TIMEOUT DURING LLM CALL
# ============================================================

class SlowLLM:
    """Модель, яка зависає під час відповіді."""

    def __init__(self):
        self.entered = False
        self.cancelled = False

    async def ainvoke(self, messages):
        self.entered = True

        try:
            await asyncio.sleep(10)

        except asyncio.CancelledError:
            self.cancelled = True
            raise

        return AIMessage(
            content="Пізня відповідь"
        )


def test_timeout_during_llm(monkeypatch, tmp_path):
    """Timeout скасовує LLM та зберігає лог окремо.

    Використовуємо tmp_path, щоб тест не змінював
    кореневий trajectory.json проєкту.
    """

    fake_llm = SlowLLM()

    monkeypatch.setattr(
        agent,
        "llm_with_tools",
        fake_llm,
    )

    trajectory_path = tmp_path / "timeout_llm.json"

    result = asyncio.run(
        agent.execute_query(
            query="Перевір реліз api",
            timeout_seconds=0.3,
            trajectory_path=str(trajectory_path),
        )
    )

    assert fake_llm.entered is True
    assert fake_llm.cancelled is True

    assert result["safety_stop"] == "timeout"
    assert result["final_answer"]["confidence"] == 0.0
    assert result["final_answer"]["sources"] == []

    # JSON-лог зберігається у тимчасовій директорії.
    assert trajectory_path.is_file()

    with trajectory_path.open(encoding="utf-8") as file:
        trajectory = json.load(file)

    assert trajectory["query"] == "Перевір реліз api"
    assert trajectory["total_steps"] >= 1

    node_names = [
        step["node_name"]
        for step in trajectory["trajectory"]
    ]

    assert "safety" in node_names
    assert "formatter" in node_names


# ============================================================
# 4. TIMEOUT AFTER SUCCESSFUL TOOL CALL
# ============================================================

class OneToolThenSlowLLM:
    """Спочатку викликає інструмент, потім зависає."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1

        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_service_status",
                        "args": {"service": "api"},
                        "id": "first-status",
                        "type": "tool_call",
                    }
                ],
            )

        await asyncio.sleep(10)

        return AIMessage(
            content="Запізніла відповідь"
        )


def test_timeout_preserves_observations(
    monkeypatch,
    tmp_path,
):
    """Після timeout зберігаються вже отримані факти.

    Лог записується лише в тимчасовий файл тесту.
    """

    fake_llm = OneToolThenSlowLLM()

    monkeypatch.setattr(
        agent,
        "llm_with_tools",
        fake_llm,
    )

    trajectory_path = (
        tmp_path / "timeout_observations.json"
    )

    result = asyncio.run(
        agent.execute_query(
            query="Перевір стан сервісу api",
            timeout_seconds=0.8,
            trajectory_path=str(trajectory_path),
        )
    )

    assert fake_llm.calls == 2
    assert result["safety_stop"] == "timeout"

    # Перший інструмент завершився до timeout.
    assert result["final_answer"]["sources"] == [
        "get_service_status"
    ]

    assert result["final_answer"]["confidence"] == 0.0

    assert "healthy" in result["final_answer"]["answer"]

    # Перевіряємо окремий JSON-файл траєкторії.
    assert trajectory_path.is_file()

    with trajectory_path.open(encoding="utf-8") as file:
        trajectory = json.load(file)

    assert trajectory["query"] == (
        "Перевір стан сервісу api"
    )

    node_names = [
        step["node_name"]
        for step in trajectory["trajectory"]
    ]

    assert "agent" in node_names
    assert "tools" in node_names
    assert "safety" in node_names
    assert "formatter" in node_names
