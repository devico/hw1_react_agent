
"""Інтеграційні тести захисту асинхронного LangGraph."""

import asyncio

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

import agent
from safety import SafetyConfig


def test_graph_max_steps_returns_partial_answer():
    """Ліміт ітерацій зупиняє граф без LLM-виклику."""

    graph = agent.build_graph(
        SafetyConfig(max_steps=1)
    )

    result = asyncio.run(graph.ainvoke({
        "messages": [
            HumanMessage(content="Перевір реліз api 1.4.0")
        ],
        "step_count": 1,
    }))

    assert result["safety_stop"] == "max_steps"
    assert result["step_count"] == 1
    assert result["final_answer"]["confidence"] == 0.0


class RepeatingLLM:
    """Тестова модель, яка повторює один tool call."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1

        return AIMessage(
            content="",
            tool_calls=[{
                "name": "get_service_status",
                "args": {"service": "api"},
                "id": f"repeat-{self.calls}",
                "type": "tool_call",
            }],
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
        SafetyConfig(max_steps=10, max_repeats=3)
    )

    result = asyncio.run(graph.ainvoke({
        "messages": [
            HumanMessage(content="Перевір статус api")
        ],
        "step_count": 0,
    }))

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

        return AIMessage(content="Пізня відповідь")


def test_timeout_during_llm(monkeypatch):
    """Загальний timeout скасовує очікування LLM."""

    fake_llm = SlowLLM()

    monkeypatch.setattr(
        agent,
        "llm_with_tools",
        fake_llm,
    )

    result = asyncio.run(agent.execute_query(
        query="Перевір реліз api",
        timeout_seconds=0.3,
    ))

    assert fake_llm.entered is True
    assert fake_llm.cancelled is True
    assert result["safety_stop"] == "timeout"
    assert result["final_answer"]["confidence"] == 0.0
    assert result["final_answer"]["sources"] == []


class OneToolThenSlowLLM:
    """Спочатку викликає tool, потім зависає."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1

        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_service_status",
                    "args": {"service": "api"},
                    "id": "first-status",
                    "type": "tool_call",
                }],
            )

        await asyncio.sleep(10)
        return AIMessage(content="Запізніла відповідь")


def test_timeout_preserves_observations(monkeypatch):
    """Після timeout зберігаються вже отримані факти."""

    fake_llm = OneToolThenSlowLLM()

    monkeypatch.setattr(
        agent,
        "llm_with_tools",
        fake_llm,
    )

    result = asyncio.run(agent.execute_query(
        query="Перевір стан сервісу api",
        timeout_seconds=0.8,
    ))

    assert fake_llm.calls == 2
    assert result["safety_stop"] == "timeout"

    # Перший інструмент завершився до спрацювання timeout.
    assert result["final_answer"]["sources"] == [
        "get_service_status"
    ]
    assert result["final_answer"]["confidence"] == 0.0
    assert "healthy" in result["final_answer"]["answer"]
