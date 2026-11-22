
"""Регресійні тести відновлення ReAct-агента після помилок."""

import asyncio

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

import agent
from schemas import FinalAnswer


class ScriptedRecoveryLLM:
    """Відтворює неправильний tool call та його виправлення."""

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1

        if self.calls == 1:
            # Помилка, яку реально створила Qwen.
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_service_status",
                        "args": {
                            "service": {
                                "description": "api",
                                "type": "string",
                            }
                        },
                        "id": "bad-status",
                        "type": "tool_call",
                    }
                ],
            )

        if self.calls == 2:
            # Модель помилково хоче завершити роботу.
            return AIMessage(
                content=(
                    "Некоректний параметр. "
                    "Спробуйте ще раз."
                )
            )

        if self.calls == 3:
            # Після додаткової інструкції модель
            # виправляє виклик інструмента.
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_service_status",
                        "args": {"service": "api"},
                        "id": "fixed-status",
                        "type": "tool_call",
                    }
                ],
            )

        return AIMessage(
            content="Перевірку стану api завершено."
        )


class StaticStructuredLLM:
    """Не використовує Ollama для форматування."""

    async def ainvoke(self, messages):
        return FinalAnswer(
            answer="Немає підтвердженого результату.",
            confidence=0.0,
            sources=[],
        )


class InvalidStructuredLLM:
    """Імітує помилку confidence=50."""

    async def ainvoke(self, messages):
        raise OutputParserException(
            "FinalAnswer validation failed: "
            "confidence=50 exceeds maximum 1"
        )


def test_agent_recovers_from_invalid_tool_arguments(
    monkeypatch,
    tmp_path,
):
    """Агент повторює некоректний виклик, а не здається."""

    fake_llm = ScriptedRecoveryLLM()

    monkeypatch.setattr(
        agent,
        "llm_with_tools",
        fake_llm,
    )

    monkeypatch.setattr(
        agent,
        "structured_llm",
        StaticStructuredLLM(),
    )

    result = asyncio.run(
        agent.execute_query(
            query="Перевір поточний стан сервісу api.",
            max_steps=10,
            timeout_seconds=10,
            trajectory_path=str(
                tmp_path / "recovery.json"
            ),
        )
    )

    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.name == "get_service_status"
    ]

    # Перший виклик має бути відхилений.
    assert any(
        message.status == "error"
        for message in tool_messages
    )

    # Після помилки агент має виконати
    # виправлений виклик.
    assert any(
        message.status == "success"
        for message in tool_messages
    )

    # Для виправлення потрібен третій виклик моделі.
    assert fake_llm.calls >= 3

    final = FinalAnswer.model_validate(
        result["final_answer"]
    )

    assert "api" in final.answer
    assert "healthy" in final.answer
    assert "get_service_status" in final.sources

    assert result.get("safety_stop") is None


def test_formatter_recovers_from_invalid_structured_output(
    monkeypatch,
):
    """Помилка парсингу не повинна завершувати граф винятком."""

    monkeypatch.setattr(
        agent,
        "structured_llm",
        InvalidStructuredLLM(),
    )

    state = {
        "messages": [
            HumanMessage(
                content="Поясни поточну ситуацію."
            ),
            AIMessage(
                content="Недостатньо інформації."
            ),
        ],
        "step_count": 1,
    }

    result = asyncio.run(
        agent.formatter_node(state)
    )

    final = FinalAnswer.model_validate(
        result["final_answer"]
    )

    # Без підтверджених даних не можна заявляти
    # високу впевненість.
    assert final.confidence == 0.0
    assert final.sources == []

    # Має бути осмислене повідомлення,
    # а не необроблений exception.
    assert final.answer.strip()
