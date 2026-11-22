
"""Регресійні тести безпеки рішень про реліз."""

import asyncio

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

import agent


SOURCES = [
    "get_service_status",
    "get_test_results",
    "check_release_policy",
]


class FakeSafety:
    """Заглушка контролера безпеки для тестів."""

    def before_tool(self, name, args):
        pass


class FakeStructuredLLM:
    """Модель, що навмисно повертає непідтверджений висновок."""

    async def ainvoke(self, messages):
        return {
            "answer": "Перевірку успішно завершено.",
            "confidence": 1.0,
            "sources": [],
        }


def release_results(
    service="api",
    version="1.4.0",
    approved=True,
):
    """Побудувати тестові результати трьох інструментів."""

    healthy = service == "api"

    return [
        {
            "tool": "get_service_status",
            "result": {
                "service": service,
                "status": "healthy" if healthy else "degraded",
            },
        },
        {
            "tool": "get_test_results",
            "result": {
                "service": service,
                "version": version,
                "failed": 0 if healthy else 4,
                "coverage_pct": 94.0 if healthy else 87.0,
            },
        },
        {
            "tool": "check_release_policy",
            "result": {
                "service": service,
                "approved": approved,
                "reasons": [] if approved else [
                    "Є провалені тести",
                ],
            },
        },
    ]


def test_rejects_tool_call_with_wrong_version():
    """Виклик get_test_results має відповідати запитаній версії."""

    state = {
        "messages": [
            HumanMessage(
                content="Чи можна випускати реліз api версії 9.9.9?"
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_test_results",
                        "args": {
                            "service": "api",
                            "version": "1.4.0",
                        },
                        "id": "wrong-version",
                        "type": "tool_call",
                    }
                ],
            ),
        ],
        "step_count": 1,
    }

    result = asyncio.run(
        agent.guarded_tools_node(state, FakeSafety())
    )

    assert len(result["messages"]) == 1

    message = result["messages"][0]

    assert isinstance(message, ToolMessage)
    assert message.status == "error"
    assert "9.9.9" in message.content


def test_release_cannot_use_another_version():
    """Дані іншої версії не можуть дозволити реліз."""

    result = agent.build_grounded_release_answer(
        "Чи можна випускати реліз api версії 9.9.9?",
        release_results(version="1.4.0"),
        SOURCES,
    )

    assert result is not None
    assert result.confidence == 0.0
    assert "дозволено" not in result.answer.lower()


def test_tool_error_cannot_produce_confident_answer(monkeypatch):
    """Помилка інструмента не підтверджує результат перевірки."""

    monkeypatch.setattr(
        agent,
        "structured_llm",
        FakeStructuredLLM(),
    )

    state = {
        "messages": [
            HumanMessage(
                content="Перевір результати тестів api версії 9.9.9"
            ),
            AIMessage(content="Результатів не знайдено."),
            ToolMessage(
                content='{"error": "Результати тестів не знайдено"}',
                tool_call_id="missing-tests",
                name="get_test_results",
                status="success",
            ),
        ],
        "step_count": 1,
    }

    result = asyncio.run(agent.formatter_node(state))
    final = result["final_answer"]

    assert final["confidence"] == 0.0
    assert final["sources"] == []
    assert "успішно завершено" not in final["answer"].lower()


def test_valid_release_remains_allowed():
    """Коректний реліз api 1.4.0 залишається дозволеним."""

    result = agent.build_grounded_release_answer(
        "Чи можна випускати реліз api версії 1.4.0?",
        release_results(),
        SOURCES,
    )

    assert result is not None
    assert result.confidence == 1.0
    assert "дозволено" in result.answer.lower()


def test_invalid_payments_release_remains_blocked():
    """Реліз payments з невдалими тестами залишається забороненим."""

    result = agent.build_grounded_release_answer(
        "Чи можна випускати реліз payments версії 2.1.0?",
        release_results(
            service="payments",
            version="2.1.0",
            approved=False,
        ),
        SOURCES,
    )

    assert result is not None
    assert result.confidence == 1.0
    assert "заборонено" in result.answer.lower()
