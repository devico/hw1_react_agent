
"""Тести JSON-логування траєкторії ReAct-агента."""

import asyncio
import json
from datetime import datetime

from langchain_core.messages import AIMessage, ToolMessage

import agent
from logger import TrajectoryLogger


# ============================================================
# 1. JSON SERIALIZATION
# ============================================================

def test_logger_saves_json(tmp_path):
    """Перевірити створення JSON-файлу з усіма полями."""

    logger = TrajectoryLogger("Перевір статус api")

    logger.log_step(
        node_name="agent",
        input_data={"query": "Перевір статус api"},
        output_data={"action": "get_service_status"},
        duration_ms=125.5,
        tool_calls=[
            {
                "name": "get_service_status",
                "args": {"service": "api"},
            }
        ],
    )

    filepath = tmp_path / "trajectory.json"
    logger.save(str(filepath))

    assert filepath.exists()

    with filepath.open(encoding="utf-8") as file:
        data = json.load(file)

    assert data["query"] == "Перевір статус api"
    assert data["total_steps"] == 1
    assert data["total_time_ms"] >= 0

    step = data["trajectory"][0]

    assert step["step_number"] == 1
    assert step["node_name"] == "agent"
    assert step["duration_ms"] == 125.5

    assert step["tool_calls"][0]["name"] == (
        "get_service_status"
    )

    # Timestamp повинен містити часову зону.
    timestamp = datetime.fromisoformat(step["timestamp"])
    assert timestamp.tzinfo is not None


# ============================================================
# 2. UNICODE AND TRUNCATION
# ============================================================

def test_logger_unicode_and_truncation():
    """Перевірити кирилицю та обмеження довжини полів."""

    logger = TrajectoryLogger("Український текст")

    logger.log_step(
        node_name="tools",
        input_data="А" * 1000,
        output_data="Сервіс працює",
        duration_ms=10,
    )

    data = logger.to_dict()
    step = data["trajectory"][0]

    assert len(step["input"]) == 500
    assert step["output"] == "Сервіс працює"
    assert step["tool_calls"] == []

    assert data["query"] == "Український текст"
    assert data["total_steps"] == 1


# ============================================================
# 3. DETERMINISTIC TEST LLM
# ============================================================

class ScriptedLLM:
    """Тестова LLM із заздалегідь визначеними діями.

    Дозволяє перевіряти реальний LangGraph
    без викликів зовнішньої або локальної моделі.
    """

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, messages):
        """Повернути наступну дію тестового агента."""

        self.calls += 1

        # Перший виклик: отримуємо статус і тести.
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_service_status",
                        "args": {"service": "api"},
                        "id": "status-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "get_test_results",
                        "args": {
                            "service": "api",
                            "version": "1.4.0",
                        },
                        "id": "tests-1",
                        "type": "tool_call",
                    },
                ],
            )

        # Другий виклик: перевірка політики релізу.
        if self.calls == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "check_release_policy",
                        "args": {
                            "service": "api",
                            "status": "healthy",
                            "failed_tests": 0,
                            "coverage_pct": 94.0,
                        },
                        "id": "policy-1",
                        "type": "tool_call",
                    }
                ],
            )

        # Третій виклик: фінальний висновок.
        return AIMessage(
            content="Перевірку завершено."
        )


# ============================================================
# 4. LANGGRAPH INTEGRATION TEST
# ============================================================

def test_real_graph_trajectory(monkeypatch, tmp_path):
    """Перевірити логування реальних вузлів LangGraph.

    Тест використовує:
    - справжній StateGraph;
    - справжні інструменти;
    - справжній ToolNode;
    - підмінну детерміновану LLM.
    """

    fake_llm = ScriptedLLM()

    monkeypatch.setattr(
        agent,
        "llm_with_tools",
        fake_llm,
    )

    filepath = tmp_path / "trajectory.json"

    result = asyncio.run(
        agent.execute_query(
            query="Чи можна випускати реліз api версії 1.4.0?",
            timeout_seconds=10,
            trajectory_path=str(filepath),
        )
    )

    # --------------------------------------------------------
    # Перевірка створення файлу
    # --------------------------------------------------------

    assert filepath.exists()
    assert fake_llm.calls == 3

    with filepath.open(encoding="utf-8") as file:
        data = json.load(file)

    # --------------------------------------------------------
    # Перевірка послідовності вузлів
    # --------------------------------------------------------

    nodes = [
        step["node_name"]
        for step in data["trajectory"]
    ]

    assert nodes == [
        "agent",
        "tools",
        "agent",
        "tools",
        "agent",
        "formatter",
    ]

    assert data["total_steps"] == 6

    # --------------------------------------------------------
    # Перевірка структури кожного запису
    # --------------------------------------------------------

    required_fields = {
        "step_number",
        "node_name",
        "input",
        "output",
        "tool_calls",
        "timestamp",
        "duration_ms",
        "elapsed_ms",
    }

    for index, step in enumerate(data["trajectory"], start=1):
        assert required_fields.issubset(step)

        assert step["step_number"] == index
        assert step["duration_ms"] >= 0
        assert step["elapsed_ms"] >= 0

        timestamp = datetime.fromisoformat(
            step["timestamp"]
        )

        assert timestamp.tzinfo is not None

    # --------------------------------------------------------
    # Перевірка фактичних викликів інструментів
    # --------------------------------------------------------

    tool_names = [
        call["name"]
        for step in data["trajectory"]
        for call in step["tool_calls"]
    ]

    assert "get_service_status" in tool_names
    assert "get_test_results" in tool_names
    assert "check_release_policy" in tool_names

    # Перевіряємо, що інструменти реально виконалися.
    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    assert len(tool_messages) == 3

    assert all(
        message.status == "success"
        for message in tool_messages
    )

    # --------------------------------------------------------
    # Перевірка фінального результату
    # --------------------------------------------------------

    assert result.get("safety_stop") is None

    final_answer = result["final_answer"]

    assert "дозволено" in final_answer["answer"]

    assert final_answer["sources"] == [
        "get_service_status",
        "get_test_results",
        "check_release_policy",
    ]

    assert final_answer["confidence"] == 1.0
