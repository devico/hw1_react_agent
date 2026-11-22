
"""ReAct-агент для перевірки сервісів і демонстраційних релізів."""

import asyncio
import json
import re
import time
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from logger import TrajectoryLogger
from safety import SafetyConfig, SafetyController, SafetyStop
from schemas import FinalAnswer
from tools import TOOLS


# ============================================================
# 1. PROMPTS
# ============================================================

SYSTEM_PROMPT = """
Ти DevOps-асистент для демонстраційних сервісів api, payments, worker.

Використовуй лише інструменти, потрібні для запиту користувача.
Стан перевіряй через get_service_status, тести — через get_test_results,
історію розгортань — через get_deployment_history.
Для рішення про реліз спочатку отримай стан і результати тестів,
а потім викликай check_release_policy з їхніми фактичними значеннями.

Не вигадуй метрики, версії та аргументи. Не роби незапитаних
висновків про дозвіл релізу. Дані демонстраційні, не production.

Аргументи інструментів передавай як значення, а не як JSON Schema:
правильно: {"service": "api"};
неправильно: {"service": {"description": "api", "type": "string"}}.
Після помилки валідації самостійно виправ аргументи і повтори
інструмент; не проси користувача повторно ввести вже наявні дані.

Відповідай українською мовою.
"""

FORMATTER_PROMPT = """
Сформуй структуровану відповідь українською мовою з фактичних даних.
Не вигадуй метрики чи джерела, не роби незапитаних рішень про реліз.
Поверни answer, confidence (число від 0 до 1) і sources.
"""


# ============================================================
# 2. STATE
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    step_count: int
    final_answer: NotRequired[dict]
    safety_stop: NotRequired[str]
    safety_message: NotRequired[str]


# ============================================================
# 3. LLM AND TOOLS
# ============================================================

llm = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)

llm_with_tools = llm.bind_tools(TOOLS)
_default_llm_with_tools = llm_with_tools

structured_llm = llm.with_structured_output(
    FinalAnswer,
    method="json_schema",
)

tool_executor = ToolNode(TOOLS)


# ============================================================
# 4. QUERY CLASSIFICATION
# ============================================================

def user_query_from_state(state: AgentState) -> str:
    return next(
        (
            str(msg.content)
            for msg in state.get("messages", [])
            if isinstance(msg, HumanMessage)
        ),
        "",
    )


def is_release_decision_query(query: str) -> bool:
    text = query.casefold()

    release_words = (
        "реліз",
        "релиз",
        "release",
    )

    decision_words = (
        "можна",
        "можно",
        "випуск",
        "выпуск",
        "дозвол",
        "разреш",
        "готов",
        "approve",
        "deploy",
    )

    return (
        any(word in text for word in release_words)
        and any(word in text for word in decision_words)
    )


def requested_version(query: str) -> str | None:
    """Визначити єдину явно запитану семантичну версію."""

    versions = set(
        re.findall(
            r"(?<![\w.])\d+\.\d+\.\d+(?![\w.])",
            query,
        )
    )

    if len(versions) != 1:
        return None

    return next(iter(versions))


def select_tools_for_query(query: str) -> list:
    """Визначити допустимі інструменти для поточного запиту."""

    text = query.casefold()
    selected = set()

    release_decision = is_release_decision_query(query)

    if release_decision:
        selected.update({
            "get_service_status",
            "get_test_results",
            "check_release_policy",
        })

    if any(
        word in text
        for word in (
            "стан",
            "статус",
            "здоров",
            "status",
            "health",
        )
    ):
        selected.add("get_service_status")

    if any(
        word in text
        for word in (
            "тест",
            "покрит",
            "coverage",
            "failed",
        )
    ):
        selected.add("get_test_results")

    if any(
        word in text
        for word in (
            "розгорт",
            "деплой",
            "депло",
            "deployment",
            "deploy",
            "історі",
        )
    ):
        selected.add("get_deployment_history")

    if not selected:
        selected.update({
            "get_service_status",
            "get_test_results",
            "get_deployment_history",
        })

    if not release_decision:
        selected.discard("check_release_policy")

    return [
        tool
        for tool in TOOLS
        if tool.name in selected
    ]


# ============================================================
# 5. ERROR RECOVERY
# ============================================================

def unresolved_failed_tools(
    state: AgentState,
) -> set[str]:
    """Знайти інструменти з помилкою без успішного повтору."""

    required = {
        tool.name
        for tool in select_tools_for_query(
            user_query_from_state(state)
        )
    }

    failed = set()
    successful = set()

    for message in state.get("messages", []):
        if not isinstance(message, ToolMessage):
            continue

        if message.name not in required:
            continue

        if message.status == "error":
            failed.add(message.name)
            continue

        try:
            data = json.loads(message.content)
        except (ValueError, TypeError):
            continue

        if (
            isinstance(data, dict)
            and "error" not in data
        ):
            successful.add(message.name)

    return failed - successful


def should_retry_failed_tools(
    state: AgentState,
) -> bool:
    """Дати LLM обмежену можливість виправити tool call."""

    failed_names = unresolved_failed_tools(state)

    if not failed_names:
        return False

    messages = state.get("messages", [])

    last_error = max(
        (
            index
            for index, msg in enumerate(messages)
            if (
                isinstance(msg, ToolMessage)
                and msg.status == "error"
                and msg.name in failed_names
            )
        ),
        default=-1,
    )

    if last_error < 0:
        return False

    non_tool_answers = sum(
        isinstance(msg, AIMessage)
        and not msg.tool_calls
        for msg in messages[last_error + 1:]
    )

    return 1 <= non_tool_answers <= 2


# ============================================================
# 6. AGENT NODE
# ============================================================

async def agent_node(
    state: AgentState,
    safety: SafetyController,
) -> dict:
    completed_steps = state.get("step_count", 0)

    try:
        safety.before_agent(completed_steps)
    except SafetyStop as error:
        return {
            "safety_stop": error.reason,
            "safety_message": str(error),
            "messages": [
                AIMessage(content=str(error))
            ],
        }

    query = user_query_from_state(state)
    allowed_tools = select_tools_for_query(query)

    allowed_names = ", ".join(
        tool.name
        for tool in allowed_tools
    )

    prompt = (
        SYSTEM_PROMPT
        + f"\nДозволені інструменти: {allowed_names}."
    )

    version = requested_version(query)

    if version:
        prompt += (
            f"\nУ запиті вказано версію {version}. "
            "Перевір тільки її."
        )

    if not is_release_decision_query(query):
        prompt += (
            "\nЗапит інформаційний: "
            "не ухвалюй рішення про реліз."
        )

    missing = unresolved_failed_tools(state)

    if missing:
        prompt += (
            "\nПопередній виклик інструмента завершився помилкою. "
            "Ще потрібні достовірні результати: "
            + ", ".join(sorted(missing))
            + ". НЕ завершуй відповідь повідомленням про помилку; "
            "повтори виклик із правильними скалярними аргументами. "
            "Наприклад, service — це рядок 'api', "
            "не опис типу поля."
        )

    # Зберігаємо сумісність із monkeypatch у unit-тестах.
    if llm_with_tools is _default_llm_with_tools:
        active_llm = llm.bind_tools(allowed_tools)
    else:
        active_llm = llm_with_tools

    response = await active_llm.ainvoke([
        SystemMessage(content=prompt),
        *state["messages"],
    ])

    safety.check_timeout()

    return {
        "messages": [response],
        "step_count": completed_steps + 1,
    }


# ============================================================
# 7. OBSERVATIONS
# ============================================================

def collect_observations(messages: list) -> dict:
    observations = {}

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        if message.status == "error":
            continue

        if message.name not in {
            "get_service_status",
            "get_test_results",
        }:
            continue

        try:
            data = json.loads(message.content)
        except (ValueError, TypeError):
            continue

        if (
            isinstance(data, dict)
            and "error" not in data
            and data.get("service")
        ):
            observations[
                (message.name, data["service"])
            ] = data

    return observations


# ============================================================
# 8. RELEASE POLICY GUARD
# ============================================================

def validate_release_policy_call(
    call: dict,
    observations: dict,
    query: str = "",
) -> str | None:
    """Перевірити фактичні метрики та версію релізу."""

    args = call["args"]
    service = args.get("service")

    if not isinstance(service, str):
        return (
            "Некоректний параметр service: "
            "очікується рядок, наприклад 'api'."
        )

    status = observations.get(
        ("get_service_status", service)
    )

    tests = observations.get(
        ("get_test_results", service)
    )

    if status is None or tests is None:
        return (
            "Помилка залежностей. Спочатку отримай "
            "get_service_status та get_test_results. "
            "Потім повтори check_release_policy."
        )

    # Не можна використовувати метрики іншої версії.
    version = requested_version(query)

    if (
        version is not None
        and tests.get("version") != version
    ):
        return (
            f"Некоректна версія тестів: потрібна {version}, "
            f"отримано {tests.get('version')}. "
            "Повтори get_test_results для запитаної версії."
        )

    if (
        args.get("status") != status.get("status")
        or args.get("failed_tests") != tests.get("failed")
        or args.get("coverage_pct") != tests.get("coverage_pct")
    ):
        return (
            "Некоректні аргументи release policy. "
            f"Використай status={status.get('status')}, "
            f"failed_tests={tests.get('failed')}, "
            f"coverage_pct={tests.get('coverage_pct')}."
        )

    return None


# ============================================================
# 9. GUARDED TOOLS NODE
# ============================================================

async def guarded_tools_node(
    state: AgentState,
    safety: SafetyController,
) -> dict:
    last_message = state["messages"][-1]

    if (
        not isinstance(last_message, AIMessage)
        or not last_message.tool_calls
    ):
        return {"messages": []}

    calls = last_message.tool_calls

    try:
        for call in calls:
            safety.before_tool(
                call["name"],
                call["args"],
            )
    except SafetyStop as error:
        return {
            "messages": [
                ToolMessage(
                    content=f"Виклик скасовано: {error}",
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
                for call in calls
            ],
            "safety_stop": error.reason,
            "safety_message": str(error),
        }

    query = user_query_from_state(state)

    allowed_names = {
        tool.name
        for tool in select_tools_for_query(query)
    }

    observations = collect_observations(
        state["messages"][:-1]
    )

    expected_version = requested_version(query)

    accepted = []
    rejected = {}

    for call in calls:
        name = call["name"]
        call_id = call["id"]

        reason = None

        if name not in allowed_names:
            reason = (
                f"Виклик {name} відхилено: "
                "інструмент не потрібен "
                "для поточного запиту."
            )

        # Відхиляємо неправильну версію до ToolNode.
        elif (
            name == "get_test_results"
            and expected_version is not None
        ):
            actual_version = call["args"].get("version")

            if actual_version != expected_version:
                reason = (
                    "Некоректна версія тестів: "
                    f"запитано {expected_version}, "
                    f"передано {actual_version}. "
                    "Повтори get_test_results "
                    f"для версії {expected_version}."
                )

        elif name == "check_release_policy":
            if not is_release_decision_query(query):
                reason = (
                    "check_release_policy: користувач "
                    "не запитував рішення про реліз."
                )
            else:
                reason = validate_release_policy_call(
                    call,
                    observations,
                    query,
                )

        if reason is None:
            accepted.append(call)
        else:
            rejected[call_id] = ToolMessage(
                content=reason,
                tool_call_id=call_id,
                name=name,
                status="error",
            )

    executed = []

    if accepted:
        synthetic = AIMessage(
            content="",
            tool_calls=accepted,
        )

        execution = await tool_executor.ainvoke({
            "messages": [synthetic],
        })

        executed = execution["messages"]

    responses = {
        msg.tool_call_id: msg
        for msg in executed
    }

    responses.update(rejected)

    return {
        "messages": [
            responses[call["id"]]
            for call in calls
        ]
    }


# ============================================================
# 10. GROUNDED RELEASE ANSWER
# ============================================================

def build_grounded_release_answer(
    query: str,
    tool_results: list[dict],
    sources: list[str],
) -> FinalAnswer | None:
    if not is_release_decision_query(query):
        return None

    def latest(name, service=None):
        for item in reversed(tool_results):
            if item.get("tool") != name:
                continue

            data = item.get("result")

            if (
                isinstance(data, dict)
                and (
                    service is None
                    or data.get("service") == service
                )
            ):
                return data

        return None

    def incomplete(message):
        return FinalAnswer(
            answer=message,
            confidence=0.0,
            sources=sources,
        )

    policy = latest("check_release_policy")

    if policy is None:
        return incomplete(
            "Готовність релізу не підтверджена: "
            "check_release_policy не було успішно виконано. "
            "Автоматичний дозвіл відсутній."
        )

    service = policy.get("service")

    status = latest(
        "get_service_status",
        service,
    )

    tests = latest(
        "get_test_results",
        service,
    )

    if not status or not tests:
        return incomplete(
            "Недостатньо підтверджених даних "
            "для рішення про реліз."
        )

    # Незалежна перевірка версії перед фінальним рішенням.
    expected_version = requested_version(query)

    if (
        expected_version is not None
        and tests.get("version") != expected_version
    ):
        return incomplete(
            f"Рішення про реліз версії {expected_version} "
            "не підтверджено: результати тестів "
            f"належать версії {tests.get('version')}."
        )

    failed = tests.get("failed")
    coverage = tests.get("coverage_pct")
    health = status.get("status")

    if (
        not isinstance(failed, int)
        or isinstance(failed, bool)
        or not isinstance(coverage, (int, float))
        or isinstance(coverage, bool)
        or health not in {"healthy", "degraded"}
        or type(policy.get("approved")) is not bool
    ):
        return incomplete(
            "Некоректні дані інструментів: "
            "рішення про реліз не підтверджено."
        )

    approved = (
        health == "healthy"
        and failed == 0
        and coverage >= 90
    )

    if approved != policy["approved"]:
        return incomplete(
            "Результати інструментів суперечать "
            "рішенню політики релізу."
        )

    version = tests.get(
        "version",
        "невідомої версії",
    )

    facts = (
        f"Сервіс {service}: статус {health}, "
        f"провалених тестів {failed}, "
        f"покриття тестами {coverage}%."
    )

    if approved:
        answer = (
            f"Реліз {service} версії {version} дозволено "
            "згідно з демонстраційною політикою. "
            f"{facts} Усі критерії виконані."
        )
    else:
        reasons = "; ".join(
            str(item)
            for item in policy.get("reasons", [])
        )

        answer = (
            f"Реліз {service} версії {version} заборонено. "
            f"{facts} Причини: {reasons}."
        )

    return FinalAnswer(
        answer=answer,
        confidence=1.0,
        sources=sources,
    )


# ============================================================
# 11. GROUNDED INFORMATION ANSWER
# ============================================================

def build_grounded_information_answer(
    query: str,
    tool_results: list[dict],
) -> FinalAnswer | None:
    if is_release_decision_query(query):
        return None

    allowed = {
        tool.name
        for tool in select_tools_for_query(query)
    }

    facts = []
    sources = []
    seen = set()

    expected_version = requested_version(query)

    for item in tool_results:
        name = item.get("tool")
        data = item.get("result")

        if (
            name not in allowed
            or not isinstance(data, dict)
        ):
            continue

        service = data.get("service")

        if not isinstance(service, str):
            continue

        key = None
        fact = None

        if (
            name == "get_service_status"
            and data.get("status") is not None
        ):
            key = (name, service)

            fact = (
                f"Сервіс {service}: статус {data['status']}."
            )

            if data.get("latency_ms") is not None:
                fact += (
                    f" Затримка {data['latency_ms']} мс."
                )

        elif name == "get_test_results":
            required = (
                "version",
                "passed",
                "failed",
                "coverage_pct",
            )

            if any(
                data.get(field) is None
                for field in required
            ):
                continue

            if (
                expected_version is not None
                and data["version"] != expected_version
            ):
                continue

            key = (
                name,
                service,
                data["version"],
            )

            fact = (
                f"Результати тестування {service} "
                f"версії {data['version']}: "
                f"пройдено тестів {data['passed']}, "
                f"провалених тестів {data['failed']}, "
                f"покриття тестами {data['coverage_pct']}%."
            )

        elif name == "get_deployment_history":
            deployments = data.get("deployments")

            if not isinstance(deployments, list):
                continue

            entries = [
                f"версія {d['version']} — {d['result']}"
                for d in deployments
                if (
                    isinstance(d, dict)
                    and "version" in d
                    and "result" in d
                )
            ]

            if not entries:
                continue

            key = (name, service)

            fact = (
                f"Історія розгортань {service}: "
                + "; ".join(entries)
                + "."
            )

        if fact is None or key in seen:
            continue

        seen.add(key)
        facts.append(fact)

        if name not in sources:
            sources.append(name)

    if not facts:
        return None

    return FinalAnswer(
        answer=" ".join(facts),
        confidence=1.0,
        sources=sources,
    )


# ============================================================
# 12. FORMATTER NODE
# ============================================================

async def formatter_node(
    state: AgentState,
) -> dict:
    query = user_query_from_state(state)

    agent_answer = ""
    tool_results = []
    actual_sources = []
    had_tool_messages = False

    for message in state["messages"]:
        if isinstance(message, ToolMessage):
            had_tool_messages = True

            if message.status == "error":
                continue

            try:
                data = json.loads(message.content)
            except (ValueError, TypeError):
                continue

            if (
                not isinstance(data, dict)
                or "error" in data
            ):
                continue

            tool_results.append({
                "tool": message.name,
                "result": data,
            })

            if message.name not in actual_sources:
                actual_sources.append(message.name)

        elif (
            isinstance(message, AIMessage)
            and not message.tool_calls
            and message.content
        ):
            agent_answer = str(message.content)

    # --------------------------------------------------------
    # Safety stop
    # --------------------------------------------------------

    if state.get("safety_stop"):
        details = (
            json.dumps(
                tool_results,
                ensure_ascii=False,
            )
            if tool_results
            else "Результатів інструментів ще немає."
        )

        final = FinalAnswer(
            answer=(
                "Виконання зупинено: "
                f"{state.get('safety_message', state['safety_stop'])} "
                f"Отримані дані: {details}. "
                "Остаточний висновок не сформовано."
            ),
            confidence=0.0,
            sources=actual_sources,
        )

        return {
            "final_answer": final.model_dump()
        }

    # --------------------------------------------------------
    # Deterministic release decision
    # --------------------------------------------------------

    release = build_grounded_release_answer(
        query,
        tool_results,
        actual_sources,
    )

    if release is not None:
        return {
            "final_answer": release.model_dump()
        }

    # --------------------------------------------------------
    # Deterministic information response
    # --------------------------------------------------------

    information = build_grounded_information_answer(
        query,
        tool_results,
    )

    if information is not None:
        return {
            "final_answer": information.model_dump()
        }

    # --------------------------------------------------------
    # Tool errors without confirmed evidence
    # --------------------------------------------------------

    if had_tool_messages and not tool_results:
        final = FinalAnswer(
            answer=(
                "Не вдалося отримати підтверджені дані "
                "з інструментів. Результат перевірки невідомий."
            ),
            confidence=0.0,
            sources=[],
        )

        return {
            "final_answer": final.model_dump()
        }

    # --------------------------------------------------------
    # Unresolved tool failure
    # --------------------------------------------------------

    if unresolved_failed_tools(state):
        final = FinalAnswer(
            answer=(
                "Не вдалося отримати достовірні результати "
                "запитаних інструментів після спроб "
                "виправлення аргументів."
            ),
            confidence=0.0,
            sources=actual_sources,
        )

        return {
            "final_answer": final.model_dump()
        }

    # --------------------------------------------------------
    # Structured LLM with validation fallback
    # --------------------------------------------------------

    evidence = {
        "query": query,
        "agent_answer": agent_answer,
        "tool_results": tool_results,
    }

    try:
        response = await structured_llm.ainvoke([
            SystemMessage(content=FORMATTER_PROMPT),
            HumanMessage(
                content=json.dumps(
                    evidence,
                    ensure_ascii=False,
                    indent=2,
                )
            ),
        ])

        validated = FinalAnswer.model_validate(
            response
        )

    except (
        OutputParserException,
        ValidationError,
        ValueError,
        TypeError,
    ):
        final = FinalAnswer(
            answer=(
                "Не вдалося сформувати валідну "
                "структуровану відповідь. "
                "Підтверджені дані відсутні або неповні."
            ),
            confidence=0.0,
            sources=actual_sources,
        )

        return {
            "final_answer": final.model_dump()
        }

    validated = validated.model_copy(
        update={"sources": actual_sources}
    )

    return {
        "final_answer": validated.model_dump(),
        "step_count": state.get("step_count", 0) + 1,
    }


# ============================================================
# 13. ROUTING
# ============================================================

def router(
    state: AgentState,
) -> Literal["tools", "agent", "formatter"]:
    if state.get("safety_stop"):
        return "formatter"

    last = state["messages"][-1]

    if (
        isinstance(last, AIMessage)
        and last.tool_calls
    ):
        return "tools"

    if should_retry_failed_tools(state):
        return "agent"

    return "formatter"


def after_tools(
    state: AgentState,
) -> Literal["agent", "formatter"]:
    if state.get("safety_stop"):
        return "formatter"

    return "agent"


# ============================================================
# 14. TRAJECTORY LOGGING
# ============================================================

def message_snapshot(
    message: AnyMessage,
) -> dict:
    data = {
        "type": message.type,
        "content": message.content,
    }

    if isinstance(message, AIMessage):
        data["tool_calls"] = message.tool_calls

    if isinstance(message, ToolMessage):
        data.update({
            "name": message.name,
            "status": message.status,
            "tool_call_id": message.tool_call_id,
        })

    return data


def state_snapshot(
    state: AgentState,
) -> dict:
    messages = state.get("messages", [])

    return {
        "step_count": state.get("step_count", 0),
        "messages_count": len(messages),
        "last_message": (
            message_snapshot(messages[-1])
            if messages
            else None
        ),
        "safety_stop": state.get("safety_stop"),
    }


def extract_tool_calls(
    node_name: str,
    state: AgentState,
    output: dict,
) -> list[dict]:
    if node_name == "agent":
        messages = output.get(
            "messages",
            [],
        )

    elif node_name == "tools":
        messages = state.get(
            "messages",
            [],
        )[-1:]

    else:
        messages = []

    return [
        {
            "name": call["name"],
            "args": call["args"],
            "id": call["id"],
        }
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]


async def observed_node(
    node_name: str,
    state: AgentState,
    callback,
    logger: TrajectoryLogger | None,
) -> dict:
    started = time.monotonic()

    try:
        output = await callback(state)

    except BaseException as error:
        if logger is not None:
            logger.log_step(
                node_name=node_name,
                input_data=state_snapshot(state),
                output_data={
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "cancelled": isinstance(
                        error,
                        asyncio.CancelledError,
                    ),
                },
                duration_ms=(
                    time.monotonic() - started
                ) * 1000,
                tool_calls=extract_tool_calls(
                    node_name,
                    state,
                    {},
                ),
            )

        raise

    if logger is not None:
        serialized = {
            key: (
                [
                    message_snapshot(msg)
                    for msg in value
                ]
                if key == "messages"
                else value
            )
            for key, value in output.items()
        }

        logger.log_step(
            node_name=node_name,
            input_data=state_snapshot(state),
            output_data=serialized,
            duration_ms=(
                time.monotonic() - started
            ) * 1000,
            tool_calls=extract_tool_calls(
                node_name,
                state,
                output,
            ),
        )

    return output


# ============================================================
# 15. GRAPH FACTORY
# ============================================================

def build_graph(
    safety_config: SafetyConfig | None = None,
    logger: TrajectoryLogger | None = None,
):
    safety = SafetyController(safety_config)
    graph = StateGraph(AgentState)

    async def agent(state: AgentState) -> dict:
        return await observed_node(
            "agent",
            state,
            lambda current: agent_node(
                current,
                safety,
            ),
            logger,
        )

    async def tools(state: AgentState) -> dict:
        return await observed_node(
            "tools",
            state,
            lambda current: guarded_tools_node(
                current,
                safety,
            ),
            logger,
        )

    async def formatter(state: AgentState) -> dict:
        return await observed_node(
            "formatter",
            state,
            formatter_node,
            logger,
        )

    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("formatter", formatter)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        router,
        {
            "tools": "tools",
            "agent": "agent",
            "formatter": "formatter",
        },
    )

    graph.add_conditional_edges(
        "tools",
        after_tools,
        {
            "agent": "agent",
            "formatter": "formatter",
        },
    )

    graph.add_edge("formatter", END)

    return graph.compile()


app = build_graph()


# ============================================================
# 16. TIMEOUT-AWARE EXECUTION
# ============================================================

async def execute_query(
    query: str,
    max_steps: int = 10,
    max_repeats: int = 3,
    timeout_seconds: float = 120.0,
    trajectory_path: str = "trajectory.json",
) -> dict:
    config = SafetyConfig(
        max_steps=max_steps,
        max_repeats=max_repeats,
        timeout_seconds=timeout_seconds,
    )

    logger = TrajectoryLogger(query)

    graph = build_graph(
        config,
        logger=logger,
    )

    state = {
        "messages": [
            HumanMessage(content=query)
        ],
        "step_count": 0,
    }

    try:
        try:
            async with asyncio.timeout(
                timeout_seconds
            ):
                async for chunk in graph.astream(
                    state,
                    stream_mode="updates",
                    config={
                        "recursion_limit": max(
                            50,
                            max_steps * 4 + 10,
                        ),
                    },
                ):
                    for update in chunk.values():
                        if not isinstance(update, dict):
                            continue

                        if "messages" in update:
                            state["messages"] = add_messages(
                                state["messages"],
                                update["messages"],
                            )

                        for key, value in update.items():
                            if key != "messages":
                                state[key] = value

        except TimeoutError:
            state["safety_stop"] = "timeout"
            state["safety_message"] = (
                "Перевищено загальний timeout "
                f"({timeout_seconds} с)"
            )

            logger.log_step(
                node_name="safety",
                input_data={
                    "timeout_seconds": timeout_seconds,
                },
                output_data={
                    "reason": "timeout",
                },
                duration_ms=0.0,
            )

            partial = await observed_node(
                "formatter",
                state,
                formatter_node,
                logger,
            )

            state.update(partial)

        return state

    finally:
        logger.save(trajectory_path)


# ============================================================
# 17. PUBLIC INTERFACE
# ============================================================

def run_query(
    query: str,
    max_steps: int = 10,
    max_repeats: int = 3,
    timeout_seconds: float = 120.0,
    trajectory_path: str = "trajectory.json",
) -> dict:
    result = asyncio.run(
        execute_query(
            query=query,
            max_steps=max_steps,
            max_repeats=max_repeats,
            timeout_seconds=timeout_seconds,
            trajectory_path=trajectory_path,
        )
    )

    print("\n=== TRAJECTORY ===")

    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"USER: {message.content}")

        elif isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(
                    f"\nAGENT → TOOL: {call['name']}"
                )
                print(
                    f"ARGUMENTS: {call['args']}"
                )

        elif isinstance(message, ToolMessage):
            prefix = (
                "TOOL ERROR"
                if message.status == "error"
                else "TOOL RESULT"
            )

            print(
                f"{prefix} [{message.name}]: "
                f"{message.content}"
            )

    if result.get("safety_stop"):
        print("\n=== SAFETY STOP ===")
        print(
            "REASON:",
            result["safety_stop"],
        )

    print("\n=== STRUCTURED FINAL ANSWER ===")

    final = FinalAnswer.model_validate(
        result["final_answer"]
    )

    print(
        final.model_dump_json(indent=2)
    )

    print(
        "\nLLM CALLS:",
        result["step_count"],
    )

    print(
        "\nTRAJECTORY FILE:",
        trajectory_path,
    )

    return result


# ============================================================
# 18. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_query(
        "Чи можна випускати реліз api версії 1.4.0? "
        "Перевір стан сервісу, результати тестів "
        "та політику релізу."
    )
