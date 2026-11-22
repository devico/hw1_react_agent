
"""ReAct-агент для перевірки готовності програмних релізів."""

import asyncio
import json
import time
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from logger import TrajectoryLogger
from safety import SafetyConfig, SafetyController, SafetyStop
from schemas import FinalAnswer
from tools import TOOLS


# ============================================================
# 1. PROMPTS
# ============================================================

SYSTEM_PROMPT = """
Ти DevOps-асистент для перевірки сервісів і релізів.

Доступні сервіси: api, payments, worker.

Правила:
1. Стан сервісу перевіряй через get_service_status.
2. Результати тестів отримуй через get_test_results.
3. Історію розгортань отримуй через get_deployment_history.
4. Для рішення про реліз спочатку отримай стан сервісу
   та результати тестів потрібної версії.
5. Лише після отримання фактичних результатів викликай
   check_release_policy.
6. Не вигадуй і не округлюй метрики.
7. Якщо інструмент повернув помилку, виправ причину.
8. Не дозволяй реліз без підтвердження політики.
9. Усі дані демонстраційні, не production.

Відповідай українською мовою.
"""

FORMATTER_PROMPT = """
Сформуй структуровану відповідь DevOps-агента.

Використовуй лише надані факти та результати інструментів.
Не вигадуй метрики або джерела.
Дані демонстраційні, а не production.

Поверни answer, confidence та sources.
"""


# ============================================================
# 2. STATE
# ============================================================

class AgentState(TypedDict):
    """Стан виконання ReAct-графа."""

    messages: Annotated[list[AnyMessage], add_messages]
    step_count: int

    final_answer: NotRequired[dict]
    safety_stop: NotRequired[str]
    safety_message: NotRequired[str]


# ============================================================
# 3. LLM
# ============================================================

llm = ChatOllama(
    model="qwen2.5:7b-instruct",
    temperature=0,
)

llm_with_tools = llm.bind_tools(TOOLS)

structured_llm = llm.with_structured_output(
    FinalAnswer,
    method="json_schema",
)

tool_executor = ToolNode(TOOLS)


# ============================================================
# 4. AGENT NODE
# ============================================================

async def agent_node(
    state: AgentState,
    safety: SafetyController,
) -> dict:
    """Викликати LLM після перевірки обмежень."""

    completed_steps = state.get("step_count", 0)

    try:
        safety.before_agent(completed_steps)

    except SafetyStop as error:
        return {
            "safety_stop": error.reason,
            "safety_message": str(error),
            "messages": [AIMessage(content=str(error))],
        }

    response = await llm_with_tools.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ])

    safety.check_timeout()

    return {
        "messages": [response],
        "step_count": completed_steps + 1,
    }


# ============================================================
# 5. OBSERVATIONS
# ============================================================

def collect_observations(messages: list) -> dict:
    """Зібрати результати успішно виконаних інструментів."""

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

        if not isinstance(data, dict) or "error" in data:
            continue

        service = data.get("service")

        if service:
            observations[(message.name, service)] = data

    return observations


# ============================================================
# 6. RELEASE POLICY GUARD
# ============================================================

def validate_release_policy_call(
    call: dict,
    observations: dict,
) -> str | None:
    """Перевірити залежності та фактичні аргументи."""

    args = call["args"]
    service = args.get("service")

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
# 7. GUARDED TOOLS NODE
# ============================================================

async def guarded_tools_node(
    state: AgentState,
    safety: SafetyController,
) -> dict:
    """Виконати дозволені інструменти із захистом."""

    last_message = state["messages"][-1]

    if not isinstance(last_message, AIMessage):
        return {"messages": []}

    calls = last_message.tool_calls

    if not calls:
        return {"messages": []}

    try:
        for call in calls:
            safety.before_tool(call["name"], call["args"])

    except SafetyStop as error:
        cancelled = [
            ToolMessage(
                content=f"Виклик скасовано: {error}",
                tool_call_id=call["id"],
                name=call["name"],
                status="error",
            )
            for call in calls
        ]

        return {
            "messages": cancelled,
            "safety_stop": error.reason,
            "safety_message": str(error),
        }

    observations = collect_observations(
        state["messages"][:-1]
    )

    allowed_calls = []
    rejected_messages = {}

    for call in calls:
        if call["name"] != "check_release_policy":
            allowed_calls.append(call)
            continue

        error = validate_release_policy_call(
            call,
            observations,
        )

        if error is None:
            allowed_calls.append(call)
        else:
            rejected_messages[call["id"]] = ToolMessage(
                content=error,
                tool_call_id=call["id"],
                name=call["name"],
                status="error",
            )

    executed_messages = []

    if allowed_calls:
        synthetic_message = AIMessage(
            content="",
            tool_calls=allowed_calls,
        )

        execution = await tool_executor.ainvoke({
            "messages": [synthetic_message],
        })

        executed_messages = execution["messages"]

    responses = {
        message.tool_call_id: message
        for message in executed_messages
    }

    responses.update(rejected_messages)

    return {
        "messages": [
            responses[call["id"]]
            for call in calls
        ]
    }


# ============================================================
# 8. GROUNDED RELEASE ANSWER
# ============================================================

def is_release_decision_query(query: str) -> bool:
    """Розпізнати запит про готовність або дозвіл релізу."""

    text = query.casefold()

    release_words = ("реліз", "релиз", "release")
    decision_words = (
        "можна", "можно", "випуск", "выпуск",
        "дозвол", "разреш", "готов", "approve",
        "deploy",
    )

    return (
        any(word in text for word in release_words)
        and any(word in text for word in decision_words)
    )


def build_grounded_release_answer(
    query: str,
    tool_results: list[dict],
    sources: list[str],
) -> FinalAnswer | None:
    """Створити висновок про реліз лише з перевірених даних."""

    if not is_release_decision_query(query):
        return None

    def find_latest(tool_name: str, service: str | None = None):
        for item in reversed(tool_results):
            if item.get("tool") != tool_name:
                continue

            data = item.get("result")

            if not isinstance(data, dict):
                continue

            if service is None or data.get("service") == service:
                return data

        return None

    def incomplete(message: str) -> FinalAnswer:
        return FinalAnswer(
            answer=message,
            confidence=0.0,
            sources=sources,
        )

    policy = find_latest("check_release_policy")

    if policy is None:
        return incomplete(
            "Готовність релізу не підтверджена: "
            "check_release_policy не було успішно виконано. "
            "Автоматичний дозвіл на реліз відсутній."
        )

    service = policy.get("service")

    status = find_latest("get_service_status", service)
    tests = find_latest("get_test_results", service)

    if not status or not tests:
        return incomplete(
            "Недостатньо підтверджених даних для рішення "
            "про готовність релізу."
        )

    failed = tests.get("failed")
    coverage = tests.get("coverage_pct")
    service_status = status.get("status")

    if (
        not isinstance(failed, int)
        or isinstance(failed, bool)
        or not isinstance(coverage, (int, float))
        or isinstance(coverage, bool)
        or service_status not in {"healthy", "degraded"}
        or type(policy.get("approved")) is not bool
    ):
        return incomplete(
            "Результати інструментів неповні або некоректні. "
            "Рішення про реліз не підтверджено."
        )

    expected_approval = (
        service_status == "healthy"
        and failed == 0
        and coverage >= 90
    )

    if policy["approved"] != expected_approval:
        return incomplete(
            "Виявлено суперечність між метриками та рішенням "
            "політики релізу. Автоматичний дозвіл неможливий."
        )

    version = tests.get("version", "невідомої версії")

    facts = (
        f"Сервіс {service}: статус {service_status}, "
        f"провалених тестів {failed}, "
        f"покриття тестами {coverage}%."
    )

    if expected_approval:
        answer = (
            f"Реліз {service} версії {version} дозволено "
            f"згідно з демонстраційною політикою. "
            f"{facts} Усі критерії виконані."
        )
    else:
        reasons = policy.get("reasons", [])
        details = "; ".join(str(reason) for reason in reasons)

        answer = (
            f"Реліз {service} версії {version} заборонено. "
            f"{facts} Причини: {details}."
        )

    return FinalAnswer(
        answer=answer,
        confidence=1.0,
        sources=sources,
    )


# ============================================================
# 9. FORMATTER NODE
# ============================================================

async def formatter_node(state: AgentState) -> dict:
    """Сформувати структурований або частковий результат."""

    user_query = ""
    agent_answer = ""
    tool_results = []
    actual_sources = []

    for message in state["messages"]:
        if isinstance(message, HumanMessage):
            if not user_query:
                user_query = str(message.content)

        elif isinstance(message, ToolMessage):
            if message.status == "error":
                continue

            try:
                data = json.loads(message.content)
            except (ValueError, TypeError):
                continue

            if not isinstance(data, dict) or "error" in data:
                continue

            tool_results.append({
                "tool": message.name,
                "result": data,
            })

            if message.name not in actual_sources:
                actual_sources.append(message.name)

        elif isinstance(message, AIMessage):
            if not message.tool_calls and message.content:
                agent_answer = str(message.content)

    if state.get("safety_stop"):
        partial_data = (
            json.dumps(tool_results, ensure_ascii=False)
            if tool_results
            else "Результатів інструментів ще немає."
        )

        final = FinalAnswer(
            answer=(
                "Виконання зупинено: "
                f"{state.get('safety_message', state['safety_stop'])} "
                f"Отримані дані: {partial_data}. "
                "Остаточний висновок не сформовано "
                "через обмеження виконання."
            ),
            confidence=0.0,
            sources=actual_sources,
        )

        return {"final_answer": final.model_dump()}

    grounded = build_grounded_release_answer(
        user_query,
        tool_results,
        actual_sources,
    )

    if grounded is not None:
        return {"final_answer": grounded.model_dump()}

    evidence = {
        "query": user_query,
        "agent_answer": agent_answer,
        "tool_results": tool_results,
    }

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

    validated = response.model_copy(
        update={"sources": actual_sources}
    )

    return {
        "final_answer": validated.model_dump(),
        "step_count": state.get("step_count", 0) + 1,
    }


# ============================================================
# 10. ROUTING
# ============================================================

def router(
    state: AgentState,
) -> Literal["tools", "formatter"]:
    """Обрати наступний вузол після agent."""

    if state.get("safety_stop"):
        return "formatter"

    last_message = state["messages"][-1]

    if (
        isinstance(last_message, AIMessage)
        and last_message.tool_calls
    ):
        return "tools"

    return "formatter"


def after_tools(
    state: AgentState,
) -> Literal["agent", "formatter"]:
    """Повернутися до agent або завершити виконання."""

    if state.get("safety_stop"):
        return "formatter"

    return "agent"


# ============================================================
# 11. TRAJECTORY LOGGING
# ============================================================

def message_snapshot(message: AnyMessage) -> dict:
    """Підготувати повідомлення для JSON-логу."""

    data = {
        "type": message.type,
        "content": message.content,
    }

    if isinstance(message, AIMessage):
        data["tool_calls"] = message.tool_calls

    if isinstance(message, ToolMessage):
        data["name"] = message.name
        data["status"] = message.status
        data["tool_call_id"] = message.tool_call_id

    return data


def state_snapshot(state: AgentState) -> dict:
    """Підготувати вхідний стан вузла до логування."""

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
    """Отримати виклики інструментів для запису траєкторії."""

    if node_name == "agent":
        messages = output.get("messages", [])

    elif node_name == "tools":
        messages = state.get("messages", [])[-1:]

    else:
        messages = []

    calls = []

    for message in messages:
        if not isinstance(message, AIMessage):
            continue

        for call in message.tool_calls:
            calls.append({
                "name": call["name"],
                "args": call["args"],
                "id": call["id"],
            })

    return calls


async def observed_node(
    node_name: str,
    state: AgentState,
    callback,
    logger: TrajectoryLogger | None,
) -> dict:
    """Виконати вузол та записати реальну тривалість.

    Помилки й скасування також логуються.
    Якщо logger не передано, вузол працює як звичайно.
    """

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
                        error, asyncio.CancelledError
                    ),
                },
                duration_ms=(
                    time.monotonic() - started
                ) * 1000,
                tool_calls=extract_tool_calls(
                    node_name, state, {}
                ),
            )

        raise

    if logger is not None:
        serialized_output = {}

        for key, value in output.items():
            if key == "messages":
                serialized_output[key] = [
                    message_snapshot(message)
                    for message in value
                ]
            else:
                serialized_output[key] = value

        logger.log_step(
            node_name=node_name,
            input_data=state_snapshot(state),
            output_data=serialized_output,
            duration_ms=(
                time.monotonic() - started
            ) * 1000,
            tool_calls=extract_tool_calls(
                node_name, state, output
            ),
        )

    return output


# ============================================================
# 12. GRAPH FACTORY
# ============================================================

def build_graph(
    safety_config: SafetyConfig | None = None,
    logger: TrajectoryLogger | None = None,
):
    """Створити ReAct-граф із захистом та логуванням."""

    safety = SafetyController(safety_config)

    graph = StateGraph(AgentState)

    async def agent(state: AgentState) -> dict:
        async def execute(current_state):
            return await agent_node(current_state, safety)

        return await observed_node(
            "agent", state, execute, logger
        )

    async def tools(state: AgentState) -> dict:
        async def execute(current_state):
            return await guarded_tools_node(
                current_state, safety
            )

        return await observed_node(
            "tools", state, execute, logger
        )

    async def formatter(state: AgentState) -> dict:
        return await observed_node(
            "formatter", state, formatter_node, logger
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
# 13. TIMEOUT-AWARE EXECUTION
# ============================================================

async def execute_query(
    query: str,
    max_steps: int = 10,
    max_repeats: int = 3,
    timeout_seconds: float = 120.0,
    trajectory_path: str = "trajectory.json",
) -> dict:
    """Виконати граф і зберегти траєкторію у JSON."""

    config = SafetyConfig(
        max_steps=max_steps,
        max_repeats=max_repeats,
        timeout_seconds=timeout_seconds,
    )

    # Кожен запит отримує власний логер і граф.
    logger = TrajectoryLogger(query)
    graph = build_graph(config, logger=logger)

    state = {
        "messages": [HumanMessage(content=query)],
        "step_count": 0,
    }

    try:
        try:
            async with asyncio.timeout(timeout_seconds):
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
            # Зберігаємо вже завершені вузли та факти.
            state["safety_stop"] = "timeout"
            state["safety_message"] = (
                f"Перевищено загальний timeout "
                f"({timeout_seconds} с)"
            )

            logger.log_step(
                node_name="safety",
                input_data={"timeout_seconds": timeout_seconds},
                output_data={"reason": "timeout"},
                duration_ms=0.0,
            )

            # Без нового LLM-виклику формуємо часткову відповідь.
            partial = await observed_node(
                "formatter",
                state,
                formatter_node,
                logger,
            )
            state.update(partial)

        return state

    finally:
        # JSON зберігається і при нормальному завершенні,
        # і при timeout, і при непередбаченій помилці.
        logger.save(trajectory_path)


# ============================================================
# 14. PUBLIC INTERFACE
# ============================================================

def run_query(
    query: str,
    max_steps: int = 10,
    max_repeats: int = 3,
    timeout_seconds: float = 120.0,
    trajectory_path: str = "trajectory.json",
) -> dict:
    """Запустити агента із Git Bash або CLI."""

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
                print(f"\nAGENT → TOOL: {call['name']}")
                print(f"ARGUMENTS: {call['args']}")

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
        print("REASON:", result["safety_stop"])

    print("\n=== STRUCTURED FINAL ANSWER ===")

    final = FinalAnswer.model_validate(
        result["final_answer"]
    )

    print(final.model_dump_json(indent=2))
    print("\nLLM CALLS:", result["step_count"])

    print("\nTRAJECTORY FILE:", trajectory_path)

    return result


# ============================================================
# 15. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_query(
        "Чи можна випускати реліз api версії 1.4.0? "
        "Перевір стан сервісу, результати тестів "
        "та політику релізу."
    )
