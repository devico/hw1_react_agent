
"""Запуск тест-кейсів ReAct-агента та збір результатів."""

import asyncio
import json
import shutil
import time
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from agent import execute_query
from schemas import FinalAnswer


# ============================================================
# 1. TEST CASES
# ============================================================

TEST_CASES = [
    {
        "id": "TC-001",
        "complexity": "simple",
        "query": "Перевір поточний стан сервісу api.",
        "expected": "Сервіс api має статус healthy.",
        "required_tools": ["get_service_status"],
        "expected_facts": {
            "get_service_status": {
                "service": "api",
                "status": "healthy",
                "latency_ms": 120,
            },
        },
        "answer_contains": ["api"],
    },
    {
        "id": "TC-002",
        "complexity": "simple",
        "query": (
            "Покажи два останні розгортання сервісу payments "
            "та результат кожного."
        ),
        "expected": (
            "Два розгортання payments: "
            "2.0.9 failed, 2.0.8 success."
        ),
        "required_tools": ["get_deployment_history"],
        "expected_facts": {
            "get_deployment_history": {
                "service": "payments",
                "deployments": [
                    {"version": "2.0.9", "result": "failed"},
                    {"version": "2.0.8", "result": "success"},
                ],
            },
        },
        "answer_contains": ["2.0.9", "2.0.8"],
    },
    {
        "id": "TC-003",
        "complexity": "medium",
        "query": (
            "Перевір стан сервісу worker та результати "
            "тестування версії 1.2.0. Повідом статус, "
            "кількість провалених тестів і покриття."
        ),
        "expected": (
            "worker healthy, failed=0, coverage=92%."
        ),
        "required_tools": [
            "get_service_status",
            "get_test_results",
        ],
        "expected_facts": {
            "get_service_status": {
                "service": "worker",
                "status": "healthy",
            },
            "get_test_results": {
                "service": "worker",
                "version": "1.2.0",
                "failed": 0,
                "coverage_pct": 92.0,
            },
        },
        "answer_contains": ["worker", "92"],
    },
    {
        "id": "TC-004",
        "complexity": "complex",
        "query": (
            "Чи можна випускати реліз api версії 1.4.0? "
            "Перевір стан сервісу, результати тестів "
            "та політику релізу."
        ),
        "expected": (
            "Реліз дозволено: healthy, 0 failed, "
            "coverage=94%, approved=true."
        ),
        "required_tools": [
            "get_service_status",
            "get_test_results",
            "check_release_policy",
        ],
        "expected_facts": {
            "get_service_status": {
                "service": "api",
                "status": "healthy",
            },
            "get_test_results": {
                "service": "api",
                "version": "1.4.0",
                "failed": 0,
                "coverage_pct": 94.0,
            },
            "check_release_policy": {
                "service": "api",
                "approved": True,
            },
        },
        "answer_contains": ["дозволено", "94"],
    },
    {
        "id": "TC-005",
        "complexity": "complex",
        "query": (
            "Чи можна випускати реліз payments версії 2.1.0? "
            "Перевір стан сервісу, результати тестів "
            "та політику релізу."
        ),
        "expected": (
            "Реліз заборонено: degraded, 4 failed, "
            "coverage=87%, approved=false."
        ),
        "required_tools": [
            "get_service_status",
            "get_test_results",
            "check_release_policy",
        ],
        "expected_facts": {
            "get_service_status": {
                "service": "payments",
                "status": "degraded",
            },
            "get_test_results": {
                "service": "payments",
                "version": "2.1.0",
                "failed": 4,
                "coverage_pct": 87.0,
            },
            "check_release_policy": {
                "service": "payments",
                "approved": False,
            },
        },
        "answer_contains": ["заборонено", "87"],
    },
]


# ============================================================
# 2. COLLECT TOOL INFORMATION
# ============================================================

def collect_tool_data(messages):
    """Зібрати фактичні виклики та результати інструментів."""

    calls = []
    observations = {}

    for message in messages:

        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls.append({
                    "name": call["name"],
                    "args": call["args"],
                    "id": call["id"],
                })

        elif isinstance(message, ToolMessage):
            if message.status == "error":
                continue

            try:
                data = json.loads(message.content)
            except (ValueError, TypeError):
                continue

            if not isinstance(data, dict):
                continue

            if "error" in data:
                continue

            observations.setdefault(
                message.name, []
            ).append(data)

    return calls, observations


# ============================================================
# 3. EVALUATION
# ============================================================

def evaluate_case(
    case: dict,
    answer: FinalAnswer,
    observations: dict,
) -> tuple[bool, list[str]]:
    """Перевірити інструменти, факти та фінальну відповідь.

    Успіх визначається перевірками, а не самооцінкою LLM.
    """

    errors = []

    for tool_name in case["required_tools"]:

        results = observations.get(tool_name, [])

        if not results:
            errors.append(
                f"Не отримано результат інструмента: {tool_name}"
            )
            continue

        expected = case["expected_facts"].get(
            tool_name, {}
        )

        # Знаходимо результат, що відповідає очікуваним фактам.
        matching = any(
            all(
                result.get(key) == value
                for key, value in expected.items()
            )
            for result in results
        )

        if not matching:
            errors.append(
                f"Некоректні фактичні дані: {tool_name}"
            )

        if tool_name not in answer.sources:
            errors.append(
                f"Відсутнє джерело у відповіді: {tool_name}"
            )

    actual_text = answer.answer.casefold()

    for fragment in case["answer_contains"]:
        if fragment.casefold() not in actual_text:
            errors.append(
                f"У фінальній відповіді немає: {fragment}"
            )

    return len(errors) == 0, errors


# ============================================================
# 4. EXECUTE ONE CASE
# ============================================================

async def run_case(case: dict) -> dict:
    """Виконати один тест-кейс та зібрати метрики."""

    test_id = case["id"]

    trajectory_path = Path("trajectories") / (
        f"{test_id}.json"
    )

    started = time.monotonic()

    record = {
        "test_id": test_id,
        "complexity": case["complexity"],
        "query": case["query"],
        "expected": case["expected"],
        "actual": "",
        "status": "error",
        "steps": 0,
        "llm_calls": 0,
        "tool_calls": [],
        "tool_call_details": [],
        "sources": [],
        "confidence": None,
        "elapsed_ms": 0,
        "max_steps_reached": False,
        "timeout_reached": False,
        "loop_detected": False,
        "safety_stop": None,
        "evaluation_errors": [],
        "trajectory_file": str(trajectory_path),
    }

    try:
        result = await execute_query(
            query=case["query"],
            max_steps=10,
            max_repeats=3,
            timeout_seconds=120.0,
            trajectory_path=str(trajectory_path),
        )

        final = FinalAnswer.model_validate(
            result["final_answer"]
        )

        calls, observations = collect_tool_data(
            result["messages"]
        )

        passed, errors = evaluate_case(
            case,
            final,
            observations,
        )

        safety_stop = result.get("safety_stop")

        if safety_stop:
            status = "partial"
        elif passed:
            status = "success"
        else:
            status = "failed"

        record.update({
            "actual": final.answer,
            "status": status,
            "llm_calls": result.get("step_count", 0),
            "tool_calls": [
                call["name"] for call in calls
            ],
            "tool_call_details": calls,
            "sources": final.sources,
            "confidence": final.confidence,
            "max_steps_reached": safety_stop == "max_steps",
            "timeout_reached": safety_stop == "timeout",
            "loop_detected": safety_stop == "loop_detected",
            "safety_stop": safety_stop,
            "evaluation_errors": errors,
        })

    except Exception as error:
        record["actual"] = (
            f"{type(error).__name__}: {error}"
        )
        record["evaluation_errors"] = [
            "Помилка виконання агента"
        ]

    finally:
        record["elapsed_ms"] = round(
            (time.monotonic() - started) * 1000,
            2,
        )

        # Кількість кроків — фактичні вузли з trajectory.json.
        if trajectory_path.exists():
            with trajectory_path.open(
                encoding="utf-8"
            ) as file:
                trajectory = json.load(file)

            record["steps"] = trajectory["total_steps"]

    return record


# ============================================================
# 5. RUN ALL CASES
# ============================================================

async def main():
    """Прогнати всі сценарії та зберегти результати."""

    Path("trajectories").mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for case in TEST_CASES:
        print(
            f"\nRunning {case['id']} "
            f"({case['complexity']})...",
            flush=True,
        )

        result = await run_case(case)
        results.append(result)

        print(
            f"  Status: {result['status']}\n"
            f"  Steps: {result['steps']}\n"
            f"  LLM calls: {result['llm_calls']}\n"
            f"  Tools: {result['tool_calls']}\n"
            f"  Time: {result['elapsed_ms']} ms",
            flush=True,
        )

        if result["evaluation_errors"]:
            print(
                "  Issues:",
                result["evaluation_errors"],
                flush=True,
            )

    # Зберігаємо повні результати тестування.
    with open(
        "test_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Для здачі окремо зберігаємо траєкторію
    # складного сценарію перевірки релізу.
    representative = Path(
        "trajectories/TC-004.json"
    )

    if representative.exists():
        shutil.copyfile(
            representative,
            "trajectory.json",
        )

    successful = sum(
        result["status"] == "success"
        for result in results
    )

    print("\n=== SUMMARY ===")
    print(f"Successful: {successful}/{len(results)}")
    print("Results: test_results.json")
    print("Trajectory: trajectory.json")


if __name__ == "__main__":
    asyncio.run(main())
