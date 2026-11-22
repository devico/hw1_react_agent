
"""Функціональні перевірки ReAct-агента та аудит якості відповідей.

Звичайний запуск звертається до Ollama та оновлює результати.
Режим --audit-existing перевіряє старі відповіді без Ollama і без запису файлів.
"""

import asyncio
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from agent import execute_query
from schemas import FinalAnswer


# ============================================================
# 1. SCENARIOS: TOOL EVIDENCE + ANSWER REQUIREMENTS
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
        "answer_requirements": {
            "service": "api",
            "status": "healthy",
            "release_decision": None,
        },
    },
    {
        "id": "TC-002",
        "complexity": "simple",
        "query": (
            "Покажи два останні розгортання сервісу payments "
            "та результат кожного."
        ),
        "expected": "Два розгортання payments: 2.0.9 failed, 2.0.8 success.",
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
        "answer_requirements": {
            "service": "payments",
            "deployments": [
                {"version": "2.0.9", "result": "failed"},
                {"version": "2.0.8", "result": "success"},
            ],
            "release_decision": None,
        },
    },
    {
        "id": "TC-003",
        "complexity": "medium",
        "query": (
            "Перевір стан сервісу worker та результати "
            "тестування версії 1.2.0. Повідом статус, "
            "кількість провалених тестів і покриття."
        ),
        "expected": "worker healthy, failed=0, coverage=92%.",
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
        "answer_requirements": {
            "service": "worker",
            "version": "1.2.0",
            "status": "healthy",
            "failed": 0,
            "coverage": 92,
            "release_decision": None,
        },
    },
    {
        "id": "TC-004",
        "complexity": "complex",
        "query": (
            "Чи можна випускати реліз api версії 1.4.0? "
            "Перевір стан сервісу, результати тестів "
            "та політику релізу."
        ),
        "expected": "Реліз дозволено: healthy, 0 failed, coverage=94%.",
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
        "answer_requirements": {
            "service": "api",
            "version": "1.4.0",
            "status": "healthy",
            "failed": 0,
            "coverage": 94,
            "release_decision": "allow",
        },
    },
    {
        "id": "TC-005",
        "complexity": "complex",
        "query": (
            "Чи можна випускати реліз payments версії 2.1.0? "
            "Перевір стан сервісу, результати тестів "
            "та політику релізу."
        ),
        "expected": "Реліз заборонено: degraded, 4 failed, coverage=87%.",
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
        "answer_requirements": {
            "service": "payments",
            "version": "2.1.0",
            "status": "degraded",
            "failed": 4,
            "coverage": 87,
            "release_decision": "deny",
        },
    },
]


# ============================================================
# 2. DETERMINISTIC CHECKS OF THE FINAL ANSWER
# ============================================================

RELEASE_CONTEXT = re.compile(
    r"\b(?:реліз\w*|релиз\w*|release|випуск\w*)\b"
)

RELEASE_ALLOW = re.compile(
    r"\b(?:дозволен\w*|схвален\w*|підтверджен\w*|"
    r"затверджен\w*|approved|"
    r"можна\s+(?:випускати|випустити|релізити|запускати)|"
    r"може\s+бути\s+(?:розповсюджен\w*|випущен\w*)|"
    r"готов\w*\s+до\s+(?:реліз\w*|випуск\w*))\b"
)

RELEASE_DENY = re.compile(
    r"\b(?:заборонен\w*|відхилен\w*|rejected|blocked|"
    r"не\s+(?:дозволен\w*|схвален\w*|підтверджен\w*|готов\w*)|"
    r"не\s+можна\s+(?:випускати|випустити|релізити|запускати)|"
    r"не\s+рекоменду\w*\s+(?:випускати|релізити)|"
    r"not\s+approved)\b"
)

DEPLOYMENT_FAILED = re.compile(
    r"\b(?:не\s*успіш\w*|невдал\w*|помил\w*|"
    r"failed|failure|unsuccessful)\b"
)

DEPLOYMENT_SUCCESS = re.compile(
    r"\b(?:успіш\w*|вдал\w*|success|successful)\b"
)

FAILED_TEST_TERMS = re.compile(
    r"\b(?:тест\w*|провален\w*|невдал\w*|"
    r"failed|failures|помил\w*)\b"
)

COVERAGE_TERMS = re.compile(
    r"\b(?:покрит\w*|coverage)\b"
)


def normalized_text(value: str) -> str:
    """Нормалізувати текст для перевірок."""
    return " ".join(value.casefold().split())


def contains_token(text: str, token: str) -> bool:
    """Знайти слово або версію з правильними межами.

    Для звичайного слова крапка є допустимим роздільником:
        'статус healthy.' -> healthy знайдено.

    Для версії крапка є частиною номера:
        '1.2.0' не повинно збігатися з '1.2.0.1'.
    """

    token = token.casefold()

    if re.fullmatch(r"\d+(?:\.\d+)+", token):
        pattern = (
            rf"(?<![\w.]){re.escape(token)}(?![\w.])"
        )
    else:
        pattern = (
            rf"(?<!\w){re.escape(token)}(?!\w)"
        )

    return bool(re.search(pattern, text))


def contains_metric_near(
    text: str,
    number: int,
    terms: re.Pattern,
    distance: int = 65,
) -> bool:
    """Перевірити значення поруч із назвою метрики.

    Номер версії 1.4.0 не вважається кількістю
    провалених тестів.
    """

    value = re.compile(
        rf"(?<![\d.]){number}(?:[.,]0+)?(?![\d.])"
    )

    number_matches = list(value.finditer(text))
    term_matches = list(terms.finditer(text))

    return any(
        abs(number_match.start() - term_match.start()) <= distance
        for number_match in number_matches
        for term_match in term_matches
    )


def release_decisions(text: str) -> set[str]:
    """Знайти явні рішення про дозвіл або заборону релізу."""

    if not RELEASE_CONTEXT.search(text):
        return set()

    markers = set()

    if RELEASE_DENY.search(text):
        markers.add("deny")

    for match in RELEASE_ALLOW.finditer(text):
        before = text[
            max(0, match.start() - 12):match.start()
        ]

        # 'не дозволено' не є позитивним рішенням.
        if re.search(r"\bне\s+$", before):
            continue

        markers.add("allow")

    return markers


def detect_deployment_outcomes(
    text: str,
    deployments: list[dict],
) -> dict[str, str | None]:
    """Прив'язати статус розгортання до конкретної версії."""

    markers = []

    for item in deployments:
        version = item["version"]

        for match in re.finditer(
            rf"(?<![\d.]){re.escape(version)}(?![\d.])",
            text,
        ):
            markers.append(
                (match.start(), match.end(), version)
            )

    markers.sort()

    outcomes = {
        item["version"]: None
        for item in deployments
    }

    for index, (_, end, version) in enumerate(markers):
        next_start = (
            markers[index + 1][0]
            if index + 1 < len(markers)
            else len(text)
        )

        passage = text[
            end:min(next_start, end + 160)
        ]

        if DEPLOYMENT_FAILED.search(passage):
            outcomes[version] = "failed"

        elif DEPLOYMENT_SUCCESS.search(passage):
            outcomes[version] = "success"

    return outcomes


def evaluate_answer(
    case: dict,
    actual: str,
) -> list[str]:
    """Перевірити зміст відповіді незалежно від інструментів."""

    errors = []
    expected = case["answer_requirements"]
    text = normalized_text(actual)

    if not contains_token(
        text,
        expected["service"],
    ):
        errors.append(
            "У відповіді не зазначено сервіс "
            f"{expected['service']}"
        )

    if (
        "version" in expected
        and not contains_token(
            text,
            expected["version"],
        )
    ):
        errors.append(
            "У відповіді не зазначено версію "
            f"{expected['version']}"
        )

    if (
        "status" in expected
        and not contains_token(
            text,
            expected["status"],
        )
    ):
        errors.append(
            "У відповіді не зазначено статус "
            f"{expected['status']}"
        )

    if (
        "failed" in expected
        and not contains_metric_near(
            text,
            expected["failed"],
            FAILED_TEST_TERMS,
        )
    ):
        errors.append(
            "Не підтверджено кількість провалених тестів: "
            f"{expected['failed']}"
        )

    if (
        "coverage" in expected
        and not contains_metric_near(
            text,
            expected["coverage"],
            COVERAGE_TERMS,
        )
    ):
        errors.append(
            "Не підтверджено покриття тестами: "
            f"{expected['coverage']}%"
        )

    if "deployments" in expected:
        detected = detect_deployment_outcomes(
            text,
            expected["deployments"],
        )

        for deployment in expected["deployments"]:
            version = deployment["version"]

            if detected[version] != deployment["result"]:
                errors.append(
                    f"Для розгортання {version} "
                    "не підтверджено результат "
                    f"{deployment['result']}"
                )

    decision = release_decisions(text)
    required_decision = expected["release_decision"]

    if required_decision is None and decision:
        errors.append(
            "Незапитаний висновок про дозвіл "
            "або заборону релізу"
        )

    elif required_decision is not None:
        if required_decision not in decision:
            errors.append(
                "Немає однозначного рішення про реліз: "
                f"{required_decision}"
            )

        if decision - {required_decision}:
            errors.append(
                "Суперечливі рішення про реліз у відповіді"
            )

    return errors


# ============================================================
# 3. TOOL EVIDENCE + EFFICIENCY
# ============================================================

def collect_tool_data(messages):
    """Зібрати виклики, результати та помилки інструментів."""

    calls = []
    observations = {}
    tool_errors = []

    for message in messages:

        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                calls.append({
                    "name": call["name"],
                    "args": call.get("args", {}),
                    "id": call.get("id"),
                })

        elif isinstance(message, ToolMessage):
            if message.status == "error":
                tool_errors.append({
                    "name": message.name,
                    "message": str(message.content),
                })
                continue

            try:
                data = json.loads(message.content)
            except (ValueError, TypeError):
                tool_errors.append({
                    "name": message.name,
                    "message": "Некоректний JSON",
                })
                continue

            if (
                not isinstance(data, dict)
                or "error" in data
            ):
                tool_errors.append({
                    "name": message.name,
                    "message": str(data),
                })
                continue

            observations.setdefault(
                message.name,
                [],
            ).append(data)

    return calls, observations, tool_errors


def evaluate_case(
    case: dict,
    answer: FinalAnswer,
    observations: dict,
):
    """Перевірити зміст відповіді та підтвердження фактами."""

    errors = evaluate_answer(
        case,
        answer.answer,
    )

    actual_sources = set(observations)

    for tool_name in case["required_tools"]:
        results = observations.get(
            tool_name,
            [],
        )

        if not results:
            errors.append(
                "Не отримано результат інструмента: "
                f"{tool_name}"
            )
            continue

        expected = case["expected_facts"].get(
            tool_name,
            {},
        )

        matching = any(
            all(
                result.get(key) == value
                for key, value in expected.items()
            )
            for result in results
        )

        if not matching:
            errors.append(
                "Некоректні фактичні дані: "
                f"{tool_name}"
            )

        if tool_name not in answer.sources:
            errors.append(
                "Відсутнє джерело у відповіді: "
                f"{tool_name}"
            )

    for source in answer.sources:
        if source not in actual_sources:
            errors.append(
                "Непідтверджене джерело відповіді: "
                f"{source}"
            )

    return len(errors) == 0, errors


def tool_diagnostics(
    case: dict,
    calls: list[dict],
    tool_errors: list[dict],
):
    """Оцінити зайві, повторні та помилкові виклики."""

    required = set(case["required_tools"])

    unexpected = sorted({
        call["name"]
        for call in calls
        if call["name"] not in required
    })

    signatures = Counter(
        (
            call["name"],
            json.dumps(
                call["args"],
                sort_keys=True,
                ensure_ascii=False,
            ),
        )
        for call in calls
    )

    repeated = sum(
        count - 1
        for count in signatures.values()
        if count > 1
    )

    warnings = []

    if unexpected:
        warnings.append(
            "Зайві інструменти: "
            + ", ".join(unexpected)
        )

    if repeated:
        warnings.append(
            "Повторні виклики з однаковими аргументами: "
            f"{repeated}"
        )

    if tool_errors:
        warnings.append(
            "Відхилені/помилкові виклики інструментів: "
            f"{len(tool_errors)}"
        )

    return {
        "total_tool_calls": len(calls),
        "unexpected_tools": unexpected,
        "repeated_tool_calls": repeated,
        "tool_error_count": len(tool_errors),
        "warnings": warnings,
    }


# ============================================================
# 4. REAL EXECUTION
# ============================================================

async def run_case(case: dict) -> dict:
    """Виконати один тестовий сценарій."""

    test_id = case["id"]

    trajectory_path = (
        Path("trajectories") / f"{test_id}.json"
    )

    trajectory_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Не використовувати застарілий лог попереднього запуску.
    trajectory_path.unlink(
        missing_ok=True,
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
        "evaluation_warnings": [],
        "tool_metrics": {},
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

        answer = FinalAnswer.model_validate(
            result["final_answer"]
        )

        (
            calls,
            observations,
            tool_errors,
        ) = collect_tool_data(
            result["messages"]
        )

        passed, errors = evaluate_case(
            case,
            answer,
            observations,
        )

        metrics = tool_diagnostics(
            case,
            calls,
            tool_errors,
        )

        stop = result.get("safety_stop")

        record.update({
            "actual": answer.answer,
            "status": (
                "partial"
                if stop
                else (
                    "success"
                    if passed
                    else "failed"
                )
            ),
            "llm_calls": result.get("step_count", 0),
            "tool_calls": [
                call["name"]
                for call in calls
            ],
            "tool_call_details": calls,
            "sources": answer.sources,
            "confidence": answer.confidence,
            "max_steps_reached": stop == "max_steps",
            "timeout_reached": stop == "timeout",
            "loop_detected": stop == "loop_detected",
            "safety_stop": stop,
            "evaluation_errors": errors,
            "evaluation_warnings": metrics["warnings"],
            "tool_metrics": {
                key: value
                for key, value in metrics.items()
                if key != "warnings"
            },
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

        if trajectory_path.exists():
            try:
                with trajectory_path.open(
                    encoding="utf-8"
                ) as file:
                    record["steps"] = json.load(
                        file
                    )["total_steps"]

            except (
                OSError,
                ValueError,
                KeyError,
            ):
                record["evaluation_warnings"].append(
                    "Не вдалося прочитати траєкторію"
                )

    return record


async def run_all_cases():
    """Запустити сценарії через Ollama і зберегти результати."""

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
                "  Errors:",
                result["evaluation_errors"],
                flush=True,
            )

        if result["evaluation_warnings"]:
            print(
                "  Warnings:",
                result["evaluation_warnings"],
                flush=True,
            )

    with Path("test_results.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    representative = Path(
        "trajectories/TC-004.json"
    )

    if representative.exists():
        shutil.copyfile(
            representative,
            "trajectory.json",
        )

    passed = sum(
        result["status"] == "success"
        for result in results
    )

    print("\n=== SUMMARY ===")
    print(f"Successful: {passed}/{len(results)}")
    print("Results: test_results.json")
    print("Trajectory: trajectory.json")


# ============================================================
# 5. AUDIT EXISTING RESULTS
# ============================================================

def audit_existing(path="test_results.json"):
    """Перевірити збережені відповіді без запуску Ollama.

    Режим не змінює файли.
    Перевіряється лише зміст відповіді, оскільки
    попередній JSON не містить усіх tool observations.
    """

    with Path(path).open(
        encoding="utf-8"
    ) as file:
        saved = json.load(file)

    cases = {
        case["id"]: case
        for case in TEST_CASES
    }

    passed = 0

    for record in saved:
        test_id = record.get("test_id")

        if test_id not in cases:
            print(
                f"{test_id}: UNKNOWN CASE"
            )
            continue

        errors = evaluate_answer(
            cases[test_id],
            record.get("actual", ""),
        )

        status = (
            "PASS"
            if not errors
            else "FAIL"
        )

        if not errors:
            passed += 1

        print(f"{test_id}: {status}")

        for error in errors:
            print(f"  - {error}")

    print(
        f"\nAnswer-only audit: "
        f"{passed}/{len(saved)}"
    )

    print(
        "Existing JSON files were not modified."
    )


# ============================================================
# 6. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if sys.argv[1:] == ["--audit-existing"]:
        audit_existing()

    elif len(sys.argv) == 1:
        asyncio.run(
            run_all_cases()
        )

    else:
        raise SystemExit(
            "Usage: python test_runner.py [--audit-existing]"
        )
