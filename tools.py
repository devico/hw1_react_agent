"""Інструменти DevOps-агента з Pydantic v2 валідацією."""

import json
import re

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator


# Детерміновані демонстраційні дані.
# Це не реальний моніторинг production-сервісів.
SERVICES = {
    "api": {"status": "healthy", "latency_ms": 120},
    "payments": {"status": "degraded", "latency_ms": 850},
    "worker": {"status": "healthy", "latency_ms": 90},
}

TEST_RESULTS = {
    ("api", "1.4.0"): {
        "passed": 120, "failed": 0, "coverage_pct": 94.0
    },
    ("payments", "2.1.0"): {
        "passed": 96, "failed": 4, "coverage_pct": 87.0
    },
    ("worker", "1.2.0"): {
        "passed": 75, "failed": 0, "coverage_pct": 92.0
    },
}

DEPLOYMENTS = {
    "api": [
        {"version": "1.3.9", "result": "success"},
        {"version": "1.3.8", "result": "success"},
    ],
    "payments": [
        {"version": "2.0.9", "result": "failed"},
        {"version": "2.0.8", "result": "success"},
    ],
    "worker": [
        {"version": "1.1.9", "result": "success"},
    ],
}


def to_json(data: dict) -> str:
    """Перетворити результат інструмента на JSON."""
    return json.dumps(data, ensure_ascii=False)


class ServiceStatusInput(BaseModel):
    """Параметри перевірки стану сервісу."""

    model_config = ConfigDict(str_strip_whitespace=True)

    service: str = Field(
        description="Назва сервісу: api, payments або worker"
    )

    @field_validator("service")
    @classmethod
    def validate_service(cls, value: str) -> str:
        if not value:
            raise ValueError("Назва сервісу не може бути порожньою")
        return value.lower()


@tool(args_schema=ServiceStatusInput)
def get_service_status(service: str) -> str:
    """Отримати поточний стан і затримку сервісу.

    Використовуй, коли потрібно перевірити здоров'я сервісу
    або дізнатися його статус: healthy чи degraded.
    Дані демонстраційні, а не з production.
    """
    data = SERVICES.get(service)
    if data is None:
        return to_json({"error": f"Невідомий сервіс: {service}"})
    return to_json({"service": service, **data})


class TestResultsInput(BaseModel):
    """Параметри отримання результатів тестів."""

    model_config = ConfigDict(str_strip_whitespace=True)

    service: str = Field(
        description="Назва сервісу: api, payments або worker"
    )
    version: str = Field(
        description="Версія релізу у форматі X.Y.Z, наприклад 1.4.0"
    )

    @field_validator("service")
    @classmethod
    def validate_service(cls, value: str) -> str:
        if not value:
            raise ValueError("Назва сервісу обов'язкова")
        return value.lower()

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if not re.fullmatch(r"\d+\.\d+\.\d+", value):
            raise ValueError("Версія повинна мати формат X.Y.Z")
        return value


@tool(args_schema=TestResultsInput)
def get_test_results(service: str, version: str) -> str:
    """Отримати результати тестування конкретного релізу.

    Використовуй для перевірки passed, failed та coverage_pct.
    Потрібно знати назву сервісу та версію релізу.
    """
    data = TEST_RESULTS.get((service, version))
    if data is None:
        return to_json({"error": "Результати тестів не знайдено"})
    return to_json({"service": service, "version": version, **data})


class DeploymentHistoryInput(BaseModel):
    """Параметри перегляду історії розгортань."""

    model_config = ConfigDict(str_strip_whitespace=True)

    service: str = Field(
        description="Назва сервісу: api, payments або worker"
    )
    limit: int = Field(
        default=3,
        description="Кількість останніх розгортань від 1 до 10"
    )

    @field_validator("service")
    @classmethod
    def validate_service(cls, value: str) -> str:
        if not value:
            raise ValueError("Назва сервісу обов'язкова")
        return value.lower()

    @field_validator("limit")
    @classmethod
    def validate_limit(cls, value: int) -> int:
        if not 1 <= value <= 10:
            raise ValueError("Ліміт повинен бути від 1 до 10")
        return value


@tool(args_schema=DeploymentHistoryInput)
def get_deployment_history(service: str, limit: int = 3) -> str:
    """Показати історію останніх розгортань сервісу.

    Використовуй для аналізу попередніх успішних і невдалих
    deployment-подій та оцінки ризиків релізу.
    """
    if service not in DEPLOYMENTS:
        return to_json({"error": f"Невідомий сервіс: {service}"})
    return to_json({
        "service": service,
        "deployments": DEPLOYMENTS[service][:limit],
    })


class ReleasePolicyInput(BaseModel):
    """Метрики для перевірки політики релізу."""

    model_config = ConfigDict(str_strip_whitespace=True)

    service: str = Field(
        description="Назва сервісу, для якого перевіряється реліз"
    )
    status: str = Field(
        description="Статус із get_service_status: healthy або degraded"
    )
    failed_tests: int = Field(
        ge=0,
        description="Кількість failed із get_test_results"
    )
    coverage_pct: float = Field(
        description="Відсоток coverage_pct із get_test_results"
    )

    @field_validator("service")
    @classmethod
    def validate_service(cls, value: str) -> str:
        if not value:
            raise ValueError("Назва сервісу обов'язкова")
        return value.lower()

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value not in {"healthy", "degraded"}:
            raise ValueError("Статус має бути healthy або degraded")
        return value

    @field_validator("coverage_pct")
    @classmethod
    def validate_coverage(cls, value: float) -> float:
        if not 0 <= value <= 100:
            raise ValueError("Coverage має бути від 0 до 100")
        return value


@tool(args_schema=ReleasePolicyInput)
def check_release_policy(
    service: str,
    status: str,
    failed_tests: int,
    coverage_pct: float,
) -> str:
    """Перевірити відповідність релізу правилам якості.

    Використовуй після get_service_status і get_test_results,
    передаючи фактично отримані метрики, а не припущення.

    Правила: healthy, нуль провалених тестів,
    покриття тестами не менше 90%.
    """
    reasons = []

    if status != "healthy":
        reasons.append("Сервіс не має статусу healthy")
    if failed_tests > 0:
        reasons.append("Є провалені тести")
    if coverage_pct < 90:
        reasons.append("Покриття тестами менше 90%")

    return to_json({
        "service": service,
        "approved": len(reasons) == 0,
        "reasons": reasons,
    })


TOOLS = [
    get_service_status,
    get_test_results,
    get_deployment_history,
    check_release_policy,
]