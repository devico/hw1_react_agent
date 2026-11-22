"""Unit-тести інструментів та Pydantic-валідації."""

import json

import pytest
from pydantic import ValidationError

from tools import (
    ServiceStatusInput,
    TestResultsInput as ResultsInputSchema,
    DeploymentHistoryInput,
    ReleasePolicyInput,
    get_service_status,
    get_test_results,
    get_deployment_history,
    check_release_policy,
)


def test_service_status_success():
    result = json.loads(
        get_service_status.invoke({"service": "api"})
    )
    assert result["status"] == "healthy"


def test_service_status_invalid():
    with pytest.raises(ValidationError, match="порожньою"):
        ServiceStatusInput(service="   ")


def test_results_success():
    result = json.loads(get_test_results.invoke({
        "service": "api",
        "version": "1.4.0",
    }))
    assert result["passed"] == 120
    assert result["failed"] == 0


def test_results_invalid_version():
    with pytest.raises(ValidationError, match="X.Y.Z"):
        ResultsInputSchema(service="api", version="invalid")


def test_deployment_history_success():
    result = json.loads(get_deployment_history.invoke({
        "service": "payments",
        "limit": 2,
    }))
    assert len(result["deployments"]) == 2
    assert result["deployments"][0]["result"] == "failed"


def test_deployment_history_invalid_limit():
    with pytest.raises(ValidationError, match="від 1 до 10"):
        DeploymentHistoryInput(service="api", limit=0)


def test_release_policy_approved():
    result = json.loads(check_release_policy.invoke({
        "service": "api",
        "status": "healthy",
        "failed_tests": 0,
        "coverage_pct": 94,
    }))
    assert result["approved"] is True


def test_release_policy_rejected():
    result = json.loads(check_release_policy.invoke({
        "service": "payments",
        "status": "degraded",
        "failed_tests": 4,
        "coverage_pct": 87,
    }))
    assert result["approved"] is False
    assert len(result["reasons"]) == 3


def test_release_policy_invalid_coverage():
    with pytest.raises(ValidationError, match="від 0 до 100"):
        ReleasePolicyInput(
            service="api",
            status="healthy",
            failed_tests=0,
            coverage_pct=150,
        )


def test_unknown_service():
    result = json.loads(
        get_service_status.invoke({"service": "unknown"})
    )
    assert "error" in result