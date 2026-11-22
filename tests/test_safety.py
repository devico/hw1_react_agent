
"""Unit-тести захисних механізмів ReAct-агента."""

import pytest

from safety import (
    LoopDetector,
    SafetyConfig,
    SafetyController,
    SafetyStop,
)


def test_max_steps():
    """Агент зупиняється після досягнення ліміту."""

    guard = SafetyController(
        SafetyConfig(max_steps=2)
    )

    guard.before_agent(completed_steps=0)
    guard.before_agent(completed_steps=1)

    with pytest.raises(SafetyStop) as error:
        guard.before_agent(completed_steps=2)

    assert error.value.reason == "max_steps"


def test_timeout():
    """Тайм-аут визначається без реального очікування."""

    current_time = [100.0]

    guard = SafetyController(
        SafetyConfig(timeout_seconds=5),
        clock=lambda: current_time[0],
    )

    current_time[0] = 104.0
    guard.check_timeout()

    current_time[0] = 105.0

    with pytest.raises(SafetyStop) as error:
        guard.check_timeout()

    assert error.value.reason == "timeout"


def test_timeout_before_tool():
    """Прострочений виклик інструмента блокується."""

    current_time = [10.0]

    guard = SafetyController(
        SafetyConfig(timeout_seconds=3),
        clock=lambda: current_time[0],
    )

    current_time[0] = 14.0

    with pytest.raises(SafetyStop) as error:
        guard.before_tool(
            "get_service_status",
            {"service": "api"},
        )

    assert error.value.reason == "timeout"


def test_loop_detection():
    """Третій однаковий виклик виявляється як цикл."""

    detector = LoopDetector(max_repeats=3)

    args = {"service": "api"}

    assert detector.check("get_service_status", args) is False
    assert detector.check("get_service_status", args) is False
    assert detector.check("get_service_status", args) is True


def test_loop_reset():
    """Зміна аргументів скидає лічильник повторень."""

    detector = LoopDetector(max_repeats=3)

    detector.check("get_service_status", {"service": "api"})
    detector.check("get_service_status", {"service": "api"})

    assert detector.check(
        "get_service_status",
        {"service": "payments"},
    ) is False

    assert detector.repeat_count == 1


def test_normalized_arguments():
    """Порядок ключів не змінює ідентичність виклику."""

    detector = LoopDetector(max_repeats=2)

    first = {"service": "api", "version": "1.4.0"}
    second = {"version": "1.4.0", "service": "api"}

    assert detector.check("get_test_results", first) is False
    assert detector.check("get_test_results", second) is True


def test_controller_loop_protection():
    """SafetyController блокує повторюваний tool call."""

    guard = SafetyController(
        SafetyConfig(max_repeats=3)
    )

    for _ in range(2):
        guard.before_tool(
            "get_service_status",
            {"service": "api"},
        )

    with pytest.raises(SafetyStop) as error:
        guard.before_tool(
            "get_service_status",
            {"service": "api"},
        )

    assert error.value.reason == "loop_detected"


@pytest.mark.parametrize(
    "invalid_config",
    [
        {"max_steps": 0},
        {"timeout_seconds": 0},
        {"max_repeats": 1},
    ],
)
def test_invalid_config(invalid_config):
    """Неприпустима конфігурація відхиляється."""

    with pytest.raises(ValueError):
        SafetyConfig(**invalid_config)
