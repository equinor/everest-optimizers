"""Tests for the evaluation cache.

Every evaluation is kept for the whole run. A request is answered from a stored
evaluation when the point is exactly equal and that single evaluation contains
everything requested. Otherwise the evaluator is called for the full request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from everest_optimizers import EvaluationMode, minimize
from everest_optimizers._problem import NLF1Problem

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from everest_optimizers import EvaluationResult
    from everest_optimizers._problem import _EvaluateCallbackResult

    Call = tuple[tuple[float, ...], EvaluationMode]

VALUES = EvaluationMode.VALUES
GRADIENTS = EvaluationMode.GRADIENTS
BOTH = EvaluationMode.BOTH


@pytest.fixture
def work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _recording_problem() -> tuple[NLF1Problem, list[Call]]:
    calls: list[Call] = []

    def evaluate(x: NDArray[np.float64], mode: EvaluationMode) -> EvaluationResult:
        calls.append((tuple(x.tolist()), mode))
        values = np.array([x @ x, x[0] - 1.0]) if mode & VALUES else None
        gradients = np.array([2 * x, [1.0, 0.0]]) if mode & GRADIENTS else None
        return values, gradients

    problem = NLF1Problem(evaluate, np.zeros(2), np.array([False]))
    return problem, calls


def _request(
    problem: NLF1Problem, x: Sequence[float] | NDArray[np.float64], mode: EvaluationMode
) -> _EvaluateCallbackResult:
    return problem._evaluate_callback(np.asarray(x, dtype=np.float64), int(mode))  # ruff: ignore[private-member-access]


def _covered_requests(calls: list[Call]) -> list[Call]:
    earlier: dict[tuple[float, ...], list[EvaluationMode]] = {}
    covered: list[Call] = []
    for x, mode in calls:
        modes = earlier.setdefault(x, [])
        if any(previous & mode == mode for previous in modes):
            covered.append((x, mode))
        modes.append(mode)
    return covered


def test_evaluation_cache_answers_a_repeated_request_without_evaluating() -> None:
    problem, calls = _recording_problem()

    first = _request(problem, [1.0, 2.0], BOTH)
    second = _request(problem, [1.0, 2.0], BOTH)

    assert len(calls) == 1
    np.testing.assert_array_equal(second[0], first[0])
    np.testing.assert_array_equal(second[1], first[1])


def test_evaluation_cache_answers_a_part_of_a_stored_evaluation() -> None:
    problem, calls = _recording_problem()
    _request(problem, [1.0, 2.0], BOTH)

    values, no_gradients = _request(problem, [1.0, 2.0], VALUES)
    no_values, gradients = _request(problem, [1.0, 2.0], GRADIENTS)

    assert len(calls) == 1
    assert values is not None
    assert no_gradients is None
    assert no_values is None
    assert gradients is not None


def test_evaluation_cache_evaluates_the_full_request_when_no_evaluation_covers_it() -> (
    None
):
    problem, calls = _recording_problem()

    _request(problem, [1.0, 2.0], VALUES)
    _request(problem, [1.0, 2.0], BOTH)

    assert [mode for _, mode in calls] == [VALUES, BOTH]


def test_evaluation_cache_does_not_combine_stored_evaluations() -> None:
    problem, calls = _recording_problem()

    _request(problem, [1.0, 2.0], VALUES)
    _request(problem, [1.0, 2.0], GRADIENTS)
    _request(problem, [1.0, 2.0], BOTH)

    assert [mode for _, mode in calls] == [VALUES, GRADIENTS, BOTH]


def test_evaluation_cache_keeps_every_point() -> None:
    problem, calls = _recording_problem()
    for i in range(300):
        _request(problem, [float(i), 0.0], BOTH)

    _request(problem, [0.0, 0.0], BOTH)

    assert len(calls) == 300


def test_evaluation_cache_matches_points_exactly() -> None:
    problem, calls = _recording_problem()
    x = np.array([0.1, 0.2])

    _request(problem, x, BOTH)
    _request(problem, np.nextafter(x, np.inf), BOTH)

    assert len(calls) == 2


def _circle(
    calls: list[Call],
) -> Callable[[NDArray[np.float64], EvaluationMode], EvaluationResult]:
    def evaluate(x: NDArray[np.float64], mode: EvaluationMode) -> EvaluationResult:
        calls.append((tuple(x.tolist()), mode))
        values = (
            np.array([np.sum((x - 2.0) ** 2), x @ x - 4.0]) if mode & VALUES else None
        )
        gradients = np.array([2 * (x - 2.0), 2 * x]) if mode & GRADIENTS else None
        return values, gradients

    return evaluate


@pytest.mark.usefixtures("work_dir")
def test_optpp_q_newton_evaluates_each_point_once_with_nonlinear_constraints() -> None:
    calls: list[Call] = []

    _, status = minimize(
        np.array([1.5, 1.5]),
        evaluate=_circle(calls),
        bounds=(np.full(2, -5.0), np.full(2, 5.0)),
        nonlinear_equalities=np.array([True]),
    )

    assert status > 0
    assert len(calls) == len({x for x, _ in calls})


@pytest.mark.usefixtures("work_dir")
@pytest.mark.parametrize("options", [{}, {"speculative": True}])
def test_optpp_q_newton_never_repeats_a_covered_request_with_linear_constraints(
    options: dict[str, bool],
) -> None:
    calls: list[Call] = []

    def evaluate(x: NDArray[np.float64], mode: EvaluationMode) -> EvaluationResult:
        calls.append((tuple(x.tolist()), mode))
        delta = x - [2.0, -1.0]
        values = np.array([delta @ delta]) if mode & VALUES else None
        gradients = 2 * delta[np.newaxis, :] if mode & GRADIENTS else None
        return values, gradients

    _, status = minimize(
        np.zeros(2),
        evaluate=evaluate,
        bounds=(np.full(2, -5.0), np.full(2, 5.0)),
        linear_constraints=(np.ones((1, 2)), np.array([-np.inf]), np.array([0.5])),
        options=options,
    )

    assert status > 0
    assert _covered_requests(calls) == []


@pytest.mark.usefixtures("work_dir")
def test_optpp_q_newton_never_repeats_a_covered_request() -> None:
    calls: list[Call] = []

    def evaluate(x: NDArray[np.float64], mode: EvaluationMode) -> EvaluationResult:
        calls.append((tuple(x.tolist()), mode))
        values = (
            np.array([100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2])
            if mode & VALUES
            else None
        )
        gradients = (
            np.array(
                [
                    [
                        -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
                        200 * (x[1] - x[0] ** 2),
                    ]
                ]
            )
            if mode & GRADIENTS
            else None
        )
        return values, gradients

    _, status = minimize(np.array([-1.2, 1.0]), evaluate=evaluate)

    assert status > 0
    assert _covered_requests(calls) == []
