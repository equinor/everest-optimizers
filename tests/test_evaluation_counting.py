"""Tests for the OPT++ evaluation counters.

OPT++ counts each objective request that reaches the evaluator, including requests
that are answered from stored evaluations. The expected values are the counts
of the reference `optpp_q_newton` implementation on the same problems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from everest_optimizers import EvaluationMode, minimize
from everest_optimizers._problem import NLF1Problem

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

    from everest_optimizers import EvaluationResult
    from everest_optimizers._optpp import Bounds, LinearConstraints
    from everest_optimizers._problem import _EvaluateCallbackResult

    Problem = Callable[
        [NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.float64]]
    ]


@pytest.fixture
def work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _circle(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.array([np.sum((x - 2.0) ** 2), x @ x - 4.0])
    gradients = np.array([2 * (x - 2.0), 2 * x])
    return values, gradients


def _mixed(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    delta = x - 1.0
    values = np.array([delta @ delta, 0.5 - x[0] ** 2 - x[1] ** 2, np.sum(x) - 1.0])
    gradients = np.array(
        [2 * delta, [-2 * x[0], -2 * x[1], 0.0], [1.0, 1.0, 1.0]],
    )
    return values, gradients


def _evaluator(
    problem: Problem, calls: list[EvaluationMode]
) -> Callable[[NDArray[np.float64], EvaluationMode], EvaluationResult]:
    def evaluate(x: NDArray[np.float64], mode: EvaluationMode) -> EvaluationResult:
        calls.append(mode)
        values, gradients = problem(x)
        return (
            values if mode & EvaluationMode.VALUES else None,
            gradients if mode & EvaluationMode.GRADIENTS else None,
        )

    return evaluate


def _iteration_rows(output_file: Path) -> list[list[str]]:
    rows = (
        line.split() for line in output_file.read_text(encoding="utf-8").splitlines()
    )
    return [row for row in rows if len(row) > 4 and row[0].isdigit()]


def test_optpp_q_newton_counts_objective_requests_answered_by_the_constraint_evaluation(
    work_dir: Path,
) -> None:
    calls: list[EvaluationMode] = []
    output_file = work_dir / "optimizer.out"

    minimize(
        np.array([1.5, 1.5]),
        evaluate=_evaluator(_circle, calls),
        bounds=(np.full(2, -5.0), np.full(2, 5.0)),
        nonlinear_equalities=np.array([True]),
        options={"output_file": str(output_file)},
    )

    assert [int(row[-3]) for row in _iteration_rows(output_file)] == [2, 3, 4]
    assert len(calls) == 4


@pytest.mark.parametrize(
    ("problem", "x0", "nonlinear_equalities", "max_evaluations", "expected_calls"),
    [
        (_circle, [1.5, 1.5], [True], 2, 3),
        (_mixed, [0.0, 0.0, 0.0], [False, True], 2, 3),
        (_mixed, [0.0, 0.0, 0.0], [False, True], 3, 4),
        (_mixed, [0.0, 0.0, 0.0], [False, True], 4, 5),
    ],
)
@pytest.mark.usefixtures("work_dir")
def test_optpp_q_newton_stops_at_max_function_evaluations(
    problem: Problem,
    x0: list[float],
    nonlinear_equalities: list[bool],
    max_evaluations: int,
    expected_calls: int,
) -> None:
    calls: list[EvaluationMode] = []

    minimize(
        np.array(x0),
        evaluate=_evaluator(problem, calls),
        bounds=(np.full(len(x0), -5.0), np.full(len(x0), 5.0)),
        nonlinear_equalities=np.array(nonlinear_equalities),
        options={"max_function_evaluations": max_evaluations},
    )

    assert len(calls) == expected_calls


def _rosenbrock(
    x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.array([100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2])
    gradients = np.array(
        [[-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]]
    )
    return values, gradients


BOUNDS = (np.full(2, -5.0), np.full(2, 5.0))
LINEAR_CONSTRAINTS = (np.ones((1, 2)), np.array([-np.inf]), np.array([10.0]))


@pytest.mark.parametrize("speculative", [False, True])
@pytest.mark.parametrize(
    ("bounds", "linear_constraints", "search_method"),
    [
        (None, None, "trust_region"),
        (None, None, "value_based_line_search"),
        (None, None, "gradient_based_line_search"),
        (BOUNDS, None, "trust_region"),
        (BOUNDS, None, "value_based_line_search"),
        (BOUNDS, None, "gradient_based_line_search"),
        (BOUNDS, LINEAR_CONSTRAINTS, None),
    ],
)
def test_optpp_q_newton_max_function_evaluations_counts_objective_requests(
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    bounds: Bounds | None,
    linear_constraints: LinearConstraints | None,
    search_method: str | None,
    *,
    speculative: bool,
) -> None:
    objective_requests = 0
    gradient_requests = 0
    evaluate_request = NLF1Problem._evaluate  # ruff: ignore[private-member-access]

    def count_requests(
        problem: NLF1Problem, x: NDArray[np.float64], mode: int
    ) -> _EvaluateCallbackResult:
        nonlocal objective_requests, gradient_requests
        objective_requests += bool(mode & EvaluationMode.VALUES)
        gradient_requests += bool(mode & EvaluationMode.GRADIENTS)
        return evaluate_request(problem, x, mode)

    monkeypatch.setattr(NLF1Problem, "_evaluate", count_requests)
    output_file = work_dir / "optimizer.out"
    options: dict[str, Any] = {
        "max_function_evaluations": 5,
        "max_iterations": 20,
        "output_file": str(output_file),
    }
    if search_method is not None:
        options["search_method"] = search_method
    if speculative:
        options["speculative"] = True

    minimize(
        np.array([-1.2, 1.0]),
        evaluate=_evaluator(_rosenbrock, []),
        bounds=bounds,
        linear_constraints=linear_constraints,
        options=options,
    )

    rows = _iteration_rows(output_file)
    evaluation_column = -3 if linear_constraints is not None else -2
    assert int(rows[-1][evaluation_column]) == objective_requests
    assert objective_requests > 5
    assert int(rows[-1][0]) < 20
    if len(rows) > 1:
        assert int(rows[-2][evaluation_column]) <= 5
    if linear_constraints is None:
        assert int(rows[-1][-1]) == gradient_requests
