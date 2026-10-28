"""Test suite for everest_optimizers.minimize() with method='optpp_q_nips'

Testing the OptQNIPS (Quasi-Newton Interior-Point Solver) method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptQNIPS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.optimize import LinearConstraint

from everest_optimizers import minimize, pyoptpp
from everest_optimizers._convert_constraints import convert_linear_constraint

if TYPE_CHECKING:
    from numpy.typing import NDArray

DEFAULT_OPTIONS = {
    "debug": False,
    "max_iterations": 200,
    "convergence_tolerance": 1e-6,
    "gradient_tolerance": 1e-6,
}


def objective(x: NDArray[np.float64]) -> float:
    return float((x[0] - 2.0) ** 2 + (x[1] + 1.0) ** 2)


def objective_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([2 * (x[0] - 2.0), 2 * (x[1] + 1.0)])


@pytest.mark.parametrize(
    ("lower", "upper", "expected_types"),
    [
        ([0.0, 1.0], [0.0, 1.0], [pyoptpp.LinearEquation]),
        ([0.0, -np.inf], [np.inf, 1.0], [pyoptpp.LinearInequality]),
        (
            [0.0, -np.inf, 1.0, -np.inf, -np.inf],
            [0.0, 2.0, 1.0, 3.0, np.inf],
            [pyoptpp.LinearInequality, pyoptpp.LinearEquation],
        ),
        ([-np.inf, -np.inf], [np.inf, np.inf], []),
    ],
)
def test_linear_constraints_are_grouped(
    lower: list[float],
    upper: list[float],
    expected_types: list[type[pyoptpp.LinearEquation | pyoptpp.LinearInequality]],
) -> None:
    coefficients = np.arange(len(lower) * 3, dtype=float).reshape(len(lower), 3)
    constraints = convert_linear_constraint(
        LinearConstraint(coefficients, lower, upper)
    )
    assert [type(constraint) for constraint in constraints] == expected_types


def test_grouped_linear_constraints_optimize() -> None:
    coefficients = np.array(
        [[1, 1, 0], [1, -2, 0], [0, 0, 1], [2, 0, -1], [0, 0, 0]],
        dtype=float,
    )
    constraint = LinearConstraint(
        coefficients,
        [1, -np.inf, 0.5, -0.2, -np.inf],
        [1, -1, 0.5, 0.4, np.inf],
    )
    target = np.array([0.2, 0.8, 0.5])
    result = minimize(
        lambda x: float(np.sum((x - target) ** 2)),
        lambda x: 2 * (x - target),
        np.array([0.3, 0.7, 0.4]),
        method="optpp_q_nips",
        constraints=constraint,
        options=DEFAULT_OPTIONS.copy(),
    )
    assert result.success
    np.testing.assert_allclose(result.x, target, atol=1e-3)


def test_linear_equality_constraint() -> None:
    A = np.array([[1, 1]])  # ruff: ignore[non-lowercase-variable-in-function]
    lb = ub = np.array([1])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    result = minimize(
        objective,
        objective_grad,
        x0,
        method="optpp_q_nips",
        constraints=constraints,
        options=DEFAULT_OPTIONS,
    )
    assert result.success
    expected_solution = np.array([2.0, -1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)
    assert result.fun < 1e-8


def test_linear_inequality_constraint() -> None:
    A = np.array([[1, 1]])  # ruff: ignore[non-lowercase-variable-in-function]
    lb = np.array([-np.inf])
    ub = np.array([1])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    result = minimize(
        objective,
        objective_grad,
        x0,
        method="optpp_q_nips",
        constraints=constraints,
        options=DEFAULT_OPTIONS,
    )
    assert result.success
    expected_solution = np.array([2.0, -1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-3, atol=1e-3)
    assert result.fun < 1e-5


def test_linear_mixed_constraints() -> None:
    A = np.array([[1, 1], [1, 0]])  # ruff: ignore[non-lowercase-variable-in-function]
    lb = np.array([1, -np.inf])
    ub = np.array([1, 1.5])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    result = minimize(
        objective,
        objective_grad,
        x0,
        method="optpp_q_nips",
        constraints=constraints,
        options=DEFAULT_OPTIONS,
    )
    assert result.success
    expected_solution = np.array([1.5, -0.5])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)
