"""Test suite for everest_optimizers.minimize() with method='optpp_q_nips'

Testing the OptQNIPS (Quasi-Newton Interior-Point Solver) method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptQNIPS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.optimize import Bounds, NonlinearConstraint

from everest_optimizers import minimize, pyoptpp
from everest_optimizers._convert_constraints import convert_nonlinear_constraint

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

DEFAULT_OPTIONS = {
    "debug": False,
    "max_iterations": 300,
    "convergence_tolerance": 1e-5,
    "gradient_tolerance": 1e-5,
    "constraint_tolerance": 1e-6,
}


def objective(x: NDArray[np.float64]) -> float:
    return float((x[0] - 2) ** 2 + (x[1] - 2) ** 2)


def objective_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([2 * (x[0] - 2), 2 * (x[1] - 2)])


def test_nonlinear_constraint_requires_analytic_jacobian() -> None:
    constraint = NonlinearConstraint(lambda x: np.array([x[0] ** 2]), 0.0, 1.0)
    with pytest.raises(TypeError, match="jac"):
        convert_nonlinear_constraint(constraint, np.array([0.5]))


def test_nonlinear_constraints_are_grouped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fun_calls = 0
    jac_calls = 0

    def constraint_fun(x: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal fun_calls
        fun_calls += 1
        return np.array([x[0] + x[1], x[0] - x[1], 2 * x[0], 3 * x[1], 0.0])

    def constraint_jac(_x: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal jac_calls
        jac_calls += 1
        return np.array([[1, 1], [1, -1], [2, 0], [0, 3], [0, 0]])

    callbacks: list[
        tuple[
            Callable[[NDArray[np.float64]], NDArray[np.float64]],
            Callable[[NDArray[np.float64]], NDArray[np.float64]],
        ]
    ] = []
    equations: list[tuple[NDArray[np.float64], int]] = []
    inequalities: list[tuple[NDArray[np.float64], NDArray[np.float64], int]] = []

    def make_nlf(
        ndim: int,
        fun: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        jac: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        initial: NDArray[np.float64],
        is_constraint: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
    ) -> int:
        assert ndim == 2
        assert is_constraint
        np.testing.assert_array_equal(initial, [1.0, 2.0])
        callbacks.append((fun, jac))
        return len(callbacks) - 1

    def make_equation(
        nlp: int, rhs: NDArray[np.float64], numconstraints: int = 1
    ) -> int:
        equations.append((rhs, numconstraints))
        return nlp

    def make_inequality(
        nlp: int,
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        numconstraints: int = 1,
    ) -> int:
        inequalities.append((lower, upper, numconstraints))
        return nlp

    monkeypatch.setattr(pyoptpp, "NLF1", make_nlf)
    monkeypatch.setattr(pyoptpp, "NLP", lambda value: value)
    monkeypatch.setattr(pyoptpp, "SerialDenseVector", np.atleast_1d)
    monkeypatch.setattr(pyoptpp, "NonLinearEquation", make_equation)
    monkeypatch.setattr(pyoptpp, "NonLinearInequality", make_inequality)

    x0 = np.array([1.0, 2.0])
    constraint = NonlinearConstraint(
        constraint_fun,
        [0.0, -np.inf, 1.0, -np.inf, -np.inf],
        [0.0, 2.0, 4.0, 9.0, np.inf],
        jac=constraint_jac,
    )
    result = convert_nonlinear_constraint(constraint, x0)

    assert len(result) == len(callbacks) == 2
    assert len(equations) == len(inequalities) == 1
    np.testing.assert_array_equal(equations[0][0], [0.0])
    assert equations[0][1] == 1
    np.testing.assert_array_equal(inequalities[0][0], [-np.inf, 1.0, -np.inf])
    np.testing.assert_array_equal(inequalities[0][1], [2.0, 4.0, 9.0])
    assert inequalities[0][2] == 3

    inequality_fun, inequality_jac = callbacks[0]
    equation_fun, equation_jac = callbacks[1]
    np.testing.assert_array_equal(equation_fun(x0), [3.0])
    np.testing.assert_array_equal(equation_jac(x0), [[1.0, 1.0]])
    np.testing.assert_array_equal(inequality_fun(x0), [-1.0, 2.0, 6.0])
    np.testing.assert_array_equal(
        inequality_jac(x0), [[1.0, -1.0], [2.0, 0.0], [0.0, 3.0]]
    )
    assert fun_calls == 1
    assert jac_calls == 1

    nearby_x = np.array([1.0 + 1e-10, 2.0])
    np.testing.assert_array_equal(equation_fun(nearby_x), [3.0 + 1e-10])
    np.testing.assert_array_equal(
        inequality_fun(nearby_x), [-1.0 + 1e-10, 2.0 + 2e-10, 6.0]
    )
    np.testing.assert_array_equal(equation_jac(nearby_x), [[1.0, 1.0]])
    np.testing.assert_array_equal(
        inequality_jac(nearby_x), [[1.0, -1.0], [2.0, 0.0], [0.0, 3.0]]
    )
    assert fun_calls == 2
    assert jac_calls == 2


def test_grouped_nonlinear_constraints_optimize() -> None:
    def fun(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([x[0] + x[1], x[2], x[0], x[1], x[0] - x[1], x[2] + 1])

    def jac(_x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array(
            [[1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, -1, 0], [0, 0, 1]]
        )

    def objective_three(x: NDArray[np.float64]) -> float:
        return float(np.sum((x - [0.2, 0.8, 0.5]) ** 2))

    def objective_three_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2 * (x - [0.2, 0.8, 0.5])

    constraint = NonlinearConstraint(
        fun,
        [1, 0.5, 0.1, -np.inf, -0.7, -np.inf],
        [1, 0.5, np.inf, 0.9, -0.5, np.inf],
        jac=jac,
    )
    result = minimize(
        objective_three,
        objective_three_grad,
        np.array([0.3, 0.7, 0.4]),
        method="optpp_q_nips",
        bounds=Bounds([-1, -1, -1], [2, 2, 2]),
        constraints=constraint,
        options=DEFAULT_OPTIONS.copy(),
    )
    assert result.success
    np.testing.assert_allclose(result.x, [0.2, 0.8, 0.5], atol=1e-3)


def test_nonlinear_equality_constraint() -> None:
    def nonlinear_constraint(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([x[0] ** 2 + x[1] ** 2 - 4.0])

    def nonlinear_constraint_jac(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([[2 * x[0], 2 * x[1]]])

    x0 = np.array([1.5, 1.5])
    bounds = Bounds((-5.0, -5.0), (5.0, 5.0))
    constraint = NonlinearConstraint(
        nonlinear_constraint, 0.0, 0.0, jac=nonlinear_constraint_jac
    )
    result = minimize(
        objective,
        objective_grad,
        x0,
        method="optpp_q_nips",
        bounds=bounds,
        constraints=constraint,
        options=DEFAULT_OPTIONS,
    )
    assert result.success

    constraint_value = result.x[0] ** 2 + result.x[1] ** 2
    np.testing.assert_allclose(constraint_value, 4.0, rtol=1e-3, atol=1e-3)
    # Optimal point on circle x^2 + y^2 = 4 closest to (2,2) is (sqrt(2), sqrt(2))
    # Objective value: f(sqrt(2), sqrt(2)) = 2*(sqrt(2) - 2)^2 ≈ 0.686
    np.testing.assert_allclose(result.fun, 0.686, rtol=1e-2, atol=1e-2)


def test_nonlinear_inequality_constraint() -> None:
    def nonlinear_constraint(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([1.0 - x[0] ** 2 - x[1] ** 2])

    def nonlinear_constraint_jac(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([[-2 * x[0], -2 * x[1]]])

    x0 = np.array([0.0, 0.0])
    bounds = Bounds((-2.0, -2.0), (2.0, 2.0))
    constraint = NonlinearConstraint(
        nonlinear_constraint, 0.0, np.inf, jac=nonlinear_constraint_jac
    )
    result = minimize(
        objective,
        objective_grad,
        x0,
        method="optpp_q_nips",
        bounds=bounds,
        constraints=constraint,
        options=DEFAULT_OPTIONS,
    )
    assert result.success

    constraint_value = result.x[0] ** 2 + result.x[1] ** 2
    assert constraint_value <= 1.0 + 1e-4
    expected_direction = np.array([2.0, 2.0]) / np.linalg.norm([2.0, 2.0])
    np.testing.assert_allclose(
        result.x / np.linalg.norm(result.x),
        expected_direction,
        rtol=1e-1,
        atol=1e-1,
    )
