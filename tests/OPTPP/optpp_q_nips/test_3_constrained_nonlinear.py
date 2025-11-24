"""Test suite for everest_optimizers.minimize() with method='optpp_q_nips'

Testing the OptQNIPS (Quasi-Newton Interior-Point Solver) method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptQNIPS.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, NonlinearConstraint

from everest_optimizers import minimize

DEFAULT_OPTIONS = {
    "debug": False,
    "max_iterations": 300,
    "convergence_tolerance": 1e-5,
    "gradient_tolerance": 1e-5,
    "constraint_tolerance": 1e-6,
}


def objective(x):
    return (x[0] - 2) ** 2 + (x[1] - 2) ** 2


def objective_grad(x):
    return np.array([2 * (x[0] - 2), 2 * (x[1] - 2)])


def test_nonlinear_equality_constraint():
    def nonlinear_constraint(x):
        return np.array([x[0] ** 2 + x[1] ** 2 - 4.0])

    def nonlinear_constraint_jac(x):
        return np.array([[2 * x[0], 2 * x[1]]])

    x0 = np.array([1.5, 1.5])
    bounds = Bounds([-5.0, -5.0], [5.0, 5.0])
    constraint = NonlinearConstraint(
        nonlinear_constraint, 0.0, 0.0, jac=nonlinear_constraint_jac
    )
    result = minimize(
        objective,
        x0,
        method="optpp_q_nips",
        jac=objective_grad,
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


def test_nonlinear_inequality_constraint():
    def nonlinear_constraint(x):
        return np.array([1.0 - x[0] ** 2 - x[1] ** 2])

    def nonlinear_constraint_jac(x):
        return np.array([[-2 * x[0], -2 * x[1]]])

    x0 = np.array([0.0, 0.0])
    bounds = Bounds([-2.0, -2.0], [2.0, 2.0])
    constraint = NonlinearConstraint(
        nonlinear_constraint, 0.0, np.inf, jac=nonlinear_constraint_jac
    )
    result = minimize(
        objective,
        x0,
        method="optpp_q_nips",
        jac=objective_grad,
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
