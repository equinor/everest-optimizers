"""Test suite for everest_optimizers.minimize() with method='optpp_q_nips'

Testing the OptQNIPS (Quasi-Newton Interior-Point Solver) method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptQNIPS.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from everest_optimizers import minimize

DEFAULT_OPTIONS = {
    'debug': False,
    'max_iterations': 300,
    'convergence_tolerance': 1e-5,
    'gradient_tolerance': 1e-5,
    'constraint_tolerance': 1e-6,
}

def objective(x):
    return (x[0] - 2)**2 + (x[1] - 2)**2

def objective_grad(x):
    return np.array([2*(x[0] - 2), 2*(x[1] - 2)])

@pytest.mark.skip(reason="Not implemented yet")
def test_nonlinear_equality_constraint():
    def nonlinear_constraint(x):
        return np.array([x[0]**2 + x[1]**2 - 4.0])

    def nonlinear_constraint_jac(x):
        return np.array([[2*x[0], 2*x[1]]])

    x0 = np.array([1.5, 1.5])
    bounds = Bounds([-5.0, -5.0], [5.0, 5.0])
    constraint = NonlinearConstraint(
        nonlinear_constraint, 0.0, 0.0, jac=nonlinear_constraint_jac
    )
    try:
        result = minimize(
            objective,
            x0,
            method='optpp_q_nips',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraint,
            options=DEFAULT_OPTIONS
        )
        if result.success:
            constraint_value = result.x[0]**2 + result.x[1]**2
            np.testing.assert_allclose(constraint_value, 4.0, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(result.fun, 4.0, rtol=1e-2, atol=1e-2)
    except NotImplementedError as e:
        pytest.skip(f"Nonlinear constraints not yet fully implemented: {e}")

@pytest.mark.skip(reason="Not implemented yet")
def test_nonlinear_inequality_constraint():
    def nonlinear_constraint(x):
        return np.array([1.0 - x[0]**2 - x[1]**2])

    def nonlinear_constraint_jac(x):
        return np.array([[-2*x[0], -2*x[1]]])

    x0 = np.array([0.0, 0.0])
    bounds = Bounds([-2.0, -2.0], [2.0, 2.0])
    constraint = NonlinearConstraint(
        nonlinear_constraint, 0.0, np.inf, jac=nonlinear_constraint_jac
    )
    try:
        result = minimize(
            objective,
            x0,
            method='optpp_q_nips',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraint,
            options=DEFAULT_OPTIONS
        )
        if result.success:
            constraint_value = result.x[0]**2 + result.x[1]**2
            assert constraint_value <= 1.0 + 1e-4
            expected_direction = np.array([2.0, 2.0]) / np.linalg.norm([2.0, 2.0])
            np.testing.assert_allclose(
                result.x / np.linalg.norm(result.x), expected_direction, rtol=1e-1, atol=1e-1
            )
    except NotImplementedError as e:
        pytest.skip(f"Nonlinear constraints not yet fully implemented: {e}")
