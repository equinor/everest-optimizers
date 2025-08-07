"""Test suite for everest_optimizers.minimize() with method='optpp_constr_q_newton'

Testing the Constrained Quasi-Newton Solver method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptConstrQNewton.
"""

from __future__ import annotations

import os
import sys

import pytest

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import LinearConstraint

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from everest_optimizers import minimize

DEFAULT_OPTIONS = {
    'debug': False,
    'max_iterations': 200,
    'convergence_tolerance': 1e-6,
    'gradient_tolerance': 1e-6,
}

def objective(x: NDArray[np.float64]) -> float:
    return (x[0] - 2.0)**2 + (x[1] + 1.0)**2

def objective_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([2 * (x[0] - 2.0), 2 * (x[1] + 1.0)])

def test_linear_equality_constraint():
    A = np.array([[1, 1]])
    lb = ub = np.array([1])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([2.0, -1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)
    assert result.fun < 1e-8

@pytest.mark.xfail(reason="Something is wrong in the implementation of constraints and bounds.")
def test_linear_inequality_constraint():
    A = np.array([[1, 1]])
    lb = np.array([-np.inf])
    ub = np.array([1])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([2.0, -1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-3, atol=1e-3)
    assert result.fun < 1e-5

@pytest.mark.xfail(reason="Something is wrong in the implementation of constraints and bounds.")
def test_linear_mixed_constraints():
    A = np.array([[1, 1], [1, 0]])
    lb = np.array([1, -np.inf])
    ub = np.array([1, 1.5])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([1.5, -0.5])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)
