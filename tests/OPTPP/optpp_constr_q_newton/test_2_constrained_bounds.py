"""Test suite for everest_optimizers.minimize() with method='optpp_constr_q_newton'

Testing the Constrained Quasi-Newton Solver method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptConstrQNewton.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint

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

def test_lower_bounds_only():
    bounds = Bounds([2.5, -np.inf], [np.inf, np.inf])
    x0 = np.array([3.0, 0.0])
    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([2.5, -1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)

def test_upper_bounds_only():
    bounds = Bounds([-np.inf, -np.inf], [np.inf, -1.5])
    x0 = np.array([0.0, -2.0])
    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([2.0, -1.5])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)

def test_mixed_bounds():
    bounds = Bounds([2.5, -np.inf], [np.inf, -1.5])
    x0 = np.array([3.0, -2.0])
    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([2.5, -1.5])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)

def test_bounds_and_linear_equality():
    bounds = Bounds([0, -np.inf], [np.inf, np.inf])
    A = np.array([[1, 1]])
    lb = ub = np.array([1])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([2.0, -1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)

@pytest.mark.xfail(reason="Something is wrong in the implementation. Produced output does not match expected output.")
def test_bounds_and_linear_inequality():
    bounds = Bounds([-np.inf, -np.inf], [1.5, np.inf])
    A = np.array([[1, 1]])
    lb = np.array([1])
    ub = np.array([np.inf])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([1.5, -0.5])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)

def test_multiple_bounds():
    bounds = Bounds([2.5, -1.5], [3.0, -1.0])
    x0 = np.array([2.8, -1.2])
    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([2.5, -1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-3, atol=1e-3)

@pytest.mark.xfail(reason="Something is wrong in the implementation of constraints and bounds.")
def test_bounds_and_multiple_linear_constraints():
    bounds = Bounds([-np.inf, -np.inf], [1.5, np.inf])
    A = np.array([[1, 1], [1, -1]])
    lb = np.array([1, 0])
    ub = np.array([np.inf, 0])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([0.5, 0.5])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-3, atol=1e-3)



@pytest.mark.parametrize(
    "x0",
    [
        np.array([0.0, 0.0]),
        np.array([3.0, 0.0]),
        np.array([-10.0, -10.0]),
        np.array([2.4, -1.0]),
        np.array([10.0, 10.0]),
    ],
)
def test_start_feasible_and_infeasible(x0: NDArray[np.float64]):
    """Test with starting points that may be inside or outside the feasible region."""
    bounds = Bounds([2.5, -np.inf], [np.inf, np.inf])
    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([2.5, -1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)
