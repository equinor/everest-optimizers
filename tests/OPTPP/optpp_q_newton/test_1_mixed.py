"""Test suite for everest_optimizers.minimize() with method='optpp_q_newton'

Testing the OptQNewton (Quasi-Newton Solver) method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptQNewton.

The tests here are intended to fail since the method optpp_q_newton is for purely unconstrained optimization problems.
"""

from __future__ import annotations

import os
import sys

import pytest

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import optimize as sp_optimize
from scipy.optimize import LinearConstraint, Bounds

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

@pytest.mark.xfail(reason="optpp_q_newton does not support neither bounds nor constraints")
def test_linear_equality_constraint():
    A = np.array([[1, 1]])
    lb = ub = np.array([1])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    result = minimize(
        objective,
        x0,
        method='optpp_q_newton',
        jac=objective_grad,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([2.0, -1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)
    assert result.fun < 1e-8

@pytest.mark.xfail(reason="optpp_q_newton does not support neither bounds nor constraints")
def test_linear_inequality_constraint():
    A = np.array([[1, 1]])
    lb = np.array([-np.inf])
    ub = np.array([1])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    result = minimize(
        objective,
        x0,
        method='optpp_q_newton',
        jac=objective_grad,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([2.0, -1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-3, atol=1e-3)

@pytest.mark.xfail(reason="optpp_q_newton does not support neither bounds nor constraints")
def test_mixed_bounds():
    bounds = Bounds([2.5, -np.inf], [np.inf, -1.5])
    x0 = np.array([3.0, -2.0])
    result = minimize(
        objective,
        x0,
        method='optpp_q_newton',
        jac=objective_grad,
        bounds=bounds,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([2.5, -1.5])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)

@pytest.mark.xfail(reason="optpp_q_newton does not support neither bounds nor constraints")
def test_bounds_and_linear_equality():
    bounds = Bounds([0, -np.inf], [np.inf, np.inf])
    A = np.array([[1, 1]])
    lb = ub = np.array([1])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    result = minimize(
        objective,
        x0,
        method='optpp_q_newton',
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([2.0, -1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)

@pytest.mark.xfail(reason="optpp_q_newton does not support neither bounds nor constraints")
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
        method='optpp_q_newton',
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([1.5, -0.5])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)

@pytest.mark.skip("Feature not implemented")
def test_unbounded_problem(): # TODO: implement detection of unbounded problems, return error
    """Test an unbounded problem, expecting a failure or specific status."""
    def unbounded_obj(x): return -x[0] - x[1]
    def unbounded_grad(x): return np.array([-1.0, -1.0])

    x0 = np.array([1.0, 1.0])

    result = minimize(
        unbounded_obj,
        x0,
        method='optpp_q_newton',
        jac=unbounded_grad,
        options=DEFAULT_OPTIONS
    )
    assert not result.success


def test_higher_dimensions_unconstrained():
    """Test a problem with 10 dimensions."""
    def high_dim_obj(x): return np.sum((x - np.arange(10))**2)
    def high_dim_grad(x): return 2 * (x - np.arange(10))

    x0 = np.zeros(10)

    res_everest = minimize(
        high_dim_obj,
        x0,
        method='optpp_q_newton',
        jac=high_dim_grad,
        options={'max_iterations': 500}
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(
        high_dim_obj, x0, method='BFGS', jac=high_dim_grad
    )
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-3, atol=1e-3)
