"""Test suite for everest_optimizers.minimize() with method='optpp_constr_q_newton'

Testing the Constrained Quasi-Newton Solver method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptConstrQNewton.
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

@pytest.mark.skip()
def test_kitchen_sink():
    """A 'kitchen sink' test with bounds, linear equality, and inequality constraints."""
    bounds = Bounds([0, -np.inf], [1.5, np.inf])
    A_eq = np.array([[1, 1]])
    b_eq = np.array([1])
    A_ineq = np.array([[1, 0]])
    lb_ineq = np.array([-np.inf])
    ub_ineq = np.array([1.5])
    constraints = [
        LinearConstraint(A_eq, b_eq, b_eq),
        LinearConstraint(A_ineq, lb_ineq, ub_ineq)
    ]
    x0 = np.array([0.0, 0.0])

    res_everest = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(
        objective, x0, method='SLSQP', jac=objective_grad, bounds=bounds, constraints=constraints
    )
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-4, atol=1e-4)

@pytest.mark.skip()
def test_active_and_inactive_constraints():
    """Test with a mix of active and inactive constraints at the solution."""
    bounds = Bounds([0, 0], [3, 3])
    constraints = LinearConstraint(np.array([[1, 1]]), lb=[4], ub=[np.inf])
    x0 = np.array([1.0, 1.0])

    res_everest = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(
        objective, x0, method='SLSQP', jac=objective_grad, bounds=bounds, constraints=constraints
    )
    assert res_scipy.success

    # The solution should be on the boundary of the linear constraint
    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.sum(res_everest.x), 4.0, atol=1e-5)

@pytest.mark.skip()
def test_redundant_constraints():
    """Test with redundant constraints that should not affect the outcome."""
    bounds = Bounds([0, 0], [3, 3])
    constraints = [
        LinearConstraint(np.array([[1, 0]]), ub=[2.5]),
        LinearConstraint(np.array([[1, 0]]), ub=[3.0]) # Redundant
    ]
    x0 = np.array([1.0, 1.0])

    res_everest = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(
        objective, x0, method='SLSQP', jac=objective_grad, bounds=bounds, constraints=constraints
    )
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-4, atol=1e-4)

@pytest.mark.skip()
def test_infeasible_problem():
    """Test an infeasible problem, expecting a failure."""
    bounds = Bounds([2, 2], [3, 3])
    constraints = LinearConstraint(np.array([[1, 1]]), ub=[3]) # Contradicts bounds
    x0 = np.array([2.5, 2.5])

    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert not result.success


@pytest.mark.skip()
def test_unbounded_problem():
    """Test an unbounded problem, expecting a failure or specific status."""
    def unbounded_obj(x): return -x[0] - x[1]
    def unbounded_grad(x): return np.array([-1.0, -1.0])

    bounds = Bounds([0, -np.inf], [np.inf, np.inf])
    x0 = np.array([1.0, 1.0])

    result = minimize(
        unbounded_obj,
        x0,
        method='optpp_constr_q_newton',
        jac=unbounded_grad,
        bounds=bounds,
        options=DEFAULT_OPTIONS
    )
    assert not result.success

@pytest.mark.skip()
def test_higher_dimensions():
    """Test a problem with 10 dimensions."""
    def high_dim_obj(x): return np.sum((x - np.arange(10))**2)
    def high_dim_grad(x): return 2 * (x - np.arange(10))

    x0 = np.zeros(10)
    bounds = Bounds(np.full(10, -5), np.full(10, 5))
    constraints = LinearConstraint(np.ones((1, 10)), lb=[10], ub=[10])

    res_everest = minimize(
        high_dim_obj,
        x0,
        method='optpp_constr_q_newton',
        jac=high_dim_grad,
        bounds=bounds,
        constraints=constraints,
        options={'max_iterations': 500}
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(
        high_dim_obj, x0, method='SLSQP', jac=high_dim_grad, bounds=bounds, constraints=constraints
    )
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-3, atol=1e-3)
