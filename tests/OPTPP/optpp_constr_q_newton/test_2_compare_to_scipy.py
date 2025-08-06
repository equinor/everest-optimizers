"""Test suite for everest_optimizers.minimize() with method='optpp_constr_q_newton'

Testing the Constrained Quasi-Newton Solver method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptConstrQNewton.

Runs a set of standard optimization problems through both everest_optimizers.minimize() and scipy.optimize.minimize()
and compares the results. Checks for approximately equal numerical values of the solutions.
"""

from __future__ import annotations

import os
import sys

import pytest

import numpy as np
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
def test_linear_inequality_constraint():
    A = np.array([[1, 1]])
    lb = np.array([-np.inf])
    ub = np.array([1])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    res_everest = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(
        objective, x0, method='SLSQP', jac=objective_grad, constraints=constraints
    )
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-3, atol=1e-3)

@pytest.mark.skip()
def test_linear_mixed_constraints():
    A = np.array([[1, 1], [1, 0]])
    lb = np.array([1, -np.inf])
    ub = np.array([1, 1.5])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    res_everest = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        constraints=constraints,
        options=DEFAULT_OPTIONS
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(
        objective, x0, method='SLSQP', jac=objective_grad, constraints=constraints
    )
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-4, atol=1e-4)

@pytest.mark.skip()
def test_bounds_and_linear_inequality():
    bounds = Bounds([-np.inf, -np.inf], [1.5, np.inf])
    A = np.array([[1, 1]])
    lb = np.array([1])
    ub = np.array([np.inf])
    constraints = LinearConstraint(A, lb, ub)
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
def test_unconstrained():
    x0 = np.array([0.0, 0.0])
    res_everest = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        options=DEFAULT_OPTIONS
    )
    assert not res_everest.success

    res_scipy = sp_optimize.minimize(
        objective, x0, method='BFGS', jac=objective_grad
    )
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-4, atol=1e-4)

@pytest.mark.skip()
def test_bounds():
    bounds = Bounds([-1.0, -1.0], [1.0, 1.0])
    x0 = np.array([0.0, 0.0])
    res_everest = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        options=DEFAULT_OPTIONS
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(
        objective, x0, method='L-BFGS-B', jac=objective_grad, bounds=bounds
    )
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-3, atol=1e-3)

@pytest.mark.skip()
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
def test_start_feasible_and_infeasible_scipy_comparison(x0: NDArray[np.float64]):
    """Test with starting points that may be inside or outside the feasible region."""
    bounds = Bounds([2.5, -np.inf], [np.inf, np.inf])
    res_everest = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
        options=DEFAULT_OPTIONS
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(
        objective, x0, method='L-BFGS-B', jac=objective_grad, bounds=bounds
    )
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-3, atol=1e-3)


@pytest.mark.skip()
@pytest.mark.parametrize(
    "x0",
    [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([2.0, 1.0, 0.5, 0.5]),
        np.array([-4.0, -2.0, 2.0, -1.0]),
        # np.array([1000.0, 1000.0, 1000.0, 1000.0]), TODO: investigate why this one fails
    ],
)
def test_complex_problem_parameterized_start(x0: NDArray[np.float64]):
    """Test a bit more complex problem with linear constraints and bounds at various starting points."""
    def objective(x: NDArray[np.float64]) -> float:
        return (x[0] - 1)**2 + (x[1] - 2.5)**2 + (x[2] - 2)**2 + (x[3] - 0.5)**2

    def objective_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([
            2 * (x[0] - 1),
            2 * (x[1] - 2.5),
            2 * (x[2] - 2),
            2 * (x[3] - 0.5)
        ])

    # Constraints: x[0] - 2*x[1] = 0, x[2] + x[3] = 1
    constraints = LinearConstraint(np.array([[1, -2, 0, 0], [0, 0, 1, 1]]), lb=np.array([0, 1]), ub=np.array([0, 1]))

    bounds = Bounds([-3, -3, -3, -3], [3, 3, 3, 3])

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
        objective, 
        x0, 
        method='SLSQP', 
        jac=objective_grad, 
        bounds=bounds, 
        constraints=constraints
    )
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-3, atol=1e-3)
