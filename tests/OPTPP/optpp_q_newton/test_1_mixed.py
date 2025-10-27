"""Test suite for everest_optimizers.minimize() with method='optpp_q_newton'

Testing the OptQNewton (Quasi-Newton Solver) method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptQNewton.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import optimize as sp_optimize
from scipy.optimize import Bounds, LinearConstraint

from everest_optimizers import minimize

DEFAULT_OPTIONS = {
    "debug": False,
    "max_iterations": 200,
    "convergence_tolerance": 1e-6,
    "gradient_tolerance": 1e-6,
}


def objective(x: NDArray[np.float64]) -> float:
    return (x[0] - 2.0) ** 2 + (x[1] + 1.0) ** 2


def objective_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([2 * (x[0] - 2.0), 2 * (x[1] + 1.0)])


def test_linear_equality_constraint():
    A = np.array([[1, 1]])
    lb = ub = np.array([1])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    with pytest.raises(
        NotImplementedError, match="optpp_q_newton does not support constraints"
    ):
        minimize(
            objective,
            x0,
            method="optpp_q_newton",
            jac=objective_grad,
            constraints=constraints,
            options=DEFAULT_OPTIONS,
        )


def test_linear_inequality_constraint():
    A = np.array([[1, 1]])
    lb = np.array([-np.inf])
    ub = np.array([1])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    with pytest.raises(
        NotImplementedError, match="optpp_q_newton does not support constraints"
    ):
        minimize(
            objective,
            x0,
            method="optpp_q_newton",
            jac=objective_grad,
            constraints=constraints,
            options=DEFAULT_OPTIONS,
        )


def test_mixed_bounds():
    bounds = Bounds([2.5, -np.inf], [np.inf, -1.5])
    x0 = np.array([3.0, -2.0])
    with pytest.raises(
        NotImplementedError, match="optpp_q_newton does not support bounds"
    ):
        minimize(
            objective,
            x0,
            method="optpp_q_newton",
            jac=objective_grad,
            bounds=bounds,
            options=DEFAULT_OPTIONS,
        )


def test_bounds_and_linear_equality():
    bounds = Bounds([0, -np.inf], [np.inf, np.inf])
    A = np.array([[1, 1]])
    lb = ub = np.array([1])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    with pytest.raises(
        NotImplementedError, match="optpp_q_newton does not support bounds"
    ):
        minimize(
            objective,
            x0,
            method="optpp_q_newton",
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options=DEFAULT_OPTIONS,
        )


def test_bounds_and_linear_inequality():
    bounds = Bounds([-np.inf, -np.inf], [1.5, np.inf])
    A = np.array([[1, 1]])
    lb = np.array([1])
    ub = np.array([np.inf])
    constraints = LinearConstraint(A, lb, ub)
    x0 = np.array([0.0, 0.0])
    with pytest.raises(
        NotImplementedError, match="optpp_q_newton does not support bounds"
    ):
        minimize(
            objective,
            x0,
            method="optpp_q_newton",
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options=DEFAULT_OPTIONS,
        )


@pytest.mark.xfail(
    reason="Graceful failure handling for unbounded problems is not implemented"
)
def test_unbounded_problem():
    """Test an unbounded problem, expecting a failure or specific status."""

    def unbounded_obj(x):
        return -x[0] - x[1]

    def unbounded_grad(x):
        return np.array([-1.0, -1.0])

    x0 = np.array([1.0, 1.0])

    result = minimize(
        unbounded_obj,
        x0,
        method="optpp_q_newton",
        jac=unbounded_grad,
        options=DEFAULT_OPTIONS,
    )
    assert not result.success


def test_higher_dimensions_unconstrained():
    """Test a problem with 10 dimensions."""

    def high_dim_obj(x):
        return np.sum((x - np.arange(10)) ** 2)

    def high_dim_grad(x):
        return 2 * (x - np.arange(10))

    x0 = np.zeros(10)

    res_everest = minimize(
        high_dim_obj,
        x0,
        method="optpp_q_newton",
        jac=high_dim_grad,
        options={"max_iterations": 500},
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(high_dim_obj, x0, method="BFGS", jac=high_dim_grad)
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-3, atol=1e-3)
