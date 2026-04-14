"""Test suite for everest_optimizers.minimize() with method='optpp_q_nips'.

Testing the OptQNIPS (Quasi-Newton Interior-Point Solver) method from
everest_optimizers.minimize(). In Dakota OPTPP this optimization algorithm is
referred to as OptQNIPS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize as sp_optimize
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

from everest_optimizers import minimize

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


def test_kitchen_sink() -> None:
    """A 'kitchen sink' test with bounds, linear equality, and inequality constraints."""
    bounds = Bounds((0.0, -np.inf), (1.5, np.inf))
    A_eq = np.array([[1, 1]])  # noqa: N806
    b_eq = np.array([1])
    A_ineq = np.array([[1, 0]])  # noqa: N806
    lb_ineq = np.array([-np.inf])
    ub_ineq = np.array([1.5])
    constraints: list[LinearConstraint | NonlinearConstraint] = [
        LinearConstraint(A_eq, b_eq, b_eq),
        LinearConstraint(A_ineq, lb_ineq, ub_ineq),
    ]
    x0 = np.array([0.0, 0.0])

    res_everest = minimize(
        objective,
        x0,
        method="optpp_q_nips",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options=DEFAULT_OPTIONS,
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(
        objective,
        x0,
        method="SLSQP",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
    )
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-4, atol=1e-4)


def test_active_and_inactive_constraints() -> None:
    """Test with a mix of active and inactive constraints at the solution."""
    bounds = Bounds((0.0, 0.0), (3.0, 3.0))
    constraints = LinearConstraint(np.array([[1, 1]]), lb=[4], ub=[np.inf])
    x0 = np.array([1.0, 1.0])

    res_everest = minimize(
        objective,
        x0,
        method="optpp_q_nips",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options=DEFAULT_OPTIONS,
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(
        objective,
        x0,
        method="SLSQP",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
    )
    assert res_scipy.success

    # The solution should be on the boundary of the linear constraint
    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.sum(res_everest.x), 4.0, atol=1e-5)


def test_redundant_constraints() -> None:
    """Test with redundant constraints that should not affect the outcome."""
    bounds = Bounds((0.0, 0.0), (3.0, 3.0))
    constraints: list[LinearConstraint | NonlinearConstraint] = [
        LinearConstraint(np.array([[1.0, 0.0]]), ub=[2.5]),
        LinearConstraint(np.array([[1.0, 0.0]]), ub=[3.0]),  # Redundant
    ]
    x0 = np.array([1.0, 1.0])

    res_everest = minimize(
        objective,
        x0,
        method="optpp_q_nips",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
        options=DEFAULT_OPTIONS,
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(
        objective,
        x0,
        method="SLSQP",
        jac=objective_grad,
        bounds=bounds,
        constraints=constraints,
    )
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-4, atol=1e-4)


def test_higher_dimensions() -> None:
    """Test a problem with 10 dimensions."""

    def high_dim_obj(x: NDArray[np.float64]) -> float:
        return np.sum((x - np.arange(10)) ** 2)

    def high_dim_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2 * (x - np.arange(10))

    x0 = np.zeros(10)
    bounds = Bounds(np.full(10, -5), np.full(10, 5))
    constraints = LinearConstraint(np.ones((1, 10)), lb=[10], ub=[10])

    res_everest = minimize(
        high_dim_obj,
        x0,
        method="optpp_q_nips",
        jac=high_dim_grad,
        bounds=bounds,
        constraints=constraints,
        options={"max_iterations": 500},
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(
        high_dim_obj,
        x0,
        method="SLSQP",
        jac=high_dim_grad,
        bounds=bounds,
        constraints=constraints,
    )
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-3, atol=1e-3)
