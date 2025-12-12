"""Test suite for everest_optimizers.minimize() with method='optpp_bcq_newton'

Testing the OptQNIPS (Quasi-Newton Interior-Point Solver) method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptQNIPS.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import Bounds

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


def test_lower_bounds_only():
    bounds = Bounds([2.5, -3], [3, 3])
    x0 = np.array([3.0, 0.0])
    result = minimize(
        objective,
        x0,
        method="optpp_bcq_newton",
        jac=objective_grad,
        bounds=bounds,
        options=DEFAULT_OPTIONS,
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
        method="optpp_bcq_newton",
        jac=objective_grad,
        bounds=bounds,
        options=DEFAULT_OPTIONS,
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
        method="optpp_bcq_newton",
        jac=objective_grad,
        bounds=bounds,
        options=DEFAULT_OPTIONS,
    )
    assert result.success
    expected_solution = np.array([2.5, -1.5])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)


def test_multiple_bounds():
    bounds = Bounds([2.5, -1.5], [3.0, -1.0])
    x0 = np.array([2.8, -1.2])
    result = minimize(
        objective,
        x0,
        method="optpp_bcq_newton",
        jac=objective_grad,
        bounds=bounds,
        options=DEFAULT_OPTIONS,
    )
    assert result.success
    expected_solution = np.array([2.5, -1.0])
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
        method="optpp_bcq_newton",
        jac=objective_grad,
        bounds=bounds,  # type: ignore[arg-type]
        options=DEFAULT_OPTIONS,
    )
    assert result.success
    expected_solution = np.array([2.5, -1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)
