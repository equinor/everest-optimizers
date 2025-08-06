"""Test suite for everest_optimizers.minimize() with method='optpp_q_newton'

Testing the OptQNewton (Quasi-Newton Solver) method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptQNewton.
"""


from __future__ import annotations

import os
import sys
import numpy as np
import pytest
from numpy.typing import NDArray


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

def test_rosenbrock_simple():
    def rosenbrock_obj(x: NDArray[np.float64]) -> float:
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

    def rosenbrock_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
        grad = np.zeros_like(x)
        grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        grad[1] = 200 * (x[1] - x[0]**2)
        return grad

    x0 = np.array([0.0, 0.0])
    result = minimize(
        rosenbrock_obj,
        x0,
        method='optpp_q_newton',
        jac=rosenbrock_grad,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    np.testing.assert_allclose(result.x, [1.0, 1.0], rtol=1e-4, atol=1e-4)
    assert result.fun < 1e-8

@pytest.mark.parametrize(
    "start_point",
    [
        np.array([10.0, 10.0]),
        np.array([-5.0, 8.0]),
        np.array([0.1, 0.9]),
        np.array([-2.0, -2.0]),
    ],
)
def test_quadratic_from_multiple_starts(start_point: NDArray[np.float64]):
    def objective(x: NDArray[np.float64]) -> float:
        return (x[0] - 2.0)**2 + (x[1] + 1.0)**2

    def objective_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.array([2 * (x[0] - 2.0), 2 * (x[1] + 1.0)])

    result = minimize(
        objective,
        start_point,
        method='optpp_q_newton',
        jac=objective_grad,
        options=DEFAULT_OPTIONS
    )
    assert result.success
    expected_solution = np.array([2.0, -1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-4, atol=1e-4)
    assert result.fun < 1e-8

