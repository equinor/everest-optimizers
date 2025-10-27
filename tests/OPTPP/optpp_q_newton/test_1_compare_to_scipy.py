"""Test suite for everest_optimizers.minimize() with method='optpp_q_newton'

Testing the OptQNewton (Quasi-Newton Solver) method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptQNewton.

Runs a set of standard optimization problems through both everest_optimizers.minimize() and scipy.optimize.minimize()
and compares the results. Checks for approximately equal numerical values of the solutions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as sp_optimize

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


def test_unconstrained():
    x0 = np.array([0.0, 0.0])
    res_everest = minimize(
        objective,
        x0,
        method="optpp_q_newton",
        jac=objective_grad,
        options=DEFAULT_OPTIONS,
    )
    assert res_everest.success

    res_scipy = sp_optimize.minimize(objective, x0, method="BFGS", jac=objective_grad)
    assert res_scipy.success

    np.testing.assert_allclose(res_everest.x, res_scipy.x, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res_everest.fun, res_scipy.fun, rtol=1e-4, atol=1e-4)
