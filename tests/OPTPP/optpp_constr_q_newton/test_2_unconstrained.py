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
    with pytest.raises(ValueError, match="Either bounds or constraints must be provided for constrained optimization"):
        minimize(
            rosenbrock_obj,
            x0,
            method='optpp_constr_q_newton',
            jac=rosenbrock_grad,
            options=DEFAULT_OPTIONS
        )
