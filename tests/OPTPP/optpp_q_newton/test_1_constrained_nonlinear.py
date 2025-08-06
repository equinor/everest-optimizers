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
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from everest_optimizers import minimize

DEFAULT_OPTIONS = {
    'debug': False,
    'max_iterations': 300,
    'convergence_tolerance': 1e-5,
    'gradient_tolerance': 1e-5,
    'constraint_tolerance': 1e-6,
}

def objective(x):
    return (x[0] - 2)**2 + (x[1] - 2)**2

def objective_grad(x):
    return np.array([2*(x[0] - 2), 2*(x[1] - 2)])

def test_nonlinear_equality_constraint():
    def nonlinear_constraint(x):
        return np.array([x[0]**2 + x[1]**2 - 4.0])

    def nonlinear_constraint_jac(x):
        return np.array([[2*x[0], 2*x[1]]])

    x0 = np.array([1.5, 1.5])
    constraint = NonlinearConstraint(
        nonlinear_constraint, 0.0, 0.0, jac=nonlinear_constraint_jac
    )

    with pytest.raises(NotImplementedError, match="optpp_q_newton does not support constraints"):
        minimize(
            objective,
            x0,
            method='optpp_q_newton',
            jac=objective_grad,
            constraints=constraint,
            options=DEFAULT_OPTIONS
        )
    
def test_nonlinear_inequality_constraint():
    def nonlinear_constraint(x):
        return np.array([1.0 - x[0]**2 - x[1]**2])

    def nonlinear_constraint_jac(x):
        return np.array([[-2*x[0], -2*x[1]]])

    x0 = np.array([0.0, 0.0])
    constraint = NonlinearConstraint(
        nonlinear_constraint, 0.0, np.inf, jac=nonlinear_constraint_jac
    )
    with pytest.raises(NotImplementedError, match="optpp_q_newton does not support constraints"):
        minimize(
            objective,
            x0,
            method='optpp_q_newton',
            jac=objective_grad,
            constraints=constraint,
            options=DEFAULT_OPTIONS
        )
