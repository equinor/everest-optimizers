"""Tests for expected solutions using everest_optimizers.minimize() with method='optpp_q_nips'

This test suite verifies that the different configuration options for the
'optpp_q_nips' solver are correctly handled.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from everest_optimizers import minimize

# --- Fixed Problem Definition ---
def objective(x: NDArray[np.float64]) -> float:
    return (x[0] - 2.0)**2 + (x[1] + 1.0)**2

def objective_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([2 * (x[0] - 2.0), 2 * (x[1] + 1.0)])

X0 = np.array([0.0, 0.0])
BOUNDS = Bounds([0, -np.inf], [np.inf, np.inf])
CONSTRAINTS = LinearConstraint(np.array([[1, 1]]), np.array([1]), np.array([1]))
EXPECTED_SOLUTION = np.array([2.0, -1.0])

# --- Tests for different options ---

@pytest.mark.parametrize("merit_function", ["argaez_tapia", "van_shanno", "norm_fmu"])
def test_merit_function_options(merit_function: str):
    """Test that the optimizer runs with different merit function settings."""
    options = {'merit_function': merit_function}
    result = minimize(
        objective, X0, method='optpp_q_nips', jac=objective_grad,
        bounds=BOUNDS, constraints=CONSTRAINTS, options=options
    )
    assert result.success
    np.testing.assert_allclose(result.x, EXPECTED_SOLUTION, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("search_strategy", ["trust_region", "line_search", "trust_pds"])
def test_search_strategy_options(search_strategy: str):
    """Test that the optimizer runs with different search strategy settings."""
    options = {'search_method': search_strategy}
    result = minimize(
        objective, X0, method='optpp_q_nips', jac=objective_grad,
        bounds=BOUNDS, constraints=CONSTRAINTS, options=options
    )
    assert result.success
    np.testing.assert_allclose(result.x, EXPECTED_SOLUTION, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("tolerance", [1e-4, 1e-6, 1e-8])
def test_convergence_tolerance_options(tolerance: float):
    """Test that the optimizer runs with different convergence tolerance settings."""
    options = {
        'convergence_tolerance': tolerance,
        'max_iterations': 100000
    }
    result = minimize(
        objective, X0, method='optpp_q_nips', jac=objective_grad,
        bounds=BOUNDS, constraints=CONSTRAINTS, options=options
    )
    assert result.success
    np.testing.assert_allclose(result.x, EXPECTED_SOLUTION, rtol=1e-3, atol=1e-3)

def test_high_convergence_tolerance_inaccurate():
    """Test that a high convergence tolerance leads to a numerically inaccurate solution."""
    options = {'convergence_tolerance': 1.0}
    result = minimize(
        objective, X0, method='optpp_q_nips', jac=objective_grad,
        bounds=BOUNDS, constraints=CONSTRAINTS, options=options
    )
    assert result.success
    assert not np.allclose(result.x, EXPECTED_SOLUTION, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("tolerance", [1e+2, 1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, -1e+1000, 1e+1000,]) # TODO: investigate if this tolerance parameter is handled correctly
def test_gradient_tolerance_options(tolerance: float):
    """Test that the optimizer runs with different gradient tolerance settings."""
    options = {
        'gradient_tolerance': tolerance,
        'max_iterations': 100000
    }
    result = minimize(
        objective, X0, method='optpp_q_nips', jac=objective_grad,
        bounds=BOUNDS, constraints=CONSTRAINTS, options=options
    )
    assert result.success
    np.testing.assert_allclose(result.x, EXPECTED_SOLUTION, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("max_iterations", [10, 100, 1000])
def test_max_iterations_option(max_iterations: int):
    """Test that the optimizer respects the max_iterations setting."""
    options = {'max_iterations': max_iterations}
    result = minimize(
        objective, X0, method='optpp_q_nips', jac=objective_grad,
        bounds=BOUNDS, constraints=CONSTRAINTS, options=options
    )
    assert result.success
    np.testing.assert_allclose(result.x, EXPECTED_SOLUTION, rtol=1e-3, atol=1e-3)

def test_too_low_max_iterations():
    """Too low max_iterations should mean it numerically does not converge"""
    options = {'max_iterations': 1}
    result = minimize(
        objective, X0, method='optpp_q_nips', jac=objective_grad,
        bounds=BOUNDS, constraints=CONSTRAINTS, options=options
    )
    assert result.success # algorithm terminates successfully 
    assert not np.allclose(result.x, EXPECTED_SOLUTION, rtol=1e-3, atol=1e-3) # but fails to converge to the expected solution


@pytest.mark.parametrize("debug_flag", [True, False])
def test_debug_option(debug_flag: bool):
    """Test that the optimizer runs with different debug flag settings."""
    options = {'debug': debug_flag}
    result = minimize(
        objective, X0, method='optpp_q_nips', jac=objective_grad,
        bounds=BOUNDS, constraints=CONSTRAINTS, options=options
    )
    assert result.success
    np.testing.assert_allclose(result.x, EXPECTED_SOLUTION, rtol=1e-4, atol=1e-4)



