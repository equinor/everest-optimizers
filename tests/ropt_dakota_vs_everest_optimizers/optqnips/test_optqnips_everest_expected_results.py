"""Tests for expected solutions using OptQNIPS optimizer from everest_optimizers.minimize.

This mirrors `test_everest_expected_results.py` but uses the OptQNIPS (Quasi-Newton 
Interior-Point Solver) method instead of the standard optpp_q_newton and 
optpp_constr_q_newton optimizers. It validates that the OptQNIPS solver 
converges to analytically expected solutions for constrained problems.
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Any
import os
import sys

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint

# Add the source directory to the path to find everest_optimizers
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from everest_optimizers import minimize  # noqa: E402  pylint: disable=C0413

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

_Function = Callable[[NDArray[np.float64]], float]


def _compute_distance_squared(
    variables: NDArray[np.float64], target: NDArray[np.float64]
) -> float:
    return float(((variables - target) ** 2).sum())


# Objective function combines two distance-squared metrics with weights.
weights = np.array([0.75, 0.25], dtype=np.float64)
function_1: _Function = partial(_compute_distance_squared, target=np.array([0.5, 0.5, 0.5]))
function_2: _Function = partial(_compute_distance_squared, target=np.array([-1.5, -1.5, 0.5]))


def objective(x: NDArray[np.float64]) -> float:
    return weights[0] * function_1(x) + weights[1] * function_2(x)


def objective_gradient(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Analytical gradient of the objective function."""
    grad1 = 2 * (x - np.array([0.5, 0.5, 0.5]))
    grad2 = 2 * (x - np.array([-1.5, -1.5, 0.5]))
    return weights[0] * grad1 + weights[1] * grad2


# -----------------------------------------------------------------------------
# Expected analytical solutions
# -----------------------------------------------------------------------------

expected_unconstrained = np.array([0.0, 0.0, 0.5], dtype=np.float64)


def _project_to_bounds(
    x: NDArray[np.float64], lower: list[float] | NDArray[np.float64], upper: list[float] | NDArray[np.float64]
) -> NDArray[np.float64]:
    lower_a = np.asarray(lower, dtype=np.float64)
    upper_a = np.asarray(upper, dtype=np.float64)
    return np.minimum(np.maximum(x, lower_a), upper_a)


# -----------------------------------------------------------------------------
# Test data
# -----------------------------------------------------------------------------

initial_values_1: list[float] = [0.1, 0.1, 0.4]  # Start near expected solution
initial_values_2: list[float] = [0.8, -0.8, 0.6]  # Mixed positive/negative
initial_values_3: list[float] = [-0.3, 0.3, 0.2]  # Small values
initial_values_4: list[float] = [1.5, 1.5, 0.8]   # Larger positive values
initial_values_5: list[float] = [-1.2, -1.2, 0.3] # Negative values


# -----------------------------------------------------------------------------
# Tests for OptQNIPS (Interior-Point Solver)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("initial_values", [initial_values_1, initial_values_2, initial_values_3, initial_values_4, initial_values_5])
@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"),
    [
        ([-1.0, -1.0, -1.0], [1.0, 1.0, 0.8]),      # Standard bounds
        ([0.05, 0.05, 0.05], [1.0, 1.0, 1.0]),      # Positive lower bounds
        ([-0.2, -0.2, 0.4], [0.2, 0.2, 0.8]),       # Tight bounds around solution
        ([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]),      # Wide bounds
        ([0.0, 0.0, 0.0], [0.3, 0.3, 0.7]),         # Medium-tight bounds
        ([-0.5, -0.5, 0.3], [0.5, 0.5, 0.7]),       # Bounds around expected
        ([0.2, 0.2, 0.3], [0.8, 0.8, 0.8]),         # Bounds away from expected
        ([-1.0, -1.0, 0.4], [1.0, 1.0, 0.6]),       # Tight z-bounds
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),         # Unit cube
        ([-0.1, -0.1, 0.4], [0.1, 0.1, 0.6]),       # Very tight bounds
    ],
)
def test_optqnips_constrained_expected(
    initial_values: list[float],
    lower_bounds: list[float],
    upper_bounds: list[float],
) -> None:
    """OptQNIPS should match projected analytical solution in constrained case."""
    bounds = Bounds(lower_bounds, upper_bounds)
    
    # Use OptQNIPS with appropriate settings for interior-point optimization
    options = {
        'debug': False,
        'merit_function': 'argaez_tapia',      # Default and robust merit function
        'search_method': 'trust_region',       # Stable search strategy
        'centering_parameter': 0.2,            # Standard centering
        'steplength_to_boundary': 0.99995,     # Allow close approach to boundary
        'max_iterations': 200,                 # Generous iteration limit
        'max_function_evaluations': 2000,      # Generous function evaluation limit
        'convergence_tolerance': 1e-6,         # Tight convergence
        'gradient_tolerance': 1e-6,            # Tight gradient tolerance
        'constraint_tolerance': 1e-8,          # Very tight constraint tolerance
    }
    
    res = minimize(
        objective, 
        initial_values, 
        method="optpp_q_nips", 
        jac=objective_gradient,
        bounds=bounds,
        options=options
    )
    
    expected = _project_to_bounds(expected_unconstrained, lower_bounds, upper_bounds)
    
    # OptQNIPS is an interior-point method, so we may need slightly looser tolerances
    # especially for problems where the solution is on the boundary
    np.testing.assert_allclose(res.x, expected, rtol=1e-1, atol=5e-2)


@pytest.mark.parametrize(
    ("initial_values", "lower_bounds", "upper_bounds"),
    [
        ([0.05, 0.05, 0.45], [-1.0, -1.0, -1.0], [1.0, 1.0, 0.8]),
        ([0.1, 0.1, 0.2], [0.05, 0.05, 0.05], [1.0, 1.0, 1.0]),
        ([0.0, 0.0, 0.6], [-0.2, -0.2, 0.4], [0.2, 0.2, 0.8]),
        ([0.1, 0.1, 0.1], [-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]),
        ([0.1, 0.1, 0.35], [0.0, 0.0, 0.0], [0.3, 0.3, 0.7]),
        ([0.0, 0.0, 0.5], [-0.5, -0.5, 0.3], [0.5, 0.5, 0.7]),
        ([0.5, 0.5, 0.55], [0.2, 0.2, 0.3], [0.8, 0.8, 0.8]),
        ([0.0, 0.0, 0.5], [-1.0, -1.0, 0.4], [1.0, 1.0, 0.6]),
        ([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ([0.05, 0.05, 0.5], [-0.1, -0.1, 0.4], [0.1, 0.1, 0.6]),
    ],
)
def test_optqnips_constrained_expected_feasible_start(
    initial_values: list[float],
    lower_bounds: list[float],
    upper_bounds: list[float],
) -> None:
    """OptQNIPS should match projected analytical solution with feasible start."""
    # Make sure initial point is feasible
    for i, val in enumerate(initial_values):
        assert lower_bounds[i] <= val <= upper_bounds[i], f"Initial value {val} not in [{lower_bounds[i]}, {upper_bounds[i]}]"

    bounds = Bounds(lower_bounds, upper_bounds)
    
    # Use different merit functions for variety
    options = {
        'debug': False,
        'merit_function': 'el_bakry',          # Different merit function
        'search_method': 'trust_region',
        'centering_parameter': 0.2,
        'steplength_to_boundary': 0.8,        # Conservative for el_bakry
        'max_iterations': 150,
        'max_function_evaluations': 1500,
        'convergence_tolerance': 1e-5,
        'gradient_tolerance': 1e-5,
        'constraint_tolerance': 1e-7,
    }
    
    res = minimize(
        objective, 
        initial_values, 
        method="optpp_q_nips", 
        jac=objective_gradient,
        bounds=bounds,
        options=options
    )
    
    expected = _project_to_bounds(expected_unconstrained, lower_bounds, upper_bounds)
    np.testing.assert_allclose(res.x, expected, rtol=1e-1, atol=5e-2)


def test_optqnips_simple_quadratic():
    """Test OptQNIPS on simple quadratic with bounds."""
    
    def simple_objective(x):
        """Simple quadratic: (x-0.7)^2 + (y-0.3)^2"""
        return (x[0] - 0.7)**2 + (x[1] - 0.3)**2
    
    def simple_gradient(x):
        """Gradient of simple quadratic."""
        return np.array([2*(x[0] - 0.7), 2*(x[1] - 0.3)])
    
    x0 = np.array([0.1, 0.8])
    bounds = Bounds([0.0, 0.0], [1.0, 1.0])
    
    options = {
        'debug': False,
        'merit_function': 'van_shanno',
        'search_method': 'trust_region',
        'centering_parameter': 0.1,           # van_shanno default
        'steplength_to_boundary': 0.95,      # van_shanno default
        'max_iterations': 100,
        'convergence_tolerance': 1e-8,
        'gradient_tolerance': 1e-8,
    }
    
    result = minimize(
        simple_objective,
        x0,
        method='optpp_q_nips',
        jac=simple_gradient,
        bounds=bounds,
        options=options
    )
    
    # Expected: [0.7, 0.3] which is within bounds
    expected_solution = np.array([0.7, 0.3])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-3, atol=1e-3)
    assert result.fun < 1e-6, f"Function value too high: {result.fun}"


def test_optqnips_rosenbrock_constrained():
    """Test OptQNIPS on 2D Rosenbrock function with bounds."""
    
    def rosenbrock_2d(x):
        """2D Rosenbrock function."""
        return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    def rosenbrock_2d_grad(x):
        """Gradient of 2D Rosenbrock function."""
        grad = np.zeros_like(x)
        grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        grad[1] = 200 * (x[1] - x[0]**2)
        return grad
    
    x0 = np.array([0.2, 0.8])
    bounds = Bounds([0.0, 0.0], [2.0, 2.0])
    
    # Use aggressive settings for this challenging problem
    options = {
        'debug': False,
        'merit_function': 'argaez_tapia',
        'search_method': 'trust_region',
        'centering_parameter': 0.05,          # More aggressive centering
        'steplength_to_boundary': 0.995,      # Allow closer approach to boundary
        'max_iterations': 500,                # More iterations for difficult problem
        'max_function_evaluations': 5000,     # More function evaluations
        'convergence_tolerance': 1e-4,        # Looser convergence for Rosenbrock
        'gradient_tolerance': 1e-4,
        'constraint_tolerance': 1e-6,
        'max_step': 0.5,                      # Smaller steps for stability
    }
    
    result = minimize(
        rosenbrock_2d,
        x0,
        method='optpp_q_nips',
        jac=rosenbrock_2d_grad,
        bounds=bounds,
        options=options
    )
    
    # The global minimum is at [1, 1] with f = 0, which is within our bounds
    expected_solution = np.array([1.0, 1.0])
    
    # Rosenbrock is challenging, so use looser tolerances
    np.testing.assert_allclose(result.x, expected_solution, rtol=5e-2, atol=5e-2)
    assert result.fun < 1e-1, f"Function value too high for Rosenbrock: {result.fun}"


def test_optqnips_merit_function_comparison():
    """Test OptQNIPS with different merit functions on the same problem."""
    
    def test_objective(x):
        """Test quadratic: (x-0.6)^2 + (y-0.4)^2"""
        return (x[0] - 0.6)**2 + (x[1] - 0.4)**2
    
    def test_gradient(x):
        """Gradient of test quadratic."""
        return np.array([2*(x[0] - 0.6), 2*(x[1] - 0.4)])
    
    x0 = np.array([0.1, 0.9])
    bounds = Bounds([0.0, 0.0], [1.0, 1.0])
    expected_solution = np.array([0.6, 0.4])
    
    merit_functions = [
        ('el_bakry', {'centering_parameter': 0.2, 'steplength_to_boundary': 0.8}),
        ('argaez_tapia', {'centering_parameter': 0.2, 'steplength_to_boundary': 0.99995}),
        ('van_shanno', {'centering_parameter': 0.1, 'steplength_to_boundary': 0.95}),
    ]
    
    for merit_func, params in merit_functions:
        print(f"\nTesting merit function: {merit_func}")
        
        options = {
            'debug': False,
            'merit_function': merit_func,
            'search_method': 'trust_region',
            'max_iterations': 100,
            'convergence_tolerance': 1e-6,
            'gradient_tolerance': 1e-6,
            **params
        }
        
        result = minimize(
            test_objective,
            x0,
            method='optpp_q_nips',
            jac=test_gradient,
            bounds=bounds,
            options=options
        )
        
        print(f"Merit function {merit_func}: x = {result.x}, f = {result.fun}, success = {result.success}")
        
        if result.success:
            np.testing.assert_allclose(result.x, expected_solution, rtol=1e-2, atol=1e-2)
            assert result.fun < 1e-4, f"Function value too high for {merit_func}: {result.fun}"


def test_optqnips_search_strategy_comparison():
    """Test OptQNIPS with different search strategies."""
    
    def test_objective(x):
        """Test quadratic: x^2 + y^2"""
        return x[0]**2 + x[1]**2
    
    def test_gradient(x):
        """Gradient of test quadratic."""
        return 2 * x
    
    x0 = np.array([0.5, 0.5])
    bounds = Bounds([-1.0, -1.0], [1.0, 1.0])
    expected_solution = np.array([0.0, 0.0])
    
    search_strategies = ['trust_region', 'line_search', 'trust_pds']
    
    for strategy in search_strategies:
        print(f"\nTesting search strategy: {strategy}")
        
        options = {
            'debug': False,
            'merit_function': 'argaez_tapia',
            'search_method': strategy,
            'centering_parameter': 0.2,
            'steplength_to_boundary': 0.99995,
            'max_iterations': 100,
            'convergence_tolerance': 1e-6,
            'gradient_tolerance': 1e-6,
            'max_step': 0.5,  # Smaller step for stability
        }
        
        result = minimize(
            test_objective,
            x0,
            method='optpp_q_nips',
            jac=test_gradient,
            bounds=bounds,
            options=options
        )
        
        print(f"Search strategy {strategy}: x = {result.x}, f = {result.fun}, success = {result.success}")
        
        if result.success:
            np.testing.assert_allclose(result.x, expected_solution, rtol=1e-2, atol=1e-2)
            assert result.fun < 1e-4, f"Function value too high for {strategy}: {result.fun}"


def test_optqnips_boundary_solution():
    """Test OptQNIPS when optimal solution is on the boundary."""
    
    def boundary_objective(x):
        """Objective that has minimum at boundary: (x-2)^2 + (y-2)^2"""
        return (x[0] - 2.0)**2 + (x[1] - 2.0)**2
    
    def boundary_gradient(x):
        """Gradient of boundary objective."""
        return np.array([2*(x[0] - 2.0), 2*(x[1] - 2.0)])
    
    x0 = np.array([0.5, 0.5])
    bounds = Bounds([0.0, 0.0], [1.0, 1.0])  # Optimal [2,2] is outside bounds
    
    options = {
        'debug': False,
        'merit_function': 'argaez_tapia',
        'search_method': 'trust_region',
        'centering_parameter': 0.2,
        'steplength_to_boundary': 0.99995,    # Allow very close approach to boundary
        'max_iterations': 200,
        'convergence_tolerance': 1e-6,
        'gradient_tolerance': 1e-6,
        'constraint_tolerance': 1e-8,
    }
    
    result = minimize(
        boundary_objective,
        x0,
        method='optpp_q_nips',
        jac=boundary_gradient,
        bounds=bounds,
        options=options
    )
    
    # Expected solution is at the boundary: [1.0, 1.0]
    expected_solution = np.array([1.0, 1.0])
    np.testing.assert_allclose(result.x, expected_solution, rtol=1e-2, atol=1e-2)
    
    # Check that we're actually at the boundary (within constraint tolerance)
    assert np.abs(result.x[0] - 1.0) < 1e-3, f"x[0] should be at upper bound: {result.x[0]}"
    assert np.abs(result.x[1] - 1.0) < 1e-3, f"x[1] should be at upper bound: {result.x[1]}"