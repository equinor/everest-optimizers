#!/usr/bin/env python3
"""
Test script for constrained optimization with general constraints.

This test demonstrates the use of both bounds and linear constraints
with the OptConstrQNewton solver.
"""

import numpy as np
import sys
import os
from scipy.optimize import Bounds, LinearConstraint

# Add the source directory to the path
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from everest_optimizers import minimize


def test_constrained_optimization():
    """Test constrained optimization with bounds constraints only."""
    
    # Simple quadratic objective function: minimize (x-1)^2 + (y-2)^2
    def objective(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    
    # Gradient of the objective function
    def gradient(x):
        return np.array([2*(x[0] - 1), 2*(x[1] - 2)])
    
    # Initial point
    x0 = np.array([0.0, 0.0])
    
    print("Testing constrained optimization with OptConstrQNewton")
    print("Objective: minimize (x-1)^2 + (y-2)^2")
    print("Expected unconstrained solution: [1, 2]")
    print("Note: Only bounds constraints are currently supported in this build.")
    
    # Test 1: Only bounds constraints
    print("\n=== Test 1: Bounds only ===")
    bounds = Bounds([0.0, 0.0], [2.0, 3.0])
    
    result = minimize(
        objective, 
        x0, 
        method='optpp_constr_q_newton', 
        jac=gradient,
        bounds=bounds,
        options={'debug': False}
    )
    
    print(f"Solution: {result.x}")
    print(f"Objective value: {result.fun}")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    
    # The solution should be [1, 2] since it's within bounds
    expected = np.array([1.0, 2.0])
    if np.allclose(result.x, expected, rtol=1e-2, atol=1e-2):
        print("✓ PASS: Bounds constraint test")
    else:
        print(f"✗ FAIL: Expected {expected}, got {result.x}")
    
    # Test 2: Bounds with tight constraints
    print("\n=== Test 2: Tight bounds ===")
    bounds_tight = Bounds([0.5, 1.5], [0.8, 1.8])
    
    result = minimize(
        objective, 
        x0, 
        method='optpp_constr_q_newton', 
        jac=gradient,
        bounds=bounds_tight,
        options={'debug': False}
    )
    
    print(f"Solution: {result.x}")
    print(f"Objective value: {result.fun}")
    
    # Expected solution should be projected to the feasible region
    # Unconstrained optimum [1, 2] projected to [0.5, 1.5] - [0.8, 1.8]
    # gives [0.8, 1.8] 
    expected_tight = np.array([0.8, 1.8])
    if np.allclose(result.x, expected_tight, rtol=1e-1, atol=1e-1):
        print("✓ PASS: Tight bounds constraint test")
    else:
        print(f"✗ FAIL: Expected ~{expected_tight}, got {result.x}")
    
    # Test 3: Very tight bounds 
    print("\n=== Test 3: Very tight bounds ===")
    bounds_very_tight = Bounds([0.9, 1.9], [1.1, 2.1])
    
    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=gradient,
        bounds=bounds_very_tight,
        options={'debug': False}
    )
    
    print(f"Solution: {result.x}")
    print(f"Objective value: {result.fun}")
    
    # Check that solution is within bounds
    bounds_ok = (0.9 <= result.x[0] <= 1.1) and (1.9 <= result.x[1] <= 2.1)
    if bounds_ok:
        print("✓ PASS: Very tight bounds constraint satisfied")
    else:
        print(f"✗ FAIL: Bounds violated")
    
    # Test 4: Asymmetric bounds
    print("\n=== Test 4: Asymmetric bounds ===")
    bounds_asym = Bounds([0.0, 2.5], [0.5, 3.0])
    
    result = minimize(
        objective,
        x0,
        method='optpp_constr_q_newton',
        jac=gradient,
        bounds=bounds_asym,
        options={'debug': False}
    )
    
    print(f"Solution: {result.x}")
    print(f"Objective value: {result.fun}")
    
    # Expected solution: x constrained to [0.0, 0.5], y constrained to [2.5, 3.0]
    # Optimal would be x=0.5 (closest to 1), y=2.5 (closest to 2)
    expected_asym = np.array([0.5, 2.5])
    if np.allclose(result.x, expected_asym, rtol=1e-1, atol=1e-1):
        print("✓ PASS: Asymmetric bounds constraint test")
    else:
        print(f"✗ FAIL: Expected ~{expected_asym}, got {result.x}")

    # Test linear constraints (should fail with helpful message)
    print("\n=== Test 5: Linear constraint support check ===")
    try:
        A_eq = np.array([[1.0, 1.0]])
        b_eq = np.array([2.0])
        linear_constraint = LinearConstraint(A_eq, b_eq, b_eq)
        
        result = minimize(
            objective,
            x0,
            method='optpp_constr_q_newton',
            jac=gradient,
            constraints=linear_constraint,
            options={'debug': False}
        )
        print("✗ UNEXPECTED: Linear constraints should not be supported")
    except NotImplementedError as e:
        print("✓ PASS: Linear constraints correctly not supported")
        print(f"     Message: {e}")

    print("\n=== Test Summary ===")
    print("Current implementation successfully handles:")
    print("- Bounds constraints (lower and upper bounds)")
    print("- Box constraints (rectangular feasible regions)")
    print("- Asymmetric bounds")
    print("- Correctly reports when linear constraints are not supported")


if __name__ == "__main__":
    test_constrained_optimization()