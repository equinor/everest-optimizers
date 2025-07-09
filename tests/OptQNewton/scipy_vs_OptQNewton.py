#!/usr/bin/env python3
# tests/OptQNewton/scipy_vs_OptQNewton.py
"""
Compare the OptQNewton optimizer from pyopttpp with SciPy's BFGS optimizer.
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize
import pytest

# Add the build directory to the Python path
# This allows importing the pyopttpp module from its build location
pyopttpp_build_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 
    'dakota-packages', 'OPTPP', 'build', 'python'
))
sys.path.insert(0, pyopttpp_build_dir)

# Try to import pyopttpp but don't fail if not available
pyopttpp_available = False
try:
    import pyopttpp
    pyopttpp_available = True
except ImportError:
    pass

# --- Rosenbrock Function Definition for SciPy ---
def rosenbrock_f(x):
    """Rosenbrock function."""
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def rosenbrock_g(x):
    """Gradient of the Rosenbrock function."""
    grad = np.zeros_like(x)
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    grad[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
    grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    grad[-1] = 200 * (x[-1] - x[-2]**2)
    return grad

# --- Rosenbrock Class for pyopttpp ---
if pyopttpp_available:
    class Rosenbrock(pyopttpp.NLF1):
        """Rosenbrock function for optimization testing with pyopttpp."""
        def __init__(self, ndim, x_init_np):
            super().__init__(ndim)
            init_vector = pyopttpp.SerialDenseVector(x_init_np)
            self.setX(init_vector)

        def evalF(self, x):
            """Evaluates the Rosenbrock function."""
            x_np = np.array(x.to_numpy(), copy=True)
            return rosenbrock_f(x_np)

        def evalG(self, x):
            """Evaluates the gradient of the Rosenbrock function."""
            x_np = np.array(x.to_numpy(), copy=True)
            return rosenbrock_g(x_np)


@pytest.mark.skipif(not pyopttpp_available, reason="pyopttpp module not available")
def test_rosenbrock_comparison():
    """Compare OptQNewton and SciPy BFGS on the Rosenbrock function."""
    ndim = 2
    x_init = np.array([-1.2, 1.0])

    # --- Run OptQNewton ---
    rosen_problem = Rosenbrock(ndim, x_init)
    rosen_problem.setIsExpensive(True)
    optimizer = pyopttpp.OptQNewton(rosen_problem)
    optimizer.setTRSize(100.0)
    optimizer.optimize()
    optimizer.cleanup()
    optqnewton_sol = rosen_problem.getXc().to_numpy()
    optqnewton_fval = rosen_problem.getF()

    # --- Run SciPy BFGS ---
    scipy_res = minimize(rosenbrock_f, x_init, method='BFGS', jac=rosenbrock_g)
    scipy_sol = scipy_res.x
    scipy_fval = scipy_res.fun

    # --- Compare Results ---
    print("\n--- Rosenbrock Function Comparison ---")
    print(f"Initial point: {x_init}")
    print(f"OptQNewton solution: {optqnewton_sol}, f(x) = {optqnewton_fval}")
    print(f"SciPy BFGS solution:   {scipy_sol}, f(x) = {scipy_fval}")

    assert np.allclose(optqnewton_sol, scipy_sol, rtol=1e-4), \
        f"Solutions do not match! OptQNewton: {optqnewton_sol}, SciPy: {scipy_sol}"
    assert np.allclose(optqnewton_fval, scipy_fval, rtol=1e-6), \
        f"Function values do not match! OptQNewton: {optqnewton_fval}, SciPy: {scipy_fval}"
    print("Results are consistent.")

# --- Himmelblau's Function Definition ---
def himmelblau_f(x):
    """Himmelblau's function."""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def himmelblau_g(x):
    """Gradient of Himmelblau's function."""
    grad = np.zeros_like(x)
    grad[0] = 4 * x[0] * (x[0]**2 + x[1] - 11) + 2 * (x[0] + x[1]**2 - 7)
    grad[1] = 2 * (x[0]**2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1]**2 - 7)
    return grad

# --- Himmelblau Class for pyopttpp ---
if pyopttpp_available:
    class Himmelblau(pyopttpp.NLF1):
        """Himmelblau's function for optimization testing with pyopttpp."""
        def __init__(self, ndim, x_init_np):
            super().__init__(ndim)
            init_vector = pyopttpp.SerialDenseVector(x_init_np)
            self.setX(init_vector)

        def evalF(self, x):
            """Evaluates Himmelblau's function."""
            x_np = np.array(x.to_numpy(), copy=True)
            return himmelblau_f(x_np)

        def evalG(self, x):
            """Evaluates the gradient of Himmelblau's function."""
            x_np = np.array(x.to_numpy(), copy=True)
            return himmelblau_g(x_np)

@pytest.mark.skipif(not pyopttpp_available, reason="pyopttpp module not available")
@pytest.mark.parametrize("x_init", [
    np.array([0.0, 0.0]),
    np.array([-1.0, 1.0]),
    np.array([-2.0, -2.0]),
    np.array([3.0, 3.0]),
])
def test_himmelblau_comparison(x_init):
    """Compare OptQNewton and SciPy BFGS on Himmelblau's function."""
    ndim = 2

    # --- Run OptQNewton ---
    problem = Himmelblau(ndim, x_init)
    optimizer = pyopttpp.OptQNewton(problem)
    optimizer.optimize()
    optimizer.cleanup()
    optqnewton_sol = problem.getXc().to_numpy()
    optqnewton_fval = problem.getF()

    # --- Run SciPy BFGS ---
    scipy_res = minimize(himmelblau_f, x_init, method='BFGS', jac=himmelblau_g)
    scipy_sol = scipy_res.x
    scipy_fval = scipy_res.fun

    # --- Compare Results ---
    print(f"\n--- Himmelblau's Function Comparison (start: {x_init}) ---")
    print(f"OptQNewton solution: {optqnewton_sol}, f(x) = {optqnewton_fval}")
    print(f"SciPy BFGS solution:   {scipy_sol}, f(x) = {scipy_fval}")

    assert np.allclose(optqnewton_sol, scipy_sol, rtol=1e-4, atol=1e-5), \
        f"Solutions do not match! OptQNewton: {optqnewton_sol}, SciPy: {scipy_sol}"
    assert np.allclose(optqnewton_fval, scipy_fval, rtol=1e-6, atol=1e-7), \
        f"Function values do not match! OptQNewton: {optqnewton_fval}, SciPy: {scipy_fval}"
    print("Results are consistent.")

if __name__ == "__main__":
    # This allows running the script directly for debugging or demonstration
    if not pyopttpp_available:
        print("pyopttpp module not found. Skipping comparison.")
        sys.exit(0)
    
    # To run this specific test function, we can call pytest on this file
    # or call the functions directly.
    try:
        print("--- Running Rosenbrock Comparison ---")
        test_rosenbrock_comparison()
        print("\n--- Running Himmelblau Comparison ---")
        # Test with a few starting points
        initial_points = [
            np.array([0.0, 0.0]),
            np.array([-1.0, 1.0]),
            np.array([-2.0, -2.0]),
            np.array([3.0, 3.0]),
        ]
        for start_point in initial_points:
            test_himmelblau_comparison(start_point)

    except Exception as e:
        print(f"An error occurred: {e}")

