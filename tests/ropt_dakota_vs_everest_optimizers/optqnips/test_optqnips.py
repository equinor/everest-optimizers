#!/usr/bin/env python3
# tests/test_optqnips.py

import numpy as np
import sys
import os
import pytest

# Add the source directory to the path
src_path = os.path.join(os.path.dirname(__file__), "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add pyopttpp path
pyopttpp_path = os.path.join(
    os.path.dirname(__file__), "..", "dakota-packages", "OPTPP", "build", "python"
)
if pyopttpp_path not in sys.path:
    sys.path.insert(0, pyopttpp_path)


class TestOptQNIPS:
    """Test the OptQNIPS (Quasi-Newton Interior-Point Solver) optimizer."""

    def test_constrained_quadratic_with_bounds(self):
        """Test OptQNIPS with a simple constrained quadratic function with bounds."""
        from everest_optimizers import minimize
        from scipy.optimize import Bounds

        def quadratic(x):
            """Simple quadratic: f(x) = (x[0] - 0.5)^2 + (x[1] - 0.5)^2"""
            return (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2

        def quadratic_grad(x):
            """Gradient of the quadratic function."""
            grad = np.zeros_like(x)
            grad[0] = 2 * (x[0] - 0.5)
            grad[1] = 2 * (x[1] - 0.5)
            return grad

        # Initial point
        x0 = np.array([0.0, 0.0])
        
        # Define bounds: 0 <= x <= 1
        bounds = Bounds(lb=[0.0, 0.0], ub=[1.0, 1.0])

        # Optimize using OptQNIPS
        result = minimize(
            quadratic, 
            x0, 
            method="optpp_q_nips", 
            jac=quadratic_grad,
            bounds=bounds,
            options={'debug': False}
        )

        print(f"OptQNIPS Result: x = {result.x}, f = {result.fun}")
        print(f"Success: {result.success}, Message: {result.message}")

        # The unconstrained optimum is at [0.5, 0.5], which is within bounds
        # So OptQNIPS should find this solution
        assert result.success, f"Optimization failed: {result.message}"
        assert np.allclose(result.x, [0.5, 0.5], rtol=1e-2), f"Expected [0.5, 0.5], got {result.x}"
        assert result.fun < 1e-4, f"Function value too high: {result.fun}"
        assert result.nfev > 0
        assert result.njev > 0

    def test_constrained_quadratic_active_bounds(self):
        """Test OptQNIPS where the optimum is on the boundary."""
        from everest_optimizers import minimize
        from scipy.optimize import Bounds

        def quadratic(x):
            """Quadratic with minimum outside the feasible region: f(x) = (x[0] - 2)^2 + (x[1] - 2)^2"""
            return (x[0] - 2.0) ** 2 + (x[1] - 2.0) ** 2

        def quadratic_grad(x):
            """Gradient of the quadratic function."""
            grad = np.zeros_like(x)
            grad[0] = 2 * (x[0] - 2.0)
            grad[1] = 2 * (x[1] - 2.0)
            return grad

        # Initial point
        x0 = np.array([0.5, 0.5])
        
        # Define bounds: 0 <= x <= 1 (minimum is at [2, 2] which is outside)
        bounds = Bounds(lb=[0.0, 0.0], ub=[1.0, 1.0])

        # Optimize using OptQNIPS
        result = minimize(
            quadratic, 
            x0, 
            method="optpp_q_nips", 
            jac=quadratic_grad,
            bounds=bounds,
            options={'debug': False}
        )

        print(f"OptQNIPS Active Bounds Result: x = {result.x}, f = {result.fun}")
        print(f"Success: {result.success}, Message: {result.message}")

        # The constrained optimum should be at [1, 1] (on the boundary)
        assert result.success, f"Optimization failed: {result.message}"
        assert np.allclose(result.x, [1.0, 1.0], rtol=1e-2), f"Expected [1.0, 1.0], got {result.x}"
        assert result.nfev > 0
        assert result.njev > 0

    def test_optqnips_parameters(self):
        """Test OptQNIPS with custom interior-point parameters."""
        from everest_optimizers import minimize
        from scipy.optimize import Bounds

        def simple_quadratic(x):
            """Simple quadratic: f(x) = x[0]^2 + x[1]^2"""
            return x[0] ** 2 + x[1] ** 2

        def simple_quadratic_grad(x):
            """Gradient of the simple quadratic."""
            return 2 * x

        # Initial point
        x0 = np.array([0.8, 0.8])
        
        # Define bounds: -1 <= x <= 1
        bounds = Bounds(lb=[-1.0, -1.0], ub=[1.0, 1.0])

        # Custom OptQNIPS parameters
        options = {
            'debug': False,
            'mu': 0.01,  # perturbation parameter
            'centering_parameter': 0.05,  # centering parameter (sigma)
            'step_length_to_bdry': 0.99,  # percentage to boundary (tau)
        }

        # Optimize using OptQNIPS with custom parameters
        result = minimize(
            simple_quadratic, 
            x0, 
            method="optpp_q_nips", 
            jac=simple_quadratic_grad,
            bounds=bounds,
            options=options
        )

        print(f"OptQNIPS Custom Params Result: x = {result.x}, f = {result.fun}")
        print(f"Success: {result.success}, Message: {result.message}")

        # The optimum should be at [0, 0]
        assert result.success, f"Optimization failed: {result.message}"
        assert np.allclose(result.x, [0.0, 0.0], rtol=1e-2), f"Expected [0.0, 0.0], got {result.x}"
        assert result.fun < 1e-4, f"Function value too high: {result.fun}"

    def test_optqnips_vs_optconstrqnewton(self):
        """Compare OptQNIPS with OptConstrQNewton on a simple problem."""
        from everest_optimizers import minimize
        from scipy.optimize import Bounds

        def quadratic(x):
            """Simple quadratic: f(x) = (x[0] - 0.3)^2 + (x[1] - 0.3)^2"""
            return (x[0] - 0.3) ** 2 + (x[1] - 0.3) ** 2

        def quadratic_grad(x):
            """Gradient of the quadratic function."""
            grad = np.zeros_like(x)
            grad[0] = 2 * (x[0] - 0.3)
            grad[1] = 2 * (x[1] - 0.3)
            return grad

        # Initial point
        x0 = np.array([0.0, 0.0])
        
        # Define bounds: 0 <= x <= 1
        bounds = Bounds(lb=[0.0, 0.0], ub=[1.0, 1.0])

        # Optimize using OptQNIPS
        result_nips = minimize(
            quadratic, 
            x0, 
            method="optpp_q_nips", 
            jac=quadratic_grad,
            bounds=bounds,
            options={'debug': False}
        )

        # Optimize using OptConstrQNewton for comparison
        result_constr = minimize(
            quadratic, 
            x0, 
            method="optpp_constr_q_newton", 
            jac=quadratic_grad,
            bounds=bounds,
            options={'debug': False}
        )

        print(f"OptQNIPS Result: x = {result_nips.x}, f = {result_nips.fun}")
        print(f"OptConstrQNewton Result: x = {result_constr.x}, f = {result_constr.fun}")

        # Both should succeed and find similar solutions
        assert result_nips.success, f"OptQNIPS failed: {result_nips.message}"
        assert result_constr.success, f"OptConstrQNewton failed: {result_constr.message}"
        
        # Both should find the true optimum at [0.3, 0.3]
        assert np.allclose(result_nips.x, [0.3, 0.3], rtol=1e-2), f"OptQNIPS: Expected [0.3, 0.3], got {result_nips.x}"
        assert np.allclose(result_constr.x, [0.3, 0.3], rtol=1e-2), f"OptConstrQNewton: Expected [0.3, 0.3], got {result_constr.x}"
        
        # Function values should be similar and small
        assert result_nips.fun < 1e-4, f"OptQNIPS function value too high: {result_nips.fun}"
        assert result_constr.fun < 1e-4, f"OptConstrQNewton function value too high: {result_constr.fun}"


if __name__ == "__main__":
    # Run the tests directly
    test_instance = TestOptQNIPS()
    
    print("Testing OptQNIPS with bounds...")
    test_instance.test_constrained_quadratic_with_bounds()
    print("✓ Passed: constrained quadratic with bounds")
    
    print("\nTesting OptQNIPS with active bounds...")
    test_instance.test_constrained_quadratic_active_bounds()
    print("✓ Passed: constrained quadratic with active bounds")
    
    print("\nTesting OptQNIPS with custom parameters...")
    test_instance.test_optqnips_parameters()
    print("✓ Passed: OptQNIPS with custom parameters")
    
    print("\nTesting OptQNIPS vs OptConstrQNewton...")
    test_instance.test_optqnips_vs_optconstrqnewton()
    print("✓ Passed: OptQNIPS vs OptConstrQNewton comparison")
    
    print("\nAll OptQNIPS tests passed! ✨")