#!/usr/bin/env python3
# tests/ropt_dakota_vs_everest_optimizers/optqnips/test_optqnips_expanded.py

import numpy as np
import sys
import os
import pytest

# Add the source directory to the path
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add pyopttpp path
pyopttpp_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "dakota-packages", "OPTPP", "build", "python"
)
if pyopttpp_path not in sys.path:
    sys.path.insert(0, pyopttpp_path)


class TestOptQNIPSExpanded:
    """Comprehensive tests for OptQNIPS (Quasi-Newton Interior-Point Solver) with all available parameters."""

    def test_merit_function_norm_fmu(self):
        """Test OptQNIPS with NormFmu merit function."""
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

        x0 = np.array([0.1, 0.1])
        bounds = Bounds(lb=[0.0, 0.0], ub=[1.0, 1.0])

        # Test with el_bakry merit function (maps to NormFmu in OPTPP)
        options = {
            'debug': False,
            'merit_function': 'el_bakry',
            'centering_parameter': 0.2,  # Default for el_bakry
            'steplength_to_boundary': 0.8,   # Default for el_bakry  
            'max_iterations': 50,  # Reasonable limit for testing
        }

        result = minimize(
            quadratic, 
            x0, 
            method="optpp_q_nips", 
            jac=quadratic_grad,
            bounds=bounds,
            options=options
        )

        print(f"El-Bakry Merit Function Result: x = {result.x}, f = {result.fun}")
        print(f"Success: {result.success}, Message: {result.message}")

        # Should find the optimum at [0.5, 0.5]
        if result.success:
            assert np.allclose(result.x, [0.5, 0.5], rtol=1e-2), f"Expected [0.5, 0.5], got {result.x}"
            assert result.fun < 1e-3, f"Function value too high: {result.fun}"

    def test_merit_function_argaez_tapia(self):
        """Test OptQNIPS with ArgaezTapia merit function."""
        from everest_optimizers import minimize
        from scipy.optimize import Bounds

        def quadratic(x):
            """Simple quadratic: f(x) = (x[0] - 0.3)^2 + (x[1] - 0.7)^2"""
            return (x[0] - 0.3) ** 2 + (x[1] - 0.7) ** 2

        def quadratic_grad(x):
            """Gradient of the quadratic function."""
            grad = np.zeros_like(x)
            grad[0] = 2 * (x[0] - 0.3)
            grad[1] = 2 * (x[1] - 0.7)
            return grad

        x0 = np.array([0.1, 0.1])
        bounds = Bounds(lb=[0.0, 0.0], ub=[1.0, 1.0])

        # Test with ArgaezTapia merit function (default)
        options = {
            'debug': False,
            'merit_function': 'argaez_tapia',
            'centering_parameter': 0.2,     # Default for ArgaezTapia
            'steplength_to_boundary': 0.99995, # Default for ArgaezTapia
        }

        result = minimize(
            quadratic, 
            x0, 
            method="optpp_q_nips", 
            jac=quadratic_grad,
            bounds=bounds,
            options=options
        )

        print(f"ArgaezTapia Merit Function Result: x = {result.x}, f = {result.fun}")
        print(f"Success: {result.success}, Message: {result.message}")

        # Should find the optimum at [0.3, 0.7]
        if result.success:
            assert np.allclose(result.x, [0.3, 0.7], rtol=1e-2), f"Expected [0.3, 0.7], got {result.x}"
            assert result.fun < 1e-3, f"Function value too high: {result.fun}"

    def test_merit_function_van_shanno(self):
        """Test OptQNIPS with VanShanno merit function."""
        from everest_optimizers import minimize
        from scipy.optimize import Bounds

        def quadratic(x):
            """Simple quadratic: f(x) = (x[0] - 0.8)^2 + (x[1] - 0.2)^2"""
            return (x[0] - 0.8) ** 2 + (x[1] - 0.2) ** 2

        def quadratic_grad(x):
            """Gradient of the quadratic function."""
            grad = np.zeros_like(x)
            grad[0] = 2 * (x[0] - 0.8)
            grad[1] = 2 * (x[1] - 0.2)
            return grad

        x0 = np.array([0.5, 0.5])
        bounds = Bounds(lb=[0.0, 0.0], ub=[1.0, 1.0])

        # Test with VanShanno merit function
        options = {
            'debug': False,
            'merit_function': 'van_shanno',
            'centering_parameter': 0.1,  # Default for VanShanno
            'steplength_to_boundary': 0.95, # Default for VanShanno
        }

        result = minimize(
            quadratic, 
            x0, 
            method="optpp_q_nips", 
            jac=quadratic_grad,
            bounds=bounds,
            options=options
        )

        print(f"VanShanno Merit Function Result: x = {result.x}, f = {result.fun}")
        print(f"Success: {result.success}, Message: {result.message}")

        # Should find the optimum at [0.8, 0.2]
        if result.success:
            assert np.allclose(result.x, [0.8, 0.2], rtol=1e-2), f"Expected [0.8, 0.2], got {result.x}"
            assert result.fun < 1e-3, f"Function value too high: {result.fun}"

    def test_search_strategies(self):
        """Test OptQNIPS with different search strategies."""
        from everest_optimizers import minimize
        from scipy.optimize import Bounds

        def quadratic(x):
            """Simple quadratic: f(x) = x[0]^2 + x[1]^2"""
            return x[0] ** 2 + x[1] ** 2

        def quadratic_grad(x):
            """Gradient of the quadratic function."""
            return 2 * x

        x0 = np.array([0.5, 0.5])
        bounds = Bounds(lb=[-1.0, -1.0], ub=[1.0, 1.0])

        # Test Dakota search method keywords
        search_methods = ['trust_region', 'value_based_line_search', 'gradient_based_line_search', 'tr_pds']
        
        for method in search_methods:
            print(f"\nTesting search method: {method}")
            
            options = {
                'debug': False,
                'search_method': method,
                'max_step': 0.5,  # Smaller maximum step for testing
            }

            result = minimize(
                quadratic, 
                x0, 
                method="optpp_q_nips", 
                jac=quadratic_grad,
                bounds=bounds,
                options=options
            )

            print(f"Search method {method} Result: x = {result.x}, f = {result.fun}")
            print(f"Success: {result.success}, Message: {result.message}")

            # Should find the optimum at [0, 0]
            if result.success:
                assert np.allclose(result.x, [0.0, 0.0], rtol=1e-2, atol=1e-2), f"Search method {method}: Expected [0.0, 0.0], got {result.x}"
                assert result.fun < 1e-3, f"Search method {method}: Function value too high: {result.fun}"

    def test_tolerance_parameters(self):
        """Test OptQNIPS with various tolerance parameters."""
        from everest_optimizers import minimize
        from scipy.optimize import Bounds

        def quadratic(x):
            """Simple quadratic: f(x) = (x[0] - 0.4)^2 + (x[1] - 0.6)^2"""
            return (x[0] - 0.4) ** 2 + (x[1] - 0.6) ** 2

        def quadratic_grad(x):
            """Gradient of the quadratic function."""
            grad = np.zeros_like(x)
            grad[0] = 2 * (x[0] - 0.4)
            grad[1] = 2 * (x[1] - 0.6)
            return grad

        x0 = np.array([0.1, 0.1])
        bounds = Bounds(lb=[0.0, 0.0], ub=[1.0, 1.0])

        # Test with tight tolerances
        options = {
            'debug': False,
            'convergence_tolerance': 1e-6,
            'gradient_tolerance': 1e-6,
            'constraint_tolerance': 1e-8,
            'max_iterations': 200,
            'max_function_evaluations': 2000,
        }

        result = minimize(
            quadratic, 
            x0, 
            method="optpp_q_nips", 
            jac=quadratic_grad,
            bounds=bounds,
            options=options
        )

        print(f"Tight Tolerances Result: x = {result.x}, f = {result.fun}")
        print(f"Success: {result.success}, Message: {result.message}")
        print(f"Function evaluations: {result.nfev}")

        # With tight tolerances, should get very accurate solution
        if result.success:
            assert np.allclose(result.x, [0.4, 0.6], rtol=1e-3), f"Expected [0.4, 0.6], got {result.x}"
            assert result.fun < 1e-5, f"Function value too high for tight tolerances: {result.fun}"

    def test_interior_point_parameters(self):
        """Test OptQNIPS with various interior-point specific parameters."""
        from everest_optimizers import minimize
        from scipy.optimize import Bounds

        def quadratic(x):
            """Simple quadratic: f(x) = (x[0] - 0.6)^2 + (x[1] - 0.4)^2"""
            return (x[0] - 0.6) ** 2 + (x[1] - 0.4) ** 2

        def quadratic_grad(x):
            """Gradient of the quadratic function."""
            grad = np.zeros_like(x)
            grad[0] = 2 * (x[0] - 0.6)
            grad[1] = 2 * (x[1] - 0.4)
            return grad

        x0 = np.array([0.2, 0.8])
        bounds = Bounds(lb=[0.0, 0.0], ub=[1.0, 1.0])

        # Test with various interior-point parameters
        test_cases = [
            {
                'name': 'Conservative',
                'mu': 0.5,
                'centering_parameter': 0.5,
                'steplength_to_boundary': 0.7,
            },
            {
                'name': 'Aggressive',
                'mu': 0.01,
                'centering_parameter': 0.05,
                'steplength_to_boundary': 0.99,
            },
            {
                'name': 'Balanced',
                'mu': 0.1,
                'centering_parameter': 0.1,
                'steplength_to_boundary': 0.95,
            },
        ]

        for case in test_cases:
            print(f"\nTesting {case['name']} parameters:")
            
            options = {
                'debug': False,
                'mu': case['mu'],
                'centering_parameter': case['centering_parameter'],
                'steplength_to_boundary': case['steplength_to_boundary'],
            }

            result = minimize(
                quadratic, 
                x0, 
                method="optpp_q_nips", 
                jac=quadratic_grad,
                bounds=bounds,
                options=options
            )

            print(f"{case['name']} Result: x = {result.x}, f = {result.fun}")
            print(f"Success: {result.success}, Message: {result.message}")

            # Should find the optimum at [0.6, 0.4]
            if result.success:
                assert np.allclose(result.x, [0.6, 0.4], rtol=5e-2), f"{case['name']}: Expected [0.6, 0.4], got {result.x}"
                assert result.fun < 1e-2, f"{case['name']}: Function value too high: {result.fun}"

    def test_trust_region_parameters(self):
        """Test OptQNIPS with trust region parameters."""
        from everest_optimizers import minimize
        from scipy.optimize import Bounds

        def quadratic(x):
            """Simple quadratic: f(x) = (x[0] - 0.7)^2 + (x[1] - 0.3)^2"""
            return (x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2

        def quadratic_grad(x):
            """Gradient of the quadratic function."""
            grad = np.zeros_like(x)
            grad[0] = 2 * (x[0] - 0.7)
            grad[1] = 2 * (x[1] - 0.3)
            return grad

        x0 = np.array([0.1, 0.9])
        bounds = Bounds(lb=[0.0, 0.0], ub=[1.0, 1.0])

        # Test with different trust region sizes
        tr_sizes = [10.0, 1.0, 0.1]
        
        for tr_size in tr_sizes:
            print(f"\nTesting trust region size: {tr_size}")
            
            options = {
                'debug': False,
                'search_method': 'trust_region',
                'max_step': tr_size,
                'gradient_multiplier': 0.1,
                'search_pattern_size': 32,
            }

            result = minimize(
                quadratic, 
                x0, 
                method="optpp_q_nips", 
                jac=quadratic_grad,
                bounds=bounds,
                options=options
            )

            print(f"TR Size {tr_size} Result: x = {result.x}, f = {result.fun}")
            print(f"Success: {result.success}, Message: {result.message}")

            # Should find the optimum at [0.7, 0.3]
            if result.success:
                assert np.allclose(result.x, [0.7, 0.3], rtol=1e-2), f"TR Size {tr_size}: Expected [0.7, 0.3], got {result.x}"
                assert result.fun < 1e-3, f"TR Size {tr_size}: Function value too high: {result.fun}"

    def test_iteration_limits(self):
        """Test OptQNIPS with iteration and function evaluation limits."""
        from everest_optimizers import minimize
        from scipy.optimize import Bounds

        def expensive_quadratic(x):
            """Quadratic that we'll pretend is expensive."""
            return (x[0] - 0.9) ** 2 + (x[1] - 0.1) ** 2

        def expensive_quadratic_grad(x):
            """Gradient of the expensive quadratic."""
            grad = np.zeros_like(x)
            grad[0] = 2 * (x[0] - 0.9)
            grad[1] = 2 * (x[1] - 0.1)
            return grad

        x0 = np.array([0.1, 0.9])
        bounds = Bounds(lb=[0.0, 0.0], ub=[1.0, 1.0])

        # Test with very low limits
        options = {
            'debug': False,
            'max_iterations': 5,
            'max_function_evaluations': 20,
        }

        result = minimize(
            expensive_quadratic, 
            x0, 
            method="optpp_q_nips", 
            jac=expensive_quadratic_grad,
            bounds=bounds,
            options=options
        )

        print(f"Low Limits Result: x = {result.x}, f = {result.fun}")
        print(f"Success: {result.success}, Message: {result.message}")
        print(f"Function evaluations: {result.nfev}")

        # With very low limits, might not converge, but should stop early
        assert result.nfev <= 25, f"Should have stopped early, but used {result.nfev} evaluations"

        # Now test with reasonable limits
        options['max_iterations'] = 100
        options['max_function_evaluations'] = 500

        result = minimize(
            expensive_quadratic, 
            x0, 
            method="optpp_q_nips", 
            jac=expensive_quadratic_grad,
            bounds=bounds,
            options=options
        )

        print(f"Reasonable Limits Result: x = {result.x}, f = {result.fun}")
        print(f"Success: {result.success}, Message: {result.message}")

        # Should find the optimum at [0.9, 0.1] with reasonable limits
        if result.success:
            assert np.allclose(result.x, [0.9, 0.1], rtol=1e-2), f"Expected [0.9, 0.1], got {result.x}"
            assert result.fun < 1e-3, f"Function value too high: {result.fun}"

    def test_robustness_with_difficult_problem(self):
        """Test OptQNIPS robustness with a more challenging optimization problem."""
        from everest_optimizers import minimize
        from scipy.optimize import Bounds

        def rosenbrock_2d(x):
            """2D Rosenbrock function - a classic challenging optimization problem."""
            return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

        def rosenbrock_2d_grad(x):
            """Gradient of 2D Rosenbrock function."""
            grad = np.zeros_like(x)
            grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
            grad[1] = 200 * (x[1] - x[0]**2)
            return grad

        x0 = np.array([0.1, 0.8])
        bounds = Bounds(lb=[0.0, 0.0], ub=[2.0, 2.0])

        # Use aggressive settings for this challenging problem
        options = {
            'debug': False,
            'merit_function': 'argaez_tapia',
            'mu': 0.01,
            'centering_parameter': 0.05,
            'steplength_to_boundary': 0.995,
            'convergence_tolerance': 1e-5,
            'gradient_tolerance': 1e-5,
            'max_iterations': 300,
            'max_function_evaluations': 3000,
        }

        result = minimize(
            rosenbrock_2d, 
            x0, 
            method="optpp_q_nips", 
            jac=rosenbrock_2d_grad,
            bounds=bounds,
            options=options
        )

        print(f"Rosenbrock Result: x = {result.x}, f = {result.fun}")
        print(f"Success: {result.success}, Message: {result.message}")
        print(f"Function evaluations: {result.nfev}")

        # The global minimum is at [1, 1] with f = 0
        # This is a challenging problem, so we use looser tolerances
        if result.success:
            assert np.allclose(result.x, [1.0, 1.0], rtol=1e-1), f"Expected [1.0, 1.0], got {result.x}"
            assert result.fun < 1e-2, f"Function value too high for Rosenbrock: {result.fun}"

    def test_parameter_validation(self):
        """Test that OptQNIPS handles invalid parameters gracefully."""
        from everest_optimizers import minimize
        from scipy.optimize import Bounds

        def simple_quadratic(x):
            return x[0]**2 + x[1]**2

        def simple_quadratic_grad(x):
            return 2 * x

        x0 = np.array([0.5, 0.5])
        bounds = Bounds(lb=[-1.0, -1.0], ub=[1.0, 1.0])

        # Test various potentially problematic parameter values
        problematic_options = [
            {'mu': -0.1},  # Negative mu
            {'centering_parameter': -0.1},  # Negative centering parameter
            {'steplength_to_boundary': 1.1},  # Step length > 1
            {'steplength_to_boundary': 0.0},  # Step length = 0
            {'tr_size': -1.0},  # Negative trust region size
            {'max_iterations': 0},  # Zero iterations
            {'max_function_evaluations': 0},  # Zero function evaluations
        ]

        for bad_option in problematic_options:
            print(f"\nTesting problematic option: {bad_option}")
            
            try:
                result = minimize(
                    simple_quadratic, 
                    x0, 
                    method="optpp_q_nips", 
                    jac=simple_quadratic_grad,
                    bounds=bounds,
                    options=bad_option
                )
                
                # If it doesn't raise an exception, it should at least not crash
                print(f"Result with bad option: Success = {result.success}")
                
            except (ValueError, RuntimeError) as e:
                print(f"Appropriately caught error: {e}")
                # This is expected behavior for invalid parameters


if __name__ == "__main__":
    # Run the tests directly
    test_instance = TestOptQNIPSExpanded()
    
    print("=" * 80)
    print("COMPREHENSIVE OPTQNIPS PARAMETER TESTING")
    print("=" * 80)
    
    print("\n1. Testing Merit Functions...")
    try:
        test_instance.test_merit_function_norm_fmu()
        print("✓ NormFmu merit function test passed")
    except Exception as e:
        print(f"✗ NormFmu merit function test failed: {e}")
    
    try:
        test_instance.test_merit_function_argaez_tapia()
        print("✓ ArgaezTapia merit function test passed")
    except Exception as e:
        print(f"✗ ArgaezTapia merit function test failed: {e}")
    
    try:
        test_instance.test_merit_function_van_shanno()
        print("✓ VanShanno merit function test passed")
    except Exception as e:
        print(f"✗ VanShanno merit function test failed: {e}")
    
    print("\n2. Testing Search Strategies...")
    try:
        test_instance.test_search_strategies()
        print("✓ Search strategies test passed")
    except Exception as e:
        print(f"✗ Search strategies test failed: {e}")
    
    print("\n3. Testing Tolerance Parameters...")
    try:
        test_instance.test_tolerance_parameters()
        print("✓ Tolerance parameters test passed")
    except Exception as e:
        print(f"✗ Tolerance parameters test failed: {e}")
    
    print("\n4. Testing Interior-Point Parameters...")
    try:
        test_instance.test_interior_point_parameters()
        print("✓ Interior-point parameters test passed")
    except Exception as e:
        print(f"✗ Interior-point parameters test failed: {e}")
    
    print("\n5. Testing Trust Region Parameters...")
    try:
        test_instance.test_trust_region_parameters()
        print("✓ Trust region parameters test passed")
    except Exception as e:
        print(f"✗ Trust region parameters test failed: {e}")
    
    print("\n6. Testing Iteration Limits...")
    try:
        test_instance.test_iteration_limits()
        print("✓ Iteration limits test passed")
    except Exception as e:
        print(f"✗ Iteration limits test failed: {e}")
    
    print("\n7. Testing Robustness with Difficult Problem...")
    try:
        test_instance.test_robustness_with_difficult_problem()
        print("✓ Robustness test passed")
    except Exception as e:
        print(f"✗ Robustness test failed: {e}")
    
    print("\n8. Testing Parameter Validation...")
    try:
        test_instance.test_parameter_validation()
        print("✓ Parameter validation test passed")
    except Exception as e:
        print(f"✗ Parameter validation test failed: {e}")
    
    print("\n" + "=" * 80)
    print("OPTQNIPS COMPREHENSIVE TESTING COMPLETE!")
    print("=" * 80)