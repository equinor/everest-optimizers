#!/usr/bin/env python3
# tests/OptQNewton/test_everest_optimizers_optimization.py

import pytest
import numpy as np
import sys
import os

# Add the source directory to the path
src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add pyopttpp path
pyopttpp_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dakota-packages', 'OPTPP', 'build', 'python')
if pyopttpp_path not in sys.path:
    sys.path.insert(0, pyopttpp_path)

class TestEverestOptimizersOptimization:
    """Test the optimization capabilities of everest_optimizers."""
    
    def test_rosenbrock_optimization(self):
        """Test optimization of the Rosenbrock function."""
        from everest_optimizers import minimize
        
        def rosenbrock(x):
            return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        
        def rosenbrock_grad(x):
            grad = np.zeros_like(x)
            grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
            grad[1] = 200 * (x[1] - x[0]**2)
            return grad
        
        x0 = np.array([-1.2, 1.0])
        result = minimize(rosenbrock, x0, method='OptQNewton', jac=rosenbrock_grad)
        
        assert result.success
        assert np.allclose(result.x, [1.0, 1.0], rtol=1e-3)
        assert result.fun < 1e-6
        assert result.nfev > 0
        assert result.njev > 0
        assert result.message == 'Optimization terminated successfully'
    
    def test_quadratic_optimization(self):
        """Test optimization of a simple quadratic function."""
        from everest_optimizers import minimize
        
        def quadratic(x):
            return (x[0] - 2)**2 + (x[1] - 3)**2
        
        x0 = np.array([0.0, 0.0])
        result = minimize(quadratic, x0, method='OptQNewton')
        
        assert result.success
        assert np.allclose(result.x, [2.0, 3.0], rtol=1e-3)
        assert result.fun < 1e-6
    
    def test_sphere_function(self):
        """Test optimization of the sphere function."""
        from everest_optimizers import minimize
        
        def sphere(x):
            return np.sum(x**2)
        
        def sphere_grad(x):
            return 2 * x
        
        x0 = np.array([1.0, 2.0, 3.0])
        result = minimize(sphere, x0, method='OptQNewton', jac=sphere_grad)
        
        assert result.success
        assert np.allclose(result.x, [0.0, 0.0, 0.0], atol=1e-7)
        assert result.fun < 1e-10
    
    def test_booth_function(self):
        """Test optimization of the Booth function."""
        from everest_optimizers import minimize
        
        def booth(x):
            return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
        
        def booth_grad(x):
            grad = np.zeros_like(x)
            grad[0] = 2*(x[0] + 2*x[1] - 7) + 4*(2*x[0] + x[1] - 5)
            grad[1] = 4*(x[0] + 2*x[1] - 7) + 2*(2*x[0] + x[1] - 5)
            return grad
        
        x0 = np.array([0.0, 0.0])
        result = minimize(booth, x0, method='OptQNewton', jac=booth_grad)
        
        assert result.success
        assert np.allclose(result.x, [1.0, 3.0], rtol=1e-3)
        assert result.fun < 1e-6
    
    def test_finite_difference_gradients(self):
        """Test optimization using finite difference gradients."""
        from everest_optimizers import minimize
        
        def quadratic(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2
        
        x0 = np.array([0.0, 0.0])
        # No jac parameter - should use finite differences
        result = minimize(quadratic, x0, method='OptQNewton')
        
        assert result.success
        assert np.allclose(result.x, [1.0, 2.0], rtol=1e-3)
        assert result.fun < 1e-6
        assert result.nfev > result.njev  # More function evals due to finite differences
    
    def test_different_search_strategies(self):
        """Test different search strategies."""
        from everest_optimizers import minimize
        
        def quadratic(x):
            return (x[0] - 1)**2 + (x[1] - 1)**2
        
        x0 = np.array([0.0, 0.0])
        strategies = ['TrustRegion', 'LineSearch', 'TrustPDS']
        
        for strategy in strategies:
            result = minimize(
                quadratic, x0, method='OptQNewton',
                options={'search_strategy': strategy}
            )
            
            assert result.success, f"Failed for strategy: {strategy}"
            assert np.allclose(result.x, [1.0, 1.0], rtol=1e-3), f"Wrong solution for strategy: {strategy}"
    
    def test_trust_region_size_option(self):
        """Test the trust region size option."""
        from everest_optimizers import minimize
        
        def quadratic(x):
            return (x[0] - 1)**2 + (x[1] - 1)**2
        
        x0 = np.array([0.0, 0.0])
        
        # Test with different trust region sizes
        for tr_size in [10.0, 50.0, 200.0]:
            result = minimize(
                quadratic, x0, method='OptQNewton',
                options={'tr_size': tr_size}
            )
            
            assert result.success, f"Failed for tr_size: {tr_size}"
            assert np.allclose(result.x, [1.0, 1.0], rtol=1e-3), f"Wrong solution for tr_size: {tr_size}"
    
    def test_high_dimensional_optimization(self):
        """Test optimization of a higher-dimensional function."""
        from everest_optimizers import minimize
        
        def high_dim_quadratic(x):
            return np.sum((x - np.arange(1, len(x) + 1))**2)
        
        def high_dim_grad(x):
            return 2 * (x - np.arange(1, len(x) + 1))
        
        x0 = np.zeros(5)
        result = minimize(high_dim_quadratic, x0, method='OptQNewton', jac=high_dim_grad)
        
        assert result.success
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.allclose(result.x, expected, rtol=1e-3)
        assert result.fun < 1e-6
    
    def test_optimization_with_noise(self):
        """Test optimization with a slightly noisy function."""
        from everest_optimizers import minimize
        
        def noisy_quadratic(x):
            # Add small deterministic "noise" based on position
            noise = 0.001 * np.sin(10 * x[0]) * np.cos(10 * x[1])
            return (x[0] - 1)**2 + (x[1] - 2)**2 + noise
        
        x0 = np.array([0.0, 0.0])
        result = minimize(noisy_quadratic, x0, method='OptQNewton')
        
        assert result.success
        # Should still find approximately the right solution
        assert np.allclose(result.x, [1.0, 2.0], rtol=1e-2)
    
    def test_function_evaluation_count(self):
        """Test that function evaluation counts are reasonable."""
        from everest_optimizers import minimize
        
        def quadratic(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2
        
        def quadratic_grad(x):
            return 2 * np.array([x[0] - 1, x[1] - 2])
        
        x0 = np.array([0.0, 0.0])
        
        # With analytical gradient
        result_with_grad = minimize(quadratic, x0, method='OptQNewton', jac=quadratic_grad)
        
        # Without gradient (finite differences)
        result_without_grad = minimize(quadratic, x0, method='OptQNewton')
        
        assert result_with_grad.success
        assert result_without_grad.success
        
        # Without gradient should use more function evaluations
        assert result_without_grad.nfev > result_with_grad.nfev
        
        # With gradient should have non-zero jacobian evaluations
        assert result_with_grad.njev > 0
    
    def test_starting_from_optimum(self):
        """Test behavior when starting from or near the optimum."""
        from everest_optimizers import minimize
        
        def quadratic(x):
            return (x[0] - 1)**2 + (x[1] - 2)**2
        
        # Start exactly at optimum
        x0 = np.array([1.0, 2.0])
        result = minimize(quadratic, x0, method='OptQNewton')
        
        assert result.success
        assert np.allclose(result.x, [1.0, 2.0], rtol=1e-6)
        assert result.fun < 1e-10
        
        # Start very close to optimum
        x0 = np.array([1.001, 2.001])
        result = minimize(quadratic, x0, method='OptQNewton')
        
        assert result.success
        assert np.allclose(result.x, [1.0, 2.0], rtol=1e-6)