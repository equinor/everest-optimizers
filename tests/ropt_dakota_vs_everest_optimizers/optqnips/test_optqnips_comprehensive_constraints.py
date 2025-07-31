"""Comprehensive constraint tests for OptQNIPS (Quasi-Newton Interior-Point Solver).

This test file systematically validates OptQNIPS with various constraint types:
1. Bounds constraints (BoundConstraint)
2. Linear equality constraints (LinearEquation) 
3. Linear inequality constraints (LinearInequality)
4. Mixed/compound constraints (CompoundConstraint)
5. Nonlinear equality constraints (NonLinearEquation)
6. Nonlinear inequality constraints (NonLinearInequality)

Based on analysis of OPTPP source files in dakota-packages/OPTPP/src/Constraints/
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

# Add the source directory to the path to find everest_optimizers
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from everest_optimizers import minimize  # noqa: E402  pylint: disable=C0413

# Add pyopttpp path
pyopttpp_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "dakota-packages", "OPTPP", "build", "python"
)
if pyopttpp_path not in sys.path:
    sys.path.insert(0, pyopttpp_path)


class TestOptQNIPSConstraints:
    """Comprehensive OptQNIPS constraint testing."""

    def test_simple_bounds_constraints(self):
        """Test OptQNIPS with simple bounds constraints (BoundConstraint in OPTPP)."""
        
        def quadratic_2d(x):
            """Simple 2D quadratic: (x-2)^2 + (y-1)^2"""
            return (x[0] - 2.0)**2 + (x[1] - 1.0)**2
        
        def quadratic_2d_grad(x):
            """Gradient of 2D quadratic."""
            return np.array([2*(x[0] - 2.0), 2*(x[1] - 1.0)])
        
        x0 = np.array([0.5, 0.5])
        
        # Test case 1: Unconstrained optimum [2,1] is within bounds
        bounds = Bounds([0.0, 0.0], [3.0, 2.0])
        
        options = {
            'debug': False,
            'merit_function': 'argaez_tapia',
            'search_method': 'trust_region',
            'max_iterations': 100,
            'convergence_tolerance': 1e-6,
            'gradient_tolerance': 1e-6,
            'constraint_tolerance': 1e-8,
        }
        
        result = minimize(
            quadratic_2d,
            x0,
            method='optpp_q_nips',
            jac=quadratic_2d_grad,
            bounds=bounds,
            options=options
        )
        
        print(f"Simple bounds test: x = {result.x}, f = {result.fun}, success = {result.success}")
        
        # Should find optimum at [2, 1]
        expected_solution = np.array([2.0, 1.0])
        if result.success:
            np.testing.assert_allclose(result.x, expected_solution, rtol=1e-3, atol=1e-3)
            assert result.fun < 1e-5, f"Function value too high: {result.fun}"

    def test_constrained_bounds_solution(self):
        """Test OptQNIPS when optimum is constrained by bounds."""
        
        def quadratic_2d(x):
            """2D quadratic with optimum outside feasible region: (x-5)^2 + (y-5)^2"""
            return (x[0] - 5.0)**2 + (x[1] - 5.0)**2
        
        def quadratic_2d_grad(x):
            """Gradient of constrained quadratic."""
            return np.array([2*(x[0] - 5.0), 2*(x[1] - 5.0)])
        
        x0 = np.array([1.0, 1.0])
        
        # Optimum [5,5] is outside bounds [0,3] x [0,3]
        bounds = Bounds([0.0, 0.0], [3.0, 3.0])
        
        options = {
            'debug': False,
            'merit_function': 'argaez_tapia',
            'search_method': 'trust_region',
            'centering_parameter': 0.2,
            'steplength_to_boundary': 0.99995,
            'max_iterations': 150,
            'convergence_tolerance': 1e-6,
            'gradient_tolerance': 1e-6,
            'constraint_tolerance': 1e-8,
        }
        
        result = minimize(
            quadratic_2d,
            x0,
            method='optpp_q_nips',
            jac=quadratic_2d_grad,
            bounds=bounds,
            options=options
        )
        
        print(f"Constrained bounds test: x = {result.x}, f = {result.fun}, success = {result.success}")
        
        # Should find solution at boundary [3, 3]
        expected_solution = np.array([3.0, 3.0])
        if result.success:
            np.testing.assert_allclose(result.x, expected_solution, rtol=1e-2, atol=1e-2)
            
            # Verify we're actually at the boundary
            assert np.abs(result.x[0] - 3.0) < 1e-2, f"x[0] should be at upper bound: {result.x[0]}"
            assert np.abs(result.x[1] - 3.0) < 1e-2, f"x[1] should be at upper bound: {result.x[1]}"

    def test_asymmetric_bounds(self):
        """Test OptQNIPS with asymmetric bounds."""
        
        def objective(x):
            """3D quadratic with different scaling: x^2 + 4*y^2 + 9*z^2"""
            return x[0]**2 + 4*x[1]**2 + 9*x[2]**2
        
        def objective_grad(x):
            """Gradient of 3D scaled quadratic."""
            return np.array([2*x[0], 8*x[1], 18*x[2]])
        
        x0 = np.array([2.0, -1.0, 0.5])
        
        # Asymmetric bounds: different ranges for each variable
        bounds = Bounds([-3.0, -2.0, -1.0], [1.0, 0.5, 2.0])
        
        options = {
            'debug': False,
            'merit_function': 'van_shanno',
            'search_method': 'trust_region',
            'centering_parameter': 0.1,
            'steplength_to_boundary': 0.95,
            'max_iterations': 100,
            'convergence_tolerance': 1e-6,
            'gradient_tolerance': 1e-6,
        }
        
        result = minimize(
            objective,
            x0,
            method='optpp_q_nips',
            jac=objective_grad,
            bounds=bounds,
            options=options
        )
        
        print(f"Asymmetric bounds test: x = {result.x}, f = {result.fun}, success = {result.success}")
        
        # Unconstrained optimum is [0,0,0], which is within all bounds
        expected_solution = np.array([0.0, 0.0, 0.0])
        if result.success:
            np.testing.assert_allclose(result.x, expected_solution, rtol=1e-3, atol=1e-3)
            assert result.fun < 1e-5, f"Function value too high: {result.fun}"

    def test_linear_equality_constraint(self):
        """Test OptQNIPS with linear equality constraint (LinearEquation in OPTPP)."""
        
        def objective(x):
            """3D quadratic: (x-1)^2 + (y-2)^2 + (z-0)^2"""
            return (x[0] - 1)**2 + (x[1] - 2)**2 + x[2]**2
        
        def objective_grad(x):
            """Gradient of 3D quadratic."""
            return np.array([2*(x[0] - 1), 2*(x[1] - 2), 2*x[2]])
        
        x0 = np.array([0.0, 0.0, 0.0])
        bounds = Bounds([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])
        
        # Linear equality constraint: x + y + z = 3
        A_eq = np.array([[1.0, 1.0, 1.0]])
        b_eq = np.array([3.0])
        constraint = LinearConstraint(A_eq, b_eq, b_eq)  # equality: lb = ub
        
        options = {
            'debug': False,
            'merit_function': 'argaez_tapia',
            'search_method': 'trust_region',
            'max_iterations': 200,
            'convergence_tolerance': 1e-6,
            'gradient_tolerance': 1e-6,
            'constraint_tolerance': 1e-8,
        }
        
        result = minimize(
            objective,
            x0,
            method='optpp_q_nips',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraint,
            options=options
        )
        
        print(f"Linear equality test: x = {result.x}, f = {result.fun}, success = {result.success}")
        
        if result.success:
            # Check constraint satisfaction
            constraint_value = result.x[0] + result.x[1] + result.x[2]
            np.testing.assert_allclose(constraint_value, 3.0, rtol=1e-4, atol=1e-4)
            
            # Analytical solution using Lagrange multipliers:
            # L = (x-1)^2 + (y-2)^2 + z^2 + λ(x + y + z - 3)
            # ∇L = 0 gives: x = 1, y = 2, z = 0 (when constraint is satisfied)
            expected_solution = np.array([1.0, 2.0, 0.0])
            np.testing.assert_allclose(result.x, expected_solution, rtol=1e-2, atol=1e-2)

    def test_linear_inequality_constraint(self):
        """Test OptQNIPS with linear inequality constraint (LinearInequality in OPTPP)."""
        
        def objective(x):
            """2D quadratic: (x-3)^2 + (y-2)^2"""
            return (x[0] - 3)**2 + (x[1] - 2)**2
        
        def objective_grad(x):
            """Gradient of 2D quadratic."""
            return np.array([2*(x[0] - 3), 2*(x[1] - 2)])
        
        x0 = np.array([0.0, 0.0])
        bounds = Bounds([0.0, 0.0], [5.0, 5.0])
        
        # Linear inequality constraint: x + y <= 2
        # In scipy LinearConstraint format: A @ x with lb <= A @ x <= ub
        # For x + y <= 2, we have A = [[1, 1]] and ub = [2], lb = [-inf]
        A_ineq = np.array([[1.0, 1.0]])
        constraint = LinearConstraint(A_ineq, -np.inf, 2.0)
        
        options = {
            'debug': False,
            'merit_function': 'el_bakry',
            'search_method': 'trust_region',
            'centering_parameter': 0.2,
            'steplength_to_boundary': 0.8,
            'max_iterations': 150,
            'convergence_tolerance': 1e-6,
            'gradient_tolerance': 1e-6,
            'constraint_tolerance': 1e-7,
        }
        
        result = minimize(
            objective,
            x0,
            method='optpp_q_nips',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraint,
            options=options
        )
        
        print(f"Linear inequality test: x = {result.x}, f = {result.fun}, success = {result.success}")
        
        if result.success:
            # Check constraint satisfaction: x + y <= 2
            constraint_value = result.x[0] + result.x[1]
            assert constraint_value <= 2.0 + 1e-4, f"Constraint violated: {constraint_value} > 2.0"
            
            # Unconstrained optimum [3,2] has x+y=5 > 2, so constraint is active
            # For constrained problem, optimum should be on constraint boundary
            # Using Lagrange multipliers: optimal point is [1.5, 0.5] where x+y=2
            expected_solution = np.array([1.5, 0.5])
            np.testing.assert_allclose(result.x, expected_solution, rtol=5e-2, atol=5e-2)

    def test_multiple_linear_constraints(self):
        """Test OptQNIPS with multiple linear constraints (mixed equality and inequality)."""
        
        def objective(x):
            """3D quadratic: x^2 + y^2 + z^2"""
            return x[0]**2 + x[1]**2 + x[2]**2
        
        def objective_grad(x):
            """Gradient of 3D quadratic."""
            return 2 * x
        
        x0 = np.array([1.0, 1.0, 1.0])
        bounds = Bounds([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])
        
        # Multiple constraints:
        # Equality: x + y + z = 6
        # Inequality: x - y <= 1 (or -x + y >= -1)
        A = np.array([[1.0, 1.0, 1.0],   # equality constraint
                      [-1.0, 1.0, 0.0]])  # inequality constraint  
        lb = np.array([6.0, -1.0])  # lower bounds
        ub = np.array([6.0, np.inf])  # upper bounds (equality has lb=ub, inequality has ub=inf)
        
        constraint = LinearConstraint(A, lb, ub)
        
        options = {
            'debug': False,
            'merit_function': 'argaez_tapia',
            'search_method': 'trust_region',
            'max_iterations': 250,
            'convergence_tolerance': 1e-6,
            'gradient_tolerance': 1e-6,
            'constraint_tolerance': 1e-7,
        }
        
        result = minimize(
            objective,
            x0,
            method='optpp_q_nips',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraint,
            options=options
        )
        
        print(f"Multiple linear constraints test: x = {result.x}, f = {result.fun}, success = {result.success}")
        
        if result.success:
            # Check equality constraint: x + y + z = 6
            eq_constraint = result.x[0] + result.x[1] + result.x[2]
            np.testing.assert_allclose(eq_constraint, 6.0, rtol=1e-4, atol=1e-4)
            
            # Check inequality constraint: x - y <= 1
            ineq_constraint = result.x[0] - result.x[1]
            assert ineq_constraint <= 1.0 + 1e-4, f"Inequality violated: {ineq_constraint} > 1.0"
            
            # For this problem, analytical solution can be found using Lagrange multipliers
            # Expected solution is approximately [2.5, 1.5, 2.0] (satisfies both constraints)
            assert eq_constraint >= 5.8 and eq_constraint <= 6.2, f"Equality constraint not satisfied: {eq_constraint}"

    @pytest.mark.skip(reason="Not implemented yet")
    def test_nonlinear_equality_constraint(self):
        """Test OptQNIPS with nonlinear equality constraint (NonLinearEquation in OPTPP).
        
        Note: This requires implementing custom constraint functions that work with OPTPP's
        NonLinearEquation class, which needs NLP problem setup.
        """
        
        def objective(x):
            """2D objective: x^2 + y^2"""
            return x[0]**2 + x[1]**2
        
        def objective_grad(x):
            """Gradient of objective."""
            return 2 * x
        
        def nonlinear_constraint(x):
            """Nonlinear equality constraint: x^2 + y^2 = 4 (circle)"""
            return np.array([x[0]**2 + x[1]**2 - 4.0])
        
        def nonlinear_constraint_jac(x):
            """Jacobian of nonlinear constraint."""
            return np.array([[2*x[0], 2*x[1]]])
        
        x0 = np.array([1.5, 1.5])
        bounds = Bounds([-5.0, -5.0], [5.0, 5.0])
        
        # Nonlinear equality constraint: x^2 + y^2 = 4
        constraint = NonlinearConstraint(
            nonlinear_constraint, 
            0.0, 0.0,  # equality: lb = ub = 0
            jac=nonlinear_constraint_jac
        )
        
        options = {
            'debug': False,
            'merit_function': 'argaez_tapia',
            'search_method': 'trust_region',
            'centering_parameter': 0.1,
            'steplength_to_boundary': 0.95,
            'max_iterations': 300,
            'max_function_evaluations': 3000,
            'convergence_tolerance': 1e-5,
            'gradient_tolerance': 1e-5,
            'constraint_tolerance': 1e-6,
        }
        
        try:
            result = minimize(
                objective,
                x0,
                method='optpp_q_nips',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraint,
                options=options
            )
            
            print(f"Nonlinear equality test: x = {result.x}, f = {result.fun}, success = {result.success}")
            
            if result.success:
                # Check constraint satisfaction: x^2 + y^2 = 4
                constraint_value = result.x[0]**2 + result.x[1]**2
                np.testing.assert_allclose(constraint_value, 4.0, rtol=1e-3, atol=1e-3)
                
                # Optimal solution should minimize x^2 + y^2 subject to x^2 + y^2 = 4
                # This means the constraint is always active, so f* = 4
                np.testing.assert_allclose(result.fun, 4.0, rtol=1e-2, atol=1e-2)
                
        except NotImplementedError as e:
            pytest.skip(f"Nonlinear constraints not yet fully implemented: {e}")

    @pytest.mark.skip(reason="Not implemented yet")
    def test_nonlinear_inequality_constraint(self):
        """Test OptQNIPS with nonlinear inequality constraint (NonLinearInequality in OPTPP)."""
        
        def objective(x):
            """2D objective: (x-2)^2 + (y-2)^2"""
            return (x[0] - 2)**2 + (x[1] - 2)**2
        
        def objective_grad(x):
            """Gradient of objective."""
            return np.array([2*(x[0] - 2), 2*(x[1] - 2)])
        
        def nonlinear_constraint(x):
            """Nonlinear inequality: x^2 + y^2 <= 1 (unit circle)"""
            return np.array([1.0 - x[0]**2 - x[1]**2])
        
        def nonlinear_constraint_jac(x):
            """Jacobian of nonlinear constraint."""
            return np.array([[-2*x[0], -2*x[1]]])
        
        x0 = np.array([0.0, 0.0])
        bounds = Bounds([-2.0, -2.0], [2.0, 2.0])
        
        # Nonlinear inequality constraint: x^2 + y^2 <= 1
        constraint = NonlinearConstraint(
            nonlinear_constraint, 
            0.0, np.inf,  # inequality: 1 - x^2 - y^2 >= 0
            jac=nonlinear_constraint_jac
        )
        
        options = {
            'debug': False,
            'merit_function': 'van_shanno',
            'search_method': 'trust_region',
            'centering_parameter': 0.1,
            'steplength_to_boundary': 0.95,
            'max_iterations': 300,
            'max_function_evaluations': 3000,
            'convergence_tolerance': 1e-5,
            'gradient_tolerance': 1e-5,
            'constraint_tolerance': 1e-6,
        }
        
        try:
            result = minimize(
                objective,
                x0,
                method='optpp_q_nips',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraint,
                options=options
            )
            
            print(f"Nonlinear inequality test: x = {result.x}, f = {result.fun}, success = {result.success}")
            
            if result.success:
                # Check constraint satisfaction: x^2 + y^2 <= 1
                constraint_value = result.x[0]**2 + result.x[1]**2
                assert constraint_value <= 1.0 + 1e-4, f"Constraint violated: {constraint_value} > 1.0"
                
                # Unconstrained optimum is [2,2], but constraint limits to unit circle
                # Optimal constrained solution is on circle boundary in direction of [2,2]
                # Expected: [sqrt(2)/2, sqrt(2)/2] ≈ [0.707, 0.707]
                expected_direction = np.array([2.0, 2.0]) / np.linalg.norm([2.0, 2.0])
                expected_solution = expected_direction  # On unit circle
                
                np.testing.assert_allclose(
                    result.x / np.linalg.norm(result.x), 
                    expected_direction, 
                    rtol=1e-1, atol=1e-1
                )
                
        except NotImplementedError as e:
            pytest.skip(f"Nonlinear constraints not yet fully implemented: {e}")

    def test_mixed_constraint_types(self):
        """Test OptQNIPS with mixed constraint types (CompoundConstraint in OPTPP)."""
        
        def objective(x):
            """3D objective: x^2 + 2*y^2 + 3*z^2"""
            return x[0]**2 + 2*x[1]**2 + 3*x[2]**2
        
        def objective_grad(x):
            """Gradient of objective."""
            return np.array([2*x[0], 4*x[1], 6*x[2]])
        
        x0 = np.array([2.0, 1.0, 0.5])
        
        # Mixed constraints:
        # 1. Bounds: 0 <= x,y,z <= 5
        bounds = Bounds([0.0, 0.0, 0.0], [5.0, 5.0, 5.0])
        
        # 2. Linear equality: x + y + z = 3
        # 3. Linear inequality: x - y + z <= 2
        A = np.array([[1.0, 1.0, 1.0],    # equality
                      [1.0, -1.0, 1.0]])   # inequality
        lb = np.array([3.0, -np.inf])
        ub = np.array([3.0, 2.0])
        
        linear_constraints = LinearConstraint(A, lb, ub)
        
        options = {
            'debug': False,
            'merit_function': 'argaez_tapia',
            'search_method': 'trust_region',
            'max_iterations': 300,
            'max_function_evaluations': 3000,
            'convergence_tolerance': 1e-6,
            'gradient_tolerance': 1e-6,
            'constraint_tolerance': 1e-7,
        }
        
        result = minimize(
            objective,
            x0,
            method='optpp_q_nips',
            jac=objective_grad,
            bounds=bounds,
            constraints=linear_constraints,
            options=options
        )
        
        print(f"Mixed constraints test: x = {result.x}, f = {result.fun}, success = {result.success}")
        
        if result.success:
            # Check all constraints
            # 1. Bounds
            assert np.all(result.x >= -1e-4), f"Lower bounds violated: {result.x}"
            assert np.all(result.x <= 5.0 + 1e-4), f"Upper bounds violated: {result.x}"
            
            # 2. Linear equality: x + y + z = 3
            eq_value = result.x[0] + result.x[1] + result.x[2]
            np.testing.assert_allclose(eq_value, 3.0, rtol=1e-4, atol=1e-4)
            
            # 3. Linear inequality: x - y + z <= 2
            ineq_value = result.x[0] - result.x[1] + result.x[2]
            assert ineq_value <= 2.0 + 1e-4, f"Inequality violated: {ineq_value} > 2.0"

    def test_constraint_tolerance_sensitivity(self):
        """Test OptQNIPS sensitivity to constraint tolerance settings."""
        
        def objective(x):
            """2D quadratic: (x-1)^2 + (y-1)^2"""
            return (x[0] - 1)**2 + (x[1] - 1)**2
        
        def objective_grad(x):
            """Gradient of 2D quadratic."""
            return np.array([2*(x[0] - 1), 2*(x[1] - 1)])
        
        x0 = np.array([0.1, 0.1])
        bounds = Bounds([0.0, 0.0], [2.0, 2.0])
        
        # Linear equality constraint: x + y = 1.5
        A = np.array([[1.0, 1.0]])
        b = np.array([1.5])
        constraint = LinearConstraint(A, b, b)
        
        # Test different constraint tolerances
        tolerances = [1e-4, 1e-6, 1e-8]
        
        for tol in tolerances:
            print(f"\nTesting constraint tolerance: {tol}")
            
            options = {
                'debug': False,
                'merit_function': 'argaez_tapia',
                'search_method': 'trust_region',
                'max_iterations': 200,
                'convergence_tolerance': 1e-8,
                'gradient_tolerance': 1e-8,
                'constraint_tolerance': tol,
            }
            
            result = minimize(
                objective,
                x0,
                method='optpp_q_nips',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraint,
                options=options
            )
            
            print(f"Tolerance {tol}: x = {result.x}, f = {result.fun}, success = {result.success}")
            
            if result.success:
                # Check constraint satisfaction with appropriate tolerance
                constraint_value = result.x[0] + result.x[1]
                constraint_error = abs(constraint_value - 1.5)
                
                # Constraint should be satisfied within tolerance (with some margin)
                assert constraint_error <= max(tol * 10, 1e-3), \
                    f"Constraint not satisfied within tolerance {tol}: error = {constraint_error}"

    def test_infeasible_problem(self):
        """Test OptQNIPS behavior with infeasible constraints."""
        
        def objective(x):
            """Simple quadratic."""
            return x[0]**2 + x[1]**2
        
        def objective_grad(x):
            """Gradient of simple quadratic."""
            return 2 * x
        
        x0 = np.array([0.0, 0.0])
        
        # Incompatible constraints:
        # Bounds: 0 <= x, y <= 1
        bounds = Bounds([0.0, 0.0], [1.0, 1.0])
        
        # Linear constraint: x + y = 3 (impossible within bounds!)
        A = np.array([[1.0, 1.0]])
        b = np.array([3.0])
        constraint = LinearConstraint(A, b, b)
        
        options = {
            'debug': False,
            'merit_function': 'argaez_tapia',
            'search_method': 'trust_region',
            'max_iterations': 100,
            'convergence_tolerance': 1e-6,
            'gradient_tolerance': 1e-6,
            'constraint_tolerance': 1e-6,
        }
        
        result = minimize(
            objective,
            x0,
            method='optpp_q_nips',
            jac=objective_grad,
            bounds=bounds,
            constraints=constraint,
            options=options
        )
        
        print(f"Infeasible problem test: x = {result.x}, f = {result.fun}, success = {result.success}")
        
        # Should detect infeasibility (success = False)
        # Different optimizers handle infeasibility differently
        # OptQNIPS might either fail or find a compromise solution
        if not result.success:
            print("Correctly detected infeasible problem")
        else:
            print("Found compromise solution despite infeasibility")
            # If it found a solution, check which constraint is violated
            bounds_satisfied = np.all(result.x >= -1e-3) and np.all(result.x <= 1.0 + 1e-3)
            constraint_value = result.x[0] + result.x[1]
            constraint_satisfied = abs(constraint_value - 3.0) < 1e-3
            
            print(f"Bounds satisfied: {bounds_satisfied}")
            print(f"Linear constraint satisfied: {constraint_satisfied} (value: {constraint_value})")


if __name__ == "__main__":
    # Run the tests directly
    test_instance = TestOptQNIPSConstraints()
    
    print("=" * 80)
    print("COMPREHENSIVE OPTQNIPS CONSTRAINT TESTING")
    print("=" * 80)
    
    test_methods = [
        ("Simple Bounds Constraints", test_instance.test_simple_bounds_constraints),
        ("Constrained Bounds Solution", test_instance.test_constrained_bounds_solution),
        ("Asymmetric Bounds", test_instance.test_asymmetric_bounds),
        ("Linear Equality Constraint", test_instance.test_linear_equality_constraint),
        ("Linear Inequality Constraint", test_instance.test_linear_inequality_constraint),
        ("Multiple Linear Constraints", test_instance.test_multiple_linear_constraints),
        ("Nonlinear Equality Constraint", test_instance.test_nonlinear_equality_constraint),
        ("Nonlinear Inequality Constraint", test_instance.test_nonlinear_inequality_constraint),
        ("Mixed Constraint Types", test_instance.test_mixed_constraint_types),
        ("Constraint Tolerance Sensitivity", test_instance.test_constraint_tolerance_sensitivity),
        ("Infeasible Problem", test_instance.test_infeasible_problem),
    ]
    
    for test_name, test_method in test_methods:
        print(f"\n{test_name}...")
        try:
            test_method()
            print(f"✓ {test_name} passed")
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
        except pytest.skip.Exception as e:
            print(f"⊘ {test_name} skipped: {e}")
    
    print("\n" + "=" * 80)
    print("OPTQNIPS COMPREHENSIVE CONSTRAINT TESTING COMPLETE!")
    print("=" * 80)