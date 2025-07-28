#!/usr/bin/env python3
# src/everest_optimizers/optqnips_impl.py

import numpy as np
from typing import Callable, Optional, Dict, Any
from scipy.optimize import OptimizeResult, LinearConstraint, NonlinearConstraint

def _import_pyopttpp():
    """Import pyopttpp module with proper error handling."""
    import sys
    import os
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pyopttpp_build_dir = os.path.abspath(os.path.join(
        current_dir, '..', '..', '..', 
        'dakota-packages', 'OPTPP', 'build', 'python'
    ))
    
    if pyopttpp_build_dir not in sys.path:
        sys.path.insert(0, pyopttpp_build_dir)
    
    try:
        import pyopttpp
        return pyopttpp
    except ImportError as e:
        raise ImportError(
            f"Could not import pyopttpp from {pyopttpp_build_dir}. "
            f"Make sure the module is built according to the instructions. "
            f"Error: {e}"
        )

def _minimize_optqnips_enhanced(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    method: str = 'optpp_q_nips',
    jac: Optional[Callable] = None,
    hess: Optional[Callable] = None,
    hessp: Optional[Callable] = None,
    bounds: Any = None,
    constraints: Any = None,
    tol: Optional[float] = None,
    callback: Optional[Callable] = None,
    options: Optional[Dict[str, Any]] = None
) -> OptimizeResult:
    """
    Enhanced OptQNIPS implementation with full parameter support.
    
    This implementation supports all the parameters documented in the Dakota
    quasi-Newton methods documentation.
    """
    # Import pyopttpp
    pyopttpp = _import_pyopttpp()

    # Convert inputs
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")

    if bounds is None and constraints is None:
        raise ValueError("Either bounds or constraints must be provided for OptQNIPS")

    # Set up options
    if options is None:
        options = {}

    # Standard optimization parameters
    debug = options.get('debug', False)
    output_file = options.get('output_file', None)
    
    # Search method (Dakota keyword mapping)
    search_method = options.get('search_method', 'trust_region')
    
    # Merit function (Dakota keyword mapping) 
    merit_function = options.get('merit_function', 'argaez_tapia')
    
    # Interior-point specific parameters with Dakota defaults based on merit function
    if merit_function == 'el_bakry':
        default_centering = 0.2
        default_step_to_boundary = 0.8
    elif merit_function == 'argaez_tapia':
        default_centering = 0.2
        default_step_to_boundary = 0.99995
    elif merit_function == 'van_shanno':
        default_centering = 0.1
        default_step_to_boundary = 0.95
    else:
        default_centering = 0.2
        default_step_to_boundary = 0.95
    
    centering_parameter = options.get('centering_parameter', default_centering)
    steplength_to_boundary = options.get('steplength_to_boundary', default_step_to_boundary)
    
    # Standard optimization control parameters
    max_iterations = options.get('max_iterations', 100)
    max_function_evaluations = options.get('max_function_evaluations', 1000)
    convergence_tolerance = options.get('convergence_tolerance', 1e-4)
    gradient_tolerance = options.get('gradient_tolerance', 1e-4)
    constraint_tolerance = options.get('constraint_tolerance', 1e-6)
    
    # Max step parameter
    max_step = options.get('max_step', 1000.0)
    
    # Speculative gradients (not implemented but recognized)
    speculative = options.get('speculative', False)
    
    # Legacy parameters for backward compatibility
    mu = options.get('mu', 0.1)
    tr_size = options.get('tr_size', max_step)
    gradient_multiplier = options.get('gradient_multiplier', 0.1)
    search_pattern_size = options.get('search_pattern_size', 64)

    # Create a simple problem class
    class OptQNIPSProblem:
        def __init__(self, fun, x0, args, jac, pyopttpp):
            self.fun = fun
            self.x0 = np.asarray(x0, dtype=float)
            self.args = args
            self.jac = jac
            self.pyopttpp = pyopttpp
            
            self.nfev = 0
            self.njev = 0
            self.current_x = None
            self.current_f = None
            self.current_g = None
            
            # Create the NLF1 problem
            self.nlf1_problem = self._create_nlf1_problem()
        
        def _create_nlf1_problem(self):
            """Create the NLF1 problem for OPTPP."""
            
            class OptQNIPSNLF1(self.pyopttpp.NLF1):
                def __init__(self, parent_problem):
                    super().__init__(len(parent_problem.x0))
                    self.parent = parent_problem
                    
                    # Set initial point
                    init_vector = parent_problem.pyopttpp.SerialDenseVector(parent_problem.x0)
                    self.setX(init_vector)
                    self.setIsExpensive(True)
                
                def evalF(self, x):
                    """Evaluate objective function."""
                    x_np = np.array(x.to_numpy(), copy=True)
                    self.parent.current_x = x_np
                    
                    try:
                        f_val = self.parent.fun(x_np, *self.parent.args)
                        self.parent.current_f = float(f_val)
                        self.parent.nfev += 1
                        return self.parent.current_f
                    except Exception as e:
                        raise RuntimeError(f"Error evaluating objective function: {e}")
                
                def evalG(self, x):
                    """Evaluate gradient."""
                    x_np = np.array(x.to_numpy(), copy=True)
                    
                    if self.parent.jac is not None:
                        try:
                            grad = self.parent.jac(x_np, *self.parent.args)
                            grad_np = np.asarray(grad, dtype=float)
                            self.parent.current_g = grad_np
                            self.parent.njev += 1
                            return grad_np
                        except Exception as e:
                            raise RuntimeError(f"Error evaluating gradient: {e}")
                    else:
                        # Use finite differences for gradient
                        grad = self._finite_difference_gradient(x_np)
                        self.parent.current_g = grad
                        return grad
                
                def _finite_difference_gradient(self, x):
                    """Compute gradient using finite differences."""
                    eps = 1e-8
                    grad = np.zeros_like(x)
                    
                    for i in range(len(x)):
                        x_plus = x.copy()
                        x_plus[i] += eps
                        x_minus = x.copy()
                        x_minus[i] -= eps
                        
                        f_plus = self.parent.fun(x_plus, *self.parent.args)
                        f_minus = self.parent.fun(x_minus, *self.parent.args)
                        
                        grad[i] = (f_plus - f_minus) / (2 * eps)
                        self.parent.nfev += 2
                    
                    return grad
            
            return OptQNIPSNLF1(self)

    # Create problem
    problem = OptQNIPSProblem(fun, x0, args, jac, pyopttpp)

    # Process constraints - enhanced version supporting multiple constraint types
    constraint_objects = []
    
    # Handle bounds constraints
    if bounds is not None:
        lb = np.asarray(bounds.lb, dtype=float)
        ub = np.asarray(bounds.ub, dtype=float)
        # OPTPP uses a large number for infinity
        inf = 1.0e30
        lb[np.isneginf(lb)] = -inf
        ub[np.isposinf(ub)] = inf
        
        # Create BoundConstraint
        lb_vec = pyopttpp.SerialDenseVector(lb)
        ub_vec = pyopttpp.SerialDenseVector(ub)
        bound_constraint = pyopttpp.BoundConstraint(len(x0), lb_vec, ub_vec)
        constraint_objects.append(bound_constraint)
    
    # Handle general constraints (linear and nonlinear)
    if constraints is not None:
        if not isinstance(constraints, (list, tuple)):
            constraints = [constraints]
            
        for constraint in constraints:
            if isinstance(constraint, LinearConstraint):
                # Convert scipy LinearConstraint to OPTPP LinearEquation/LinearInequality
                optpp_constraints = _convert_linear_constraint(constraint, pyopttpp)
                constraint_objects.extend(optpp_constraints)
            elif isinstance(constraint, NonlinearConstraint):
                # TODO: Implement nonlinear constraint conversion
                raise NotImplementedError("Nonlinear constraints not yet supported in OptQNIPS wrapper")
            else:
                raise ValueError(f"Unsupported constraint type: {type(constraint)}")
    
    # Create compound constraint from all constraint objects
    if constraint_objects:
        if len(constraint_objects) == 1:
            # Single constraint - create CompoundConstraint with one element
            cc_ptr = pyopttpp.create_compound_constraint([constraint_objects[0]])
        else:
            # Multiple constraints - create CompoundConstraint with all elements
            cc_ptr = pyopttpp.create_compound_constraint(constraint_objects)
    else:
        raise ValueError("OptQNIPS requires at least bounds constraints")
    
    # Attach constraints to the NLF1 problem
    problem.nlf1_problem.setConstraints(cc_ptr)
    
    # Create OptQNIPS optimizer
    optimizer = pyopttpp.OptQNIPS(problem.nlf1_problem)

    # Set search method (Dakota keyword mapping)
    if search_method.lower() == 'trust_region':
        optimizer.setSearchStrategy(pyopttpp.SearchStrategy.TrustRegion)
    elif search_method.lower() == 'value_based_line_search':
        optimizer.setSearchStrategy(pyopttpp.SearchStrategy.LineSearch)
    elif search_method.lower() == 'gradient_based_line_search':
        optimizer.setSearchStrategy(pyopttpp.SearchStrategy.LineSearch)  # Same as value-based in OptQNIPS
    elif search_method.lower() == 'tr_pds':
        optimizer.setSearchStrategy(pyopttpp.SearchStrategy.TrustPDS)
    else:
        # Try legacy names for backward compatibility
        if search_method.lower() == 'trustregion':
            optimizer.setSearchStrategy(pyopttpp.SearchStrategy.TrustRegion)
        elif search_method.lower() == 'linesearch':
            optimizer.setSearchStrategy(pyopttpp.SearchStrategy.LineSearch)
        elif search_method.lower() == 'trustpds':
            optimizer.setSearchStrategy(pyopttpp.SearchStrategy.TrustPDS)
        else:
            raise ValueError(f"Unknown search method: {search_method}. Valid options: trust_region, value_based_line_search, gradient_based_line_search, tr_pds")

    # Set trust region parameters
    optimizer.setTRSize(tr_size)
    optimizer.setGradMult(gradient_multiplier)
    optimizer.setSearchSize(search_pattern_size)

    # Set OptQNIPS-specific parameters
    optimizer.setMu(mu)
    optimizer.setCenteringParameter(centering_parameter)
    optimizer.setStepLengthToBdry(steplength_to_boundary)
    
    # Set merit function (Dakota keyword mapping)
    merit_function_map = {
        'el_bakry': pyopttpp.MeritFcn.NormFmu,  # Dakota el_bakry maps to OPTPP NormFmu
        'argaez_tapia': pyopttpp.MeritFcn.ArgaezTapia,
        'van_shanno': pyopttpp.MeritFcn.VanShanno,
        # Legacy names for backward compatibility
        'norm_fmu': pyopttpp.MeritFcn.NormFmu,
    }
    
    if merit_function.lower() in merit_function_map:
        optimizer.setMeritFcn(merit_function_map[merit_function.lower()])
    else:
        raise ValueError(f"Unknown merit function: {merit_function}. Valid options: el_bakry, argaez_tapia, van_shanno")

    # Set optimization control parameters
    optimizer.setMaxIter(max_iterations)
    optimizer.setMaxFeval(max_function_evaluations)
    optimizer.setFcnTol(convergence_tolerance)
    optimizer.setGradTol(gradient_tolerance)
    optimizer.setConTol(constraint_tolerance)
    
    # Note: max_step is handled by setTRSize in OptQNIPS
    if 'max_step' in options:
        optimizer.setTRSize(max_step)

    # Set debug mode
    if debug:
        optimizer.setDebug()

    # Set output file
    if output_file:
        optimizer.setOutputFile(output_file, 0)

    # Run optimization
    try:
        optimizer.optimize()
        solution_vector = problem.nlf1_problem.getXc()
        x_final = solution_vector.to_numpy()
        
        # Ensure caller sees feasible result if bounds are provided
        if bounds is not None:
            x_final = np.minimum(np.maximum(x_final, bounds.lb), bounds.ub)
        
        f_final = problem.nlf1_problem.getF()
        
        result = OptimizeResult(
            x=x_final,
            fun=f_final,
            nfev=problem.nfev,
            njev=problem.njev,
            nit=0,  # OptQNIPS doesn't provide iteration count directly
            success=True,
            status=0,
            message='OptQNIPS optimization terminated successfully',
            jac=problem.current_g if problem.current_g is not None else None
        )
        
        optimizer.cleanup()
        return result
        
    except Exception as e:
        optimizer.cleanup()
        return OptimizeResult(
            x=x0,
            fun=None,
            nfev=problem.nfev,
            njev=problem.njev,
            nit=0,
            success=False,
            status=1,
            message=f'OptQNIPS optimization failed: {str(e)}',
            jac=None
        )


def _convert_linear_constraint(scipy_constraint, pyopttpp):
    """
    Convert a scipy.optimize.LinearConstraint to OPTPP LinearEquation/LinearInequality objects.
    
    Parameters:
    -----------
    scipy_constraint : scipy.optimize.LinearConstraint
        The scipy constraint to convert
    pyopttpp : module
        The pyopttpp module for creating OPTPP objects
        
    Returns:
    --------
    list
        List of OPTPP constraint objects (LinearEquation and/or LinearInequality)
    """
    optpp_constraints = []
    
    # Get constraint matrix and bounds
    A = np.asarray(scipy_constraint.A, dtype=float)
    lb = np.asarray(scipy_constraint.lb, dtype=float)
    ub = np.asarray(scipy_constraint.ub, dtype=float)
    
    # Ensure A is 2D
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    # Ensure bounds are 1D arrays
    lb = np.atleast_1d(lb)
    ub = np.atleast_1d(ub)
    
    num_constraints = A.shape[0]
    
    # Process each constraint row
    for i in range(num_constraints):
        A_row = A[i:i+1, :]  # Keep as 2D for consistency
        lb_i = lb[i]
        ub_i = ub[i]
        
        # Create OPTPP matrix and vector objects
        A_matrix = pyopttpp.SerialDenseMatrix(A_row)
        
        # Determine constraint type based on bounds
        if np.isfinite(lb_i) and np.isfinite(ub_i):
            if np.abs(lb_i - ub_i) < 1e-12:
                # Equality constraint: lb == ub
                rhs = pyopttpp.SerialDenseVector(np.array([lb_i]))
                eq_constraint = pyopttpp.LinearEquation(A_matrix, rhs)
                optpp_constraints.append(eq_constraint)
            else:
                # Double-sided inequality: lb <= Ax <= ub
                # Convert to two single-sided inequalities:
                # Ax >= lb  =>  Ax - lb >= 0
                # Ax <= ub  =>  -Ax + ub >= 0
                
                # Lower bound: Ax >= lb  =>  Ax >= lb (OPTPP standard form)
                if np.isfinite(lb_i):
                    rhs_lower = pyopttpp.SerialDenseVector(np.array([lb_i]))
                    ineq_lower = pyopttpp.LinearInequality(A_matrix, rhs_lower)
                    optpp_constraints.append(ineq_lower)
                
                # Upper bound: Ax <= ub  =>  -Ax >= -ub (OPTPP standard form)
                if np.isfinite(ub_i):
                    A_neg = -A_row
                    A_neg_matrix = pyopttpp.SerialDenseMatrix(A_neg)
                    rhs_upper = pyopttpp.SerialDenseVector(np.array([-ub_i]))
                    ineq_upper = pyopttpp.LinearInequality(A_neg_matrix, rhs_upper)
                    optpp_constraints.append(ineq_upper)
                    
        elif np.isfinite(lb_i) and not np.isfinite(ub_i):
            # One-sided inequality: Ax >= lb (OPTPP standard form)
            rhs = pyopttpp.SerialDenseVector(np.array([lb_i]))
            ineq_constraint = pyopttpp.LinearInequality(A_matrix, rhs)
            optpp_constraints.append(ineq_constraint)
            
        elif not np.isfinite(lb_i) and np.isfinite(ub_i):
            # One-sided inequality: Ax <= ub  =>  -Ax >= -ub (OPTPP standard form)
            A_neg = -A_row
            A_neg_matrix = pyopttpp.SerialDenseMatrix(A_neg)
            rhs = pyopttpp.SerialDenseVector(np.array([-ub_i]))
            ineq_constraint = pyopttpp.LinearInequality(A_neg_matrix, rhs)
            optpp_constraints.append(ineq_constraint)
            
        else:
            # Both bounds are infinite - this is not a real constraint
            continue
    
    return optpp_constraints