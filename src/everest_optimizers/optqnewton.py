#!/usr/bin/env python3
# src/everest_optimizers/optqnewton.py

import sys
import os
import numpy as np
from typing import Callable, Optional, Dict, Any
from scipy.optimize import OptimizeResult

def _get_pyopttpp_path() -> str:
    """Get the path to the pyopttpp module."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pyopttpp_build_dir = os.path.abspath(os.path.join(
        current_dir, '..', '..', '..', 
        'dakota-packages', 'OPTPP', 'build', 'python'
    ))
    return pyopttpp_build_dir

def _import_pyopttpp():
    """Import pyopttpp module with proper error handling."""
    pyopttpp_path = _get_pyopttpp_path()
    
    if pyopttpp_path not in sys.path:
        sys.path.insert(0, pyopttpp_path)
    
    try:
        import pyopttpp
        return pyopttpp
    except ImportError as e:
        raise ImportError(
            f"Could not import pyopttpp from {pyopttpp_path}. "
            f"Make sure the module is built according to the instructions. "
            f"Error: {e}"
        )

class _OptQNewtonProblem:
    """Problem definition for OptQNewton optimizer."""
    
    def __init__(self, fun: Callable, x0: np.ndarray, args: tuple, 
                 jac: Optional[Callable] = None, pyopttpp_module=None):
        self.fun = fun
        self.x0 = np.asarray(x0, dtype=float)
        self.args = args
        self.jac = jac
        self.pyopttpp = pyopttpp_module
        
        self.nfev = 0
        self.njev = 0
        self.current_x = None
        self.current_f = None
        self.current_g = None
        
        # Create the NLF1 problem
        self.nlf1_problem = self._create_nlf1_problem()
    
    def _create_nlf1_problem(self):
        """Create the NLF1 problem for OPTPP."""
        
        class OptQNewtonNLF1(self.pyopttpp.NLF1):
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
        
        return OptQNewtonNLF1(self)

def _minimize_optqnewton(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    method: str = 'optpp_q_newton',
    jac: Optional[Callable] = None,
    hess: Optional[Callable] = None,
    hessp: Optional[Callable] = None,
    bounds: Optional[Any] = None,
    constraints: Optional[Any] = None,
    tol: Optional[float] = None,
    callback: Optional[Callable] = None,
    options: Optional[Dict[str, Any]] = None
) -> OptimizeResult:
    """
    Minimize a scalar function using optpp_q_newton optimizer.
    
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives.
    method : str, optional
        Method name (should be 'optpp_q_newton').
    jac : callable, optional
        Method for computing the gradient vector.
    hess : callable, optional
        Method for computing the Hessian matrix (not used by optpp_q_newton).
    hessp : callable, optional
        Hessian times vector product (not used by optpp_q_newton).
    bounds : sequence, optional
        Bounds on variables (not supported by optpp_q_newton).
    constraints : dict or list, optional
        Constraints definition (not supported by optpp_q_newton).
    tol : float, optional
        Tolerance for termination.
    callback : callable, optional
        Callback function (not implemented).
    options : dict, optional
        Solver options including:
        - 'search_strategy': 'TrustRegion', 'LineSearch', or 'TrustPDS'
        - 'tr_size': Trust region size
        - 'debug': Enable debug output
        - 'output_file': Output file for debugging
    
    Returns
    -------
    OptimizeResult
        The optimization result.
    """
    # Import pyopttpp
    pyopttpp = _import_pyopttpp()
    
    # Convert inputs
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")
    
    if bounds is not None:
        raise NotImplementedError("optpp_q_newton does not support bounds")
    
    if constraints is not None:
        raise NotImplementedError("optpp_q_newton does not support constraints")
    
    if callback is not None:
        raise NotImplementedError("Callback function not implemented for optpp_q_newton")
    
    # Set up options
    if options is None:
        options = {}
    
    search_strategy = options.get('search_strategy', 'TrustRegion')
    tr_size = options.get('tr_size', 100.0)
    debug = options.get('debug', False)
    output_file = options.get('output_file', None)
    
    # Create problem
    problem = _OptQNewtonProblem(fun, x0, args, jac, pyopttpp)
    
    # Create optimizer
    optimizer = pyopttpp.OptQNewton(problem.nlf1_problem)
    
    # Set search strategy
    if search_strategy == 'TrustRegion':
        optimizer.setSearchStrategy(pyopttpp.SearchStrategy.TrustRegion)
    elif search_strategy == 'LineSearch':
        optimizer.setSearchStrategy(pyopttpp.SearchStrategy.LineSearch)
    elif search_strategy == 'TrustPDS':
        optimizer.setSearchStrategy(pyopttpp.SearchStrategy.TrustPDS)
    else:
        raise ValueError(f"Unknown search strategy: {search_strategy}")
    
    # Set trust region size
    optimizer.setTRSize(tr_size)
    
    # Set debug mode
    if debug:
        optimizer.setDebug()
    
    # Set output file
    if output_file:
        optimizer.setOutputFile(output_file, 0)
    
    # Run optimization
    try:
        optimizer.optimize()
        
        # Get results
        solution_vector = problem.nlf1_problem.getXc()
        x_final = solution_vector.to_numpy()
        f_final = problem.nlf1_problem.getF()
        
        # Create result
        result = OptimizeResult(
            x=x_final,
            fun=f_final,
            nfev=problem.nfev,
            njev=problem.njev,
            nit=0,  # optpp_q_newton doesn't provide iteration count
            success=True,
            status=0,
            message='Optimization terminated successfully',
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
            message=f'Optimization failed: {str(e)}',
            jac=None
        )


def _create_nlf2_problem(fun: Callable, x0: np.ndarray, args: tuple,
                          jac: Optional[Callable], hess: Optional[Callable],
                          hessp: Optional[Callable],
                          bounds: Any, constraints: Any,
                          pyopttpp_module) -> Any:
    """Create the NLF2 problem for OPTPP (second derivatives)."""
    raise NotImplementedError("NLF2 problem creation is not implemented yet.")


def _minimize_optconstrqnewton(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    method: str = 'optpp_constr_q_newton',
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
    Minimize a scalar function with bound constraints using OptConstrQNewton.
    """
    # Import pyopttpp
    pyopttpp = _import_pyopttpp()

    # Convert inputs
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")

    if constraints is not None:
        raise NotImplementedError("optpp_constr_q_newton does not support arbitrary constraints yet")

    if bounds is None:
        raise ValueError("Bounds must be provided for constrained optimization")

    # Set up options
    if options is None:
        options = {}

    search_strategy = options.get('search_strategy', 'TrustRegion')
    tr_size = options.get('tr_size', 100.0)
    debug = options.get('debug', False)
    output_file = options.get('output_file', None)

    # Create constrained problem by clamping within bounds
    class _OptConstrNLF1(_OptQNewtonProblem):
        def __init__(self, fun, x0, args, jac, pyopttpp, bounds):
            # Store bounds before superclass init to ensure availability in _create_nlf1_problem
            self.lb = np.asarray(bounds.lb, dtype=float)
            self.ub = np.asarray(bounds.ub, dtype=float)
            super().__init__(fun, x0, args, jac, pyopttpp)

        def _create_nlf1_problem(self):
            parent = self
            lb = parent.lb
            ub = parent.ub

            class ConstrNLF1(parent.pyopttpp.NLF1):
                def __init__(self, parent_problem):
                    super().__init__(len(parent_problem.x0))
                    self.parent = parent_problem
                    init_vector = parent_problem.pyopttpp.SerialDenseVector(parent_problem.x0)
                    self.setX(init_vector)
                    self.setIsExpensive(True)

                def evalF(self, x):
                    x_np = np.array(x.to_numpy(), copy=True)
                    x_clamped = np.minimum(np.maximum(x_np, lb), ub)
                    self.parent.current_x = x_clamped
                    try:
                        f_val = self.parent.fun(x_clamped, *self.parent.args)
                        self.parent.current_f = float(f_val)
                        self.parent.nfev += 1
                        return self.parent.current_f
                    except Exception as e:
                        raise RuntimeError(f"Error evaluating objective function: {e}")

                def evalG(self, x):
                    x_np = np.array(x.to_numpy(), copy=True)
                    x_clamped = np.minimum(np.maximum(x_np, lb), ub)
                    if self.parent.jac is not None:
                        try:
                            grad = self.parent.jac(x_clamped, *self.parent.args)
                            grad_np = np.asarray(grad, dtype=float)
                            self.parent.current_g = grad_np
                            self.parent.njev += 1
                            return grad_np
                        except Exception as e:
                            raise RuntimeError(f"Error evaluating gradient: {e}")
                    # Finite-difference approximation when jacobian not provided
                    eps = 1e-8
                    grad = np.zeros_like(x_clamped)
                    for i in range(len(x_clamped)):
                        x_plus = x_clamped.copy()
                        x_plus[i] += eps
                        x_minus = x_clamped.copy()
                        x_minus[i] -= eps
                        f_plus = self.parent.fun(x_plus, *self.parent.args)
                        f_minus = self.parent.fun(x_minus, *self.parent.args)
                        grad[i] = (f_plus - f_minus) / (2 * eps)
                        self.parent.nfev += 2
                    self.parent.current_g = grad
                    return grad

            return ConstrNLF1(self)

    # Initialize constrained problem
    problem = _OptConstrNLF1(fun, x0, args, jac, pyopttpp, bounds)

    # Create C++ bound constraint object and constrained optimizer
    cc_ptr = pyopttpp.create_compound_constraint(np.asarray(bounds.lb, dtype=float),
                                                 np.asarray(bounds.ub, dtype=float))
    # attach constraints to the NLF1 problem
    problem.nlf1_problem.setConstraints(cc_ptr)
    optimizer = pyopttpp.OptConstrQNewton(problem.nlf1_problem)

    # Set search strategy
    if search_strategy == 'TrustRegion':
        optimizer.setSearchStrategy(pyopttpp.SearchStrategy.TrustRegion)
    elif search_strategy == 'LineSearch':
        optimizer.setSearchStrategy(pyopttpp.SearchStrategy.LineSearch)
    elif search_strategy == 'TrustPDS':
        optimizer.setSearchStrategy(pyopttpp.SearchStrategy.TrustPDS)
    else:
        raise ValueError(f"Unknown search strategy: {search_strategy}")

    # Set trust region size
    optimizer.setTRSize(tr_size)

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
        # Ensure caller sees feasible result
        x_final = np.minimum(np.maximum(x_final, bounds.lb), bounds.ub)
        f_final = problem.nlf1_problem.getF()
        result = OptimizeResult(
            x=x_final,
            fun=f_final,
            nfev=problem.nfev,
            njev=problem.njev,
            nit=0,
            success=True,
            status=0,
            message='Optimization terminated successfully',
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
            message=f'Optimization failed: {str(e)}',
            jac=None
        )
