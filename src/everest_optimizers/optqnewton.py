#!/usr/bin/env python3
# src/everest_optimizers/optqnewton.py

import sys
import os
import numpy as np
from typing import Callable, Optional, Union, Dict, Any
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
    method: str = 'OptQNewton',
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
    Minimize a scalar function using OptQNewton optimizer.
    
    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives.
    method : str, optional
        Method name (should be 'OptQNewton').
    jac : callable, optional
        Method for computing the gradient vector.
    hess : callable, optional
        Method for computing the Hessian matrix (not used by OptQNewton).
    hessp : callable, optional
        Hessian times vector product (not used by OptQNewton).
    bounds : sequence, optional
        Bounds on variables (not supported by OptQNewton).
    constraints : dict or list, optional
        Constraints definition (not supported by OptQNewton).
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
        raise NotImplementedError("OptQNewton does not support bounds")
    
    if constraints is not None:
        raise NotImplementedError("OptQNewton does not support constraints")
    
    if callback is not None:
        raise NotImplementedError("Callback function not implemented for OptQNewton")
    
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
            nit=0,  # OptQNewton doesn't provide iteration count
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