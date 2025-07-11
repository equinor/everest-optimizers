#!/usr/bin/env python3
# src/everest_optimizers/minimize.py

import numpy as np
from typing import Callable, Optional, Union, Dict, Any
from scipy.optimize import OptimizeResult

from .optqnewton import _minimize_optqnewton

def minimize(
    fun: Callable,
    x0: Union[np.ndarray, list],
    args: tuple = (),
    method: str = 'OptQNewton',
    jac: Optional[Union[Callable, str, bool]] = None,
    hess: Optional[Union[Callable, str]] = None,
    hessp: Optional[Callable] = None,
    bounds: Optional[Any] = None,
    constraints: Optional[Any] = None,
    tol: Optional[float] = None,
    callback: Optional[Callable] = None,
    options: Optional[Dict[str, Any]] = None
) -> OptimizeResult:
    """
    Minimization of scalar function of one or more variables.
    
    This function provides a unified interface to various optimization algorithms,
    similar to scipy.optimize.minimize but with additional optimizers from 
    everest-optimizers.
    
    Parameters
    ----------
    fun : callable
        The objective function to be minimized:
        
            fun(x, *args) -> float
        
        where x is a 1-D array with shape (n,) and args is a tuple of the
        fixed parameters needed to completely specify the function.
    
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,), where n is the
        number of independent variables.
    
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives
        (fun, jac functions).
    
    method : str, optional
        Type of solver. Currently supported:
        - 'OptQNewton': OptQNewton optimizer from OPTPP
        
        More optimizers may be added in the future.
    
    jac : {callable, str, bool}, optional
        Method for computing the gradient vector. If it is a callable, it should
        be a function that returns the gradient vector:
        
            jac(x, *args) -> array_like, shape (n,)
        
        If None, gradients will be estimated using finite differences.
    
    hess : {callable, str}, optional
        Method for computing the Hessian matrix. Not used by OptQNewton.
    
    hessp : callable, optional
        Hessian times vector product. Not used by OptQNewton.
    
    bounds : sequence, optional
        Bounds on variables. Not supported by OptQNewton.
    
    constraints : dict or list, optional
        Constraints definition. Not supported by OptQNewton.
    
    tol : float, optional
        Tolerance for termination.
    
    callback : callable, optional
        Callback function. Not implemented for OptQNewton.
    
    options : dict, optional
        A dictionary of solver options. For OptQNewton, supported options are:
        
        - 'search_strategy' : str
            Search strategy: 'TrustRegion' (default), 'LineSearch', or 'TrustPDS'
        - 'tr_size' : float
            Trust region size (default: 100.0)
        - 'debug' : bool
            Enable debug output (default: False)
        - 'output_file' : str
            Output file for debug information (default: None)
    
    Returns
    -------
    res : OptimizeResult
        The optimization result represented as an OptimizeResult object.
        Important attributes are:
        - x : ndarray
            The solution array
        - fun : float
            Value of the objective function at the solution
        - success : bool
            Whether the optimizer exited successfully
        - message : str
            Description of the termination cause
        - nfev : int
            Number of function evaluations
        - njev : int
            Number of jacobian evaluations
    
    Notes
    -----
    This function is designed to be a drop-in replacement for scipy.optimize.minimize
    for the supported methods. The OptQNewton method is a quasi-Newton optimization
    algorithm from the OPTPP library.
    
    Examples
    --------
    Minimize the Rosenbrock function:
    
    >>> import numpy as np
    >>> from everest_optimizers import minimize
    >>> 
    >>> def rosenbrock(x):
    ...     return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    >>> 
    >>> def rosenbrock_grad(x):
    ...     grad = np.zeros_like(x)
    ...     grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
    ...     grad[1] = 200 * (x[1] - x[0]**2)
    ...     return grad
    >>> 
    >>> x0 = np.array([-1.2, 1.0])
    >>> result = minimize(rosenbrock, x0, method='OptQNewton', jac=rosenbrock_grad)
    >>> print(result.x)  # Should be close to [1.0, 1.0]
    """
    # Convert x0 to numpy array
    x0 = np.asarray(x0, dtype=float)
    
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")
    
    # Convert args to tuple if not already
    if not isinstance(args, tuple):
        args = (args,)
    
    # Route to the appropriate optimizer
    if method.lower() == 'optqnewton':
        return _minimize_optqnewton(
            fun=fun,
            x0=x0,
            args=args,
            method=method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            constraints=constraints,
            tol=tol,
            callback=callback,
            options=options
        )
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: 'OptQNewton'")