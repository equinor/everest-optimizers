#!/usr/bin/env python3

from collections.abc import Callable, Collection
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult

from everest_optimizers._conminmfd import minimize_conmin_mfd
from everest_optimizers._optbcqnewton import minimize_optbcqnewton
from everest_optimizers._optqnewton import minimize_optqnewton
from everest_optimizers._optqnips import minimize_optqnips


def minimize(
    fun: Callable,
    x0: npt.NDArray[np.float64],
    args: tuple | None = (),
    method: str = "optpp_q_newton",
    jac: Callable[..., npt.NDArray[np.float64]] | None = None,
    bounds: Bounds | None = None,
    constraints: list[LinearConstraint | NonlinearConstraint]
    | LinearConstraint
    | NonlinearConstraint
    | None = None,
    callback: Callable | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """
    Minimization of scalar function of one or more variables.

    This function is intended to be a drop-in replacement for
    scipy.optimize.minimize. The optpp_q_newton method is a quasi-Newton
    optimization algorithm from the OPTPP library.

    Parameters (parameter structure is based on scipy.optimize.minimize)
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
        - 'optpp_q_newton': optpp_q_newton optimizer from OPTPP
        - 'optpp_q_nips': quasi-Newton interior-point solver from OPTPP
        - 'conmin_mfd': CONMIN solver

        More optimizers may be added in the future.

    jac : callable, optional
        Method for computing the gradient vector. If it is a callable, it should
        be a function that returns the gradient vector:

            jac(x, *args) -> array_like, shape (n,)

        If None, gradients will be estimated using finite differences.

    bounds : sequence, optional
        Bounds on variables. Supported by 'optpp_q_nips'.

    constraints : dict or list, optional
        Constraints definition. Supported by 'optpp_q_nips'.

    callback : callable, optional
        Callback function. Not implemented for optpp_q_newton.

    options : dict, optional
        A dictionary of solver options. For optpp_q_newton, supported options are:

        - 'search_method' : str
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
    for the supported methods. The optpp_q_newton method is a quasi-Newton optimization
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
    >>> result = minimize(rosenbrock, x0, method='optpp_q_newton', jac=rosenbrock_grad)
    >>> print(result.x)  # Should be close to [1.0, 1.0]
    """
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")

    if not isinstance(args, tuple):
        args = (args,)

    if constraints is not None and not isinstance(constraints, Collection):
        constraints = [constraints]

    match method.lower():
        case "optpp_q_newton":
            return minimize_optqnewton(
                fun=fun,
                x0=x0,
                args=args,
                jac=jac,
                bounds=bounds,
                constraints=constraints,
                callback=callback,
                options=options,
            )
        case "optpp_bcq_newton":
            return minimize_optbcqnewton(
                fun=fun,
                x0=x0,
                args=args,
                jac=jac,
                bounds=bounds,
                constraints=constraints,
                callback=callback,
                options=options,
            )
        case "conmin_mfd":
            return minimize_conmin_mfd(
                fun=fun,
                x0=x0,
                args=args,
                bounds=bounds,
                constraints=constraints,
                options=options,
            )
        case "optpp_q_nips":
            return minimize_optqnips(
                fun=fun,
                x0=x0,
                args=args,
                jac=jac,
                bounds=bounds,
                constraints=constraints,
                options=options,
            )
        case other:
            raise ValueError(
                f"Unknown method: {other}. Supported methods: 'optpp_q_newton', 'optpp_bcq_newton', 'optpp_q_nips', 'conmin_mfd'"
            )
