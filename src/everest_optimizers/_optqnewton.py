from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult

from everest_optimizers import pyoptpp

from ._problem import NLF1Problem
from ._utils import remove_default_output, run_newton, set_basic_newton_options


def minimize_optqnewton(
    fun: Callable,
    x0: npt.NDArray,
    args: tuple = (),
    jac: Callable[..., npt.NDArray[np.float64]] | None = None,
    bounds: Bounds | None = None,
    constraints: list[LinearConstraint | NonlinearConstraint] | None = None,
    callback: Any | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """
    Minimize a scalar function using the OPT++ OptQNewton optimizer.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : ndarray
        Initial guess. Must be 1d.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives.
    jac : callable, optional
        Method for computing the gradient vector.
    bounds : sequence, optional
        Bounds on variables (not supported by optpp_q_newton).
    constraints : list, optional
        Constraints definition (not supported by optpp_q_newton).
    options : dict, optional
        Solver options including:
        - 'search_method': 'TrustRegion', 'LineSearch', or 'TrustPDS'
        - 'tr_size': Trust region size
        - 'debug': Enable debug output
        - 'output_file': Output file for debugging

    Returns
    -------
    OptimizeResult
        The optimization result.
    """
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")

    if bounds is not None:
        raise NotImplementedError("optpp_q_newton does not support bounds")

    if constraints:
        raise NotImplementedError("optpp_q_newton does not support constraints")

    problem = NLF1Problem(fun, x0, args, jac, callback)
    with remove_default_output(options):
        optimizer = pyoptpp.OptQNewton(problem.nlf1_problem)
    set_basic_newton_options(optimizer, options)
    return run_newton(optimizer, problem, x0)
