from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult

from everest_optimizers import pyoptpp
from everest_optimizers._convert_constraints import convert_bound_constraint

from ._problem import NLF1Problem
from ._utils import remove_default_output, run_newton, set_basic_newton_options


def minimize_optbcqnewton(
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
    Minimize a scalar function using the OPT++ OptBCQNewton optimizer.

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
        Bounds on variables (required by optpp_bcq_newton).
    constraints : list, optional
        Constraints definition (not supported by optpp_bcq_newton).
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

    if bounds is None:
        raise ValueError("OptBCQNewton requires bound constraints")

    if constraints:
        raise NotImplementedError("optpp_bcq_newton does not support constraints")

    # Make sure to start with a feasible estimate:
    x0 = np.minimum(np.maximum(x0, bounds.lb), bounds.ub)

    problem = NLF1Problem(fun, x0, args, jac, callback)
    cc_ptr = pyoptpp.create_compound_constraint(
        [convert_bound_constraint(bounds, len(x0))]
    )
    problem.nlf1_problem.setConstraints(cc_ptr)
    with remove_default_output(options):
        optimizer = pyoptpp.OptBCQNewton(problem.nlf1_problem)
    set_basic_newton_options(optimizer, options)
    return run_newton(optimizer, problem, x0)
