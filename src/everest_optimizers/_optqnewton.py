from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult

from everest_optimizers import pyoptpp

from ._problem import NLF1Problem


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
    Minimize a scalar function using optpp_q_newton optimizer.

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
        - 'search_strategy': 'TrustRegion', 'LineSearch', or 'TrustPDS'
        - 'tr_size': Trust region size
        - 'debug': Enable debug output
        - 'output_file': Output file for debugging

    Returns
    -------
    OptimizeResult
        The optimization result.
    """
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")

    if bounds is not None:
        raise NotImplementedError("optpp_q_newton does not support bounds")

    if constraints is not None:
        raise NotImplementedError("optpp_q_newton does not support constraints")

    if options is None:
        options = {}

    search_strategy = options.get("search_strategy", "TrustRegion")
    tr_size = options.get("tr_size", 100.0)
    debug = options.get("debug", False)
    output_file = options.get("output_file", None)

    problem = NLF1Problem(fun, x0, args, jac, callback)
    optimizer = pyoptpp.OptQNewton(problem.nlf1_problem)

    match search_strategy:
        case "TrustRegion":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustRegion)
        case "LineSearch":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.LineSearch)
        case "TrustPDS":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustPDS)
        case other:
            raise ValueError(
                f"Unknown search strategy: {other}. Valid options: TrustRegion, LineSearch, TrustPDS"
            )

    optimizer.setTRSize(tr_size)
    if debug:
        optimizer.setDebug()
    if output_file:
        optimizer.setOutputFile(output_file, 0)

    optimizer.optimize()

    solution_vector = problem.nlf1_problem.getXc()
    x_final = solution_vector.to_numpy()
    f_final = problem.nlf1_problem.getF()

    result = OptimizeResult(  # type: ignore[call-arg]
        x=x_final,
        fun=f_final,
        nfev=problem.nfev,
        njev=problem.njev,
        nit=0,  # optpp_q_newton doesn't provide iteration count
        success=True,
        status=0,
        message="Optimization terminated successfully",
        jac=problem.current_g if problem.current_g is not None else None,
    )

    optimizer.cleanup()
    return result
