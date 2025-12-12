from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult

from everest_optimizers import pyoptpp
from everest_optimizers._convert_constraints import (
    convert_bound_constraint,
)

from ._problem import NLF1Problem


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
    Minimize a scalar function using optpp_bcq_newton optimizer.

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
        - 'search_strategy': 'TrustRegion', 'LineSearch', or 'TrustPDS'
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

    if constraints is not None:
        raise NotImplementedError("optpp_bcq_newton does not support constraints")

    if options is None:
        options = {}

    # Make sure to start with a feasible estimate:
    x0 = np.minimum(np.maximum(x0, bounds.lb), bounds.ub)

    search_strategy = options.get("search_strategy", "LineSearch")
    debug = options.get("debug", False)
    output_file = options.get("output_file", None)

    # Standard optimization control parameters
    max_iterations = options.get("max_iterations", 100)
    max_function_evaluations = options.get("max_function_evaluations", 1000)
    convergence_tolerance = options.get("convergence_tolerance", 1e-4)
    gradient_tolerance = options.get("gradient_tolerance", 1e-4)
    max_step = options.get("max_step", 1000.0)

    # Legacy parameters for backward compatibility
    tr_size = options.get("tr_size", max_step)
    gradient_multiplier = options.get("gradient_multiplier", 0.1)
    search_pattern_size = options.get("search_pattern_size", 64)

    problem = NLF1Problem(fun, x0, args, jac, callback)
    cc_ptr = pyoptpp.create_compound_constraint(
        [convert_bound_constraint(bounds, len(x0))]
    )
    problem.nlf1_problem.setConstraints(cc_ptr)
    optimizer = pyoptpp.OptBCQNewton(problem.nlf1_problem)

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
    optimizer.setGradMult(gradient_multiplier)
    optimizer.setSearchSize(search_pattern_size)

    # Set optimization control parameters
    optimizer.setMaxIter(max_iterations)
    optimizer.setMaxFeval(max_function_evaluations)
    optimizer.setFcnTol(convergence_tolerance)
    optimizer.setGradTol(gradient_tolerance)

    if "max_step" in options:
        optimizer.setTRSize(max_step)
    if debug:
        optimizer.setDebug()
    if output_file:
        optimizer.setOutputFile(output_file, 0)

    try:
        optimizer.optimize()

        solution_vector = problem.nlf1_problem.getXc()
        x_final = solution_vector.to_numpy()
        f_final = problem.nlf1_problem.getF()

        result = OptimizeResult(  # type: ignore[call-arg]
            x=x_final,
            fun=f_final,
            nfev=problem.nfev,
            njev=problem.njev,
            nit=0,  # optpp_bcq_newton doesn't provide iteration count
            success=True,
            status=0,
            message="Optimization terminated successfully",
            jac=problem.current_g if problem.current_g is not None else None,
        )

        optimizer.cleanup()
        return result

    except Exception as e:
        optimizer.cleanup()
        return OptimizeResult(  # type: ignore[call-arg]
            x=x0,
            fun=None,
            nfev=problem.nfev,
            njev=problem.njev,
            nit=0,
            success=False,
            status=1,
            message=f"Optimization failed: {e!s}",
            jac=None,
        )
