from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult

from everest_optimizers import pyoptpp
from everest_optimizers._convert_constraints import (
    convert_bound_constraint,
    convert_linear_constraint,
    convert_nonlinear_constraint,
)
from everest_optimizers.pyoptpp import (
    BoundConstraint,
    LinearEquation,
    LinearInequality,
    NonLinearEquation,
    NonLinearInequality,
)

from ._problem import NLF1Problem


def minimize_optqnips(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    jac: Callable[..., npt.NDArray[np.float64]] | None = None,
    bounds: Bounds | None = None,
    constraints: list[LinearConstraint | NonlinearConstraint] | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """
    This implementation supports all the parameters documented in the Dakota
    quasi-Newton methods documentation.

    options only supports max_step
    """
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")

    if bounds is None and constraints is None:
        raise ValueError("Either bounds or constraints must be provided for OptQNIPS")

    if options is None:
        options = {}

    # Standard optimization parameters
    debug = options.get("debug", False)
    output_file = options.get("output_file", None)

    search_method = options.get("search_method", "trust_region")
    merit_function = options.get("merit_function", "argaez_tapia")

    # Interior-point specific parameters with Dakota defaults based on merit function
    match merit_function.lower():
        case "el_bakry":
            default_centering = 0.2
            default_step_to_boundary = 0.8
        case "argaez_tapia":
            default_centering = 0.2
            default_step_to_boundary = 0.99995
        case "van_shanno":
            default_centering = 0.1
            default_step_to_boundary = 0.95
        case _:
            default_centering = 0.2
            default_step_to_boundary = 0.95

    centering_parameter = options.get("centering_parameter", default_centering)
    steplength_to_boundary = options.get(
        "steplength_to_boundary", default_step_to_boundary
    )

    # Standard optimization control parameters
    max_iterations = options.get("max_iterations", 100)
    max_function_evaluations = options.get("max_function_evaluations", 1000)
    convergence_tolerance = options.get("convergence_tolerance", 1e-4)
    gradient_tolerance = options.get("gradient_tolerance", 1e-4)
    constraint_tolerance = options.get("constraint_tolerance", 1e-6)
    max_step = options.get("max_step", 1000.0)

    # Legacy parameters for backward compatibility
    mu = options.get("mu", 0.1)
    tr_size = options.get("tr_size", max_step)
    gradient_multiplier = options.get("gradient_multiplier", 0.1)
    search_pattern_size = options.get("search_pattern_size", 64)

    constraint_objects: list[
        BoundConstraint
        | NonLinearEquation
        | NonLinearInequality
        | LinearEquation
        | LinearInequality
    ] = []

    if bounds is not None:
        constraint_objects.append(convert_bound_constraint(bounds, len(x0)))

    if constraints is not None:
        for constraint in constraints:
            if isinstance(constraint, LinearConstraint):
                linear_constraints = convert_linear_constraint(constraint)
                constraint_objects.extend(linear_constraints)
            elif isinstance(constraint, NonlinearConstraint):
                nonlinear_constraints = convert_nonlinear_constraint(constraint, x0)
                constraint_objects.extend(nonlinear_constraints)
            else:
                raise ValueError(f"Unsupported constraint type: {type(constraint)}")

    if constraint_objects:
        cc_ptr = pyoptpp.create_compound_constraint(constraint_objects)
    else:
        raise ValueError("OptQNIPS requires at least bounds constraints")

    problem = NLF1Problem(fun, x0, args, jac)
    problem.nlf1_problem.setConstraints(cc_ptr)
    optimizer = pyoptpp.OptQNIPS(problem.nlf1_problem)

    match search_method.lower():
        case "trust_region" | "trustregion":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustRegion)
        case "line_search" | "linesearch":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.LineSearch)
        case "trust_pds" | "trustpds":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustPDS)
        case _:
            raise ValueError(
                f"Unknown search method: {search_method}. Valid options: trust_region, line_search, trust_pds"
            )

    # Set trust region parameters
    optimizer.setTRSize(tr_size)
    optimizer.setGradMult(gradient_multiplier)
    optimizer.setSearchSize(search_pattern_size)

    # Set OptQNIPS-specific parameters
    optimizer.setMu(mu)
    optimizer.setCenteringParameter(centering_parameter)
    optimizer.setStepLengthToBdry(steplength_to_boundary)

    match merit_function.lower():
        case "el_bakry":
            optimizer.setMeritFcn(pyoptpp.MeritFcn.NormFmu)
        case "argaez_tapia":
            optimizer.setMeritFcn(pyoptpp.MeritFcn.ArgaezTapia)
        case "van_shanno":
            optimizer.setMeritFcn(pyoptpp.MeritFcn.VanShanno)
        case "norm_fmu":
            optimizer.setMeritFcn(pyoptpp.MeritFcn.NormFmu)
        case merit_fn:
            raise ValueError(
                f"Unknown merit function: {merit_fn}. Valid options: el_bakry, argaez_tapia, van_shanno"
            )

    # Set optimization control parameters
    optimizer.setMaxIter(max_iterations)
    optimizer.setMaxFeval(max_function_evaluations)
    optimizer.setFcnTol(convergence_tolerance)
    optimizer.setGradTol(gradient_tolerance)
    optimizer.setConTol(constraint_tolerance)

    if "max_step" in options:
        optimizer.setTRSize(max_step)
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
        nit=0,  # OptQNIPS doesn't provide iteration count directly
        success=True,
        status=0,
        message="OptQNIPS optimization terminated successfully",
        jac=problem.current_g if problem.current_g is not None else None,
    )

    optimizer.cleanup()
    return result
