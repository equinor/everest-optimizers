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
from ._utils import remove_default_output, run_newton, set_basic_newton_options


def minimize_optqnips(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    jac: Callable[..., npt.NDArray[np.float64]] | None = None,
    bounds: Bounds | None = None,
    constraints: list[LinearConstraint | NonlinearConstraint] | None = None,
    callback: Any | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """Minimize a scalar function using the OPT++ OptQNIPS optimizer.

    This implementation supports all the parameters documented in the Dakota
     quasi-Newton methods documentation.

     options only supports max_step
    """
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")

    if bounds is None and not constraints:
        raise ValueError("Either bounds or constraints must be provided for OptQNIPS")

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

    problem = NLF1Problem(fun, x0, args, jac, callback)
    problem.nlf1_problem.setConstraints(cc_ptr)
    with remove_default_output(options):
        optimizer = pyoptpp.OptQNIPS(problem.nlf1_problem)

    # Set and remove OptQNIPS-specific options:
    if options is not None:
        merit_function = options.pop("merit_function", "argaez_tapia")
        match merit_function.lower():
            case "el_bakry":
                default_centering = 0.2
                default_step_to_boundary = 0.8
                optimizer.setMeritFcn(pyoptpp.MeritFcn.NormFmu)
            case "argaez_tapia":
                default_centering = 0.2
                default_step_to_boundary = 0.99995
                optimizer.setMeritFcn(pyoptpp.MeritFcn.ArgaezTapia)
            case "van_shanno":
                default_centering = 0.1
                default_step_to_boundary = 0.95
                optimizer.setMeritFcn(pyoptpp.MeritFcn.VanShanno)
            case merit_fn:
                raise ValueError(
                    f"Unknown merit function: {merit_fn}. Valid options: el_bakry, argaez_tapia, van_shanno"
                )

        optimizer.setConTol(options.pop("constraint_tolerance", 1e-6))
        optimizer.setMu(options.pop("mu", 0.1))
        optimizer.setCenteringParameter(
            options.pop("centering_parameter", default_centering)
        )
        optimizer.setStepLengthToBdry(
            options.pop("steplength_to_boundary", default_step_to_boundary)
        )

    set_basic_newton_options(optimizer, options)
    return run_newton(optimizer, problem, x0)
