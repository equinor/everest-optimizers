from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult

from everest_optimizers import pyoptpp
from everest_optimizers._convert_constraints import convert_bound_constraint

from ._problem import NLF1Problem
from ._utils import remove_default_output, run_newton, set_basic_newton_options


def minimize_optbcqnewton(  # noqa: PLR0913, PLR0917
    fun: Callable[..., float],
    x0: NDArray[np.float64],
    args: tuple[Any, ...] = (),
    jac: Callable[..., NDArray[np.float64]] | None = None,
    bounds: Bounds | None = None,
    constraints: list[LinearConstraint | NonlinearConstraint] | None = None,
    callback: Callable[..., None] | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        msg = "x0 must be 1-dimensional"
        raise ValueError(msg)

    if bounds is None:
        msg = "OptBCQNewton requires bound constraints"
        raise ValueError(msg)

    if constraints:
        msg = "optpp_bcq_newton does not support constraints"
        raise NotImplementedError(msg)

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
