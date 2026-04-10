from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult

from everest_optimizers import pyoptpp

from ._problem import NLF1Problem
from ._utils import remove_default_output, run_newton, set_basic_newton_options


def minimize_optqnewton(  # noqa: PLR0913, PLR0917
    fun: Callable[..., float],
    x0: npt.NDArray[np.float64],
    args: tuple[Any, ...] = (),
    jac: Callable[..., npt.NDArray[np.float64]] | None = None,
    bounds: Bounds | None = None,
    constraints: list[LinearConstraint | NonlinearConstraint] | None = None,
    callback: Callable[..., None] | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        msg = "x0 must be 1-dimensional"
        raise ValueError(msg)

    if bounds is not None:
        msg = "optpp_q_newton does not support bounds"
        raise NotImplementedError(msg)

    if constraints:
        msg = "optpp_q_newton does not support constraints"
        raise NotImplementedError(msg)

    problem = NLF1Problem(fun, x0, args, jac, callback)
    with remove_default_output(options):
        optimizer = pyoptpp.OptQNewton(problem.nlf1_problem)
    set_basic_newton_options(optimizer, options)
    return run_newton(optimizer, problem, x0)
