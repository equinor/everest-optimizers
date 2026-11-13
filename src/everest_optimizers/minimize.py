"""Minimization with the OPT++ `optpp_q_newton` method."""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._evaluation import Evaluator
from ._optpp import Bounds, LinearConstraints
from ._optpp_q_newton import minimize_optpp_q_newton


def minimize(  # ruff: ignore[too-many-arguments]
    x0: NDArray[np.float64],
    *,
    evaluate: Evaluator,
    bounds: Bounds | None = None,
    linear_constraints: LinearConstraints | None = None,
    nonlinear_equalities: NDArray[np.bool_] | None = None,
    output_file: Path | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[NDArray[np.float64], int]:
    """Minimize an objective with the `optpp_q_newton` method.

    Args:
        x0:                   The initial point, a one-dimensional array.
        evaluate:             Evaluates the objective and the nonlinear constraints.
        bounds:               The lower and upper bounds of the variables.
        linear_constraints:   The coefficients, lower bounds and upper bounds of
                              the linear constraints.
        nonlinear_equalities: For each nonlinear constraint, whether it is an
                              equality.
        output_file:          The file to receive OPT++ output, or `None` to
                              discard it.
        options:              The options, keyed by option name.

    Returns:
        The final point and the OPT++ termination status.

    Raises:
        ValueError: If an argument has an invalid shape, a lower bound exceeds
                    its upper bound, or an option is invalid.

    """
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.ndim != 1:
        msg = "x0 must be one-dimensional"
        raise ValueError(msg)
    if bounds is not None:
        lower, upper = bounds
        bounds = (
            np.broadcast_to(np.asarray(lower, dtype=np.float64), x0.shape),
            np.broadcast_to(np.asarray(upper, dtype=np.float64), x0.shape),
        )
        if np.any(bounds[0] > bounds[1]):
            msg = "The lower bounds must not exceed the upper bounds"
            raise ValueError(msg)
    if linear_constraints is not None:
        linear_constraints = _linear_constraints(linear_constraints, x0.size)
    equalities = np.asarray(
        [] if nonlinear_equalities is None else nonlinear_equalities, dtype=np.bool_
    )
    if equalities.ndim != 1:
        msg = "nonlinear_equalities must be one-dimensional"
        raise ValueError(msg)
    return minimize_optpp_q_newton(
        evaluate, x0, bounds, linear_constraints, equalities, output_file, options
    )


def _linear_constraints(
    linear_constraints: LinearConstraints, size: int
) -> LinearConstraints | None:
    coefficients, lower, upper = linear_constraints
    coefficients = np.atleast_2d(np.asarray(coefficients, dtype=np.float64))
    rows = coefficients.shape[0]
    if coefficients.shape != (rows, size):
        msg = f"The linear constraint coefficients must have shape (m, {size})"
        raise ValueError(msg)
    if rows == 0:
        return None
    return (
        coefficients,
        np.broadcast_to(np.asarray(lower, dtype=np.float64), (rows,)),
        np.broadcast_to(np.asarray(upper, dtype=np.float64), (rows,)),
    )
