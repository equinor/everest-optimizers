from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

from everest_optimizers import pyoptpp


def _cache_callback(
    callback: Callable[[NDArray[np.float64]], Any],
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    last_x: NDArray[np.float64] | None = None
    last_value: NDArray[np.float64] | None = None

    def evaluate(x: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal last_x, last_value
        if last_x is not None and last_value is not None and np.array_equal(last_x, x):
            return last_value
        x_copy = np.array(x, copy=True)
        value = np.array(callback(x), dtype=float, copy=True)
        last_x, last_value = x_copy, value
        return value

    return evaluate


def _create_constraint(
    constraint_func: Callable[..., Any],
    constraint_jac: Callable[..., Any],
    x0: NDArray[np.float64],
    constraint_mask: NDArray[np.bool_],
) -> pyoptpp.NLF1:
    def evaluate_constraint_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
        x_np = np.asarray(x, copy=True)
        c_values = np.atleast_1d(constraint_func(x_np))
        if c_values.shape != constraint_mask.shape:
            msg = (
                "Nonlinear constraint function size does not match its bounds; "
                "provide explicit bounds for each constraint component"
            )
            raise ValueError(msg)
        return np.asarray(c_values, dtype=float)[constraint_mask]

    def evaluate_constraint_gradient(x: NDArray[np.float64]) -> NDArray[np.float64]:
        x_np = np.asarray(x, copy=True)
        jac = constraint_jac(x_np)
        jac = np.atleast_2d(jac)
        if jac.shape != (constraint_mask.size, len(x0)):
            msg = "Nonlinear constraint Jacobian shape does not match its bounds"
            raise ValueError(msg)
        return np.asarray(jac[constraint_mask, :], dtype=float)

    x0_vector = pyoptpp.SerialDenseVector(x0)
    return pyoptpp.NLF1(
        len(x0),
        evaluate_constraint_function,
        evaluate_constraint_gradient,
        x0_vector,
        True,  # ruff: ignore[boolean-positional-value-in-call]
    )


def convert_nonlinear_constraint(
    scipy_constraint: NonlinearConstraint, x0: NDArray[np.float64]
) -> list[pyoptpp.NonLinearEquation | pyoptpp.NonLinearInequality]:
    optpp_constraints: list[
        pyoptpp.NonLinearEquation | pyoptpp.NonLinearInequality
    ] = []

    if not callable(scipy_constraint.jac):
        msg = "Nonlinear constraints require a callable jacobian"
        raise TypeError(msg)

    lower_bounds, upper_bounds = np.broadcast_arrays(
        np.atleast_1d(np.asarray(scipy_constraint.lb, dtype=float)),
        np.atleast_1d(np.asarray(scipy_constraint.ub, dtype=float)),
    )
    if lower_bounds.ndim != 1:
        msg = "Nonlinear constraint bounds must be scalar or one-dimensional"
        raise ValueError(msg)
    finite_lower = np.isfinite(lower_bounds)
    finite_upper = np.isfinite(upper_bounds)
    equality_mask = np.zeros(lower_bounds.size, dtype=bool)
    both_finite = finite_lower & finite_upper
    equality_mask[both_finite] = np.isclose(
        lower_bounds[both_finite] - upper_bounds[both_finite], 0, atol=1e-12
    )
    cached_fun = _cache_callback(scipy_constraint.fun)
    cached_jac = _cache_callback(scipy_constraint.jac)

    inequality_mask = (finite_lower | finite_upper) & ~equality_mask
    if inequality_mask.any():
        constraint = _create_constraint(cached_fun, cached_jac, x0, inequality_mask)
        nlp = pyoptpp.NLP(constraint)
        lower = pyoptpp.SerialDenseVector(lower_bounds[inequality_mask])
        upper = pyoptpp.SerialDenseVector(upper_bounds[inequality_mask])
        optpp_constraints.append(
            pyoptpp.NonLinearInequality(
                nlp, lower, upper, np.count_nonzero(inequality_mask)
            )
        )

    if equality_mask.any():
        constraint = _create_constraint(cached_fun, cached_jac, x0, equality_mask)
        nlp = pyoptpp.NLP(constraint)
        optpp_constraints.append(
            pyoptpp.NonLinearEquation(
                nlp,
                pyoptpp.SerialDenseVector(lower_bounds[equality_mask]),
                np.count_nonzero(equality_mask),
            )
        )

    return optpp_constraints


def convert_linear_constraint(
    scipy_constraint: LinearConstraint,
) -> list[pyoptpp.LinearEquation | pyoptpp.LinearInequality]:
    optpp_constraints: list[pyoptpp.LinearEquation | pyoptpp.LinearInequality] = []

    # Get constraint matrix and bounds
    coefficients = np.asarray(scipy_constraint.A, dtype=float)
    coefficients = np.atleast_2d(coefficients)

    num_constraints = coefficients.shape[0]
    for i in range(num_constraints):
        row = coefficients[i : i + 1, :]  # Keep as 2D for consistency
        matrix = pyoptpp.SerialDenseMatrix(row)
        lb = scipy_constraint.lb[i]
        ub = scipy_constraint.ub[i]

        if not np.isfinite(lb) and not np.isfinite(ub):
            # Not a real constraint as both bounds are infinite
            continue

        if np.isclose(lb - ub, 0, atol=1e-12):
            # Equality constraint: lb == ub
            rhs = pyoptpp.SerialDenseVector(lb)
            optpp_constraints.append(pyoptpp.LinearEquation(matrix, rhs))

        else:
            lower = pyoptpp.SerialDenseVector(lb)
            upper = pyoptpp.SerialDenseVector(ub)
            optpp_constraints.append(pyoptpp.LinearInequality(matrix, lower, upper))

    return optpp_constraints


def convert_bound_constraint(bounds: Bounds, x0_length: int) -> pyoptpp.BoundConstraint:
    lb_vec = pyoptpp.SerialDenseVector(np.asarray(bounds.lb, dtype=float))
    ub_vec = pyoptpp.SerialDenseVector(np.asarray(bounds.ub, dtype=float))
    return pyoptpp.BoundConstraint(x0_length, lb_vec, ub_vec)
