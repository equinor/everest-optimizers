import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint

from everest_optimizers import pyoptpp


def _create_constraint(constraint_func, constraint_jac, x0, constraint_index):
    # Create callback functions for constraint evaluation
    def eval_cf(x):
        x_np = np.array(x.to_numpy(), copy=True)
        try:
            c_values = np.atleast_1d(constraint_func(x_np))
            result = c_values[constraint_index]

            return np.array([result])
        except Exception as e:
            raise RuntimeError(f"Error evaluating nonlinear constraint: {e}") from e

    def eval_cg(x):
        x_np = np.array(x.to_numpy(), copy=True)
        try:
            if constraint_jac is not None:
                jac = constraint_jac(x_np)
                jac = np.atleast_2d(jac)

                grad_row = jac[constraint_index, :]
            else:
                grad_row = _finite_difference_constraint_gradient(
                    x_np, constraint_func, constraint_index
                )

            return grad_row.reshape(len(x0), 1)
        except Exception as e:
            raise RuntimeError(
                f"Error evaluating nonlinear constraint gradient: {e}"
            ) from e

    x0_vector = pyoptpp.SerialDenseVector(x0)
    nlf1 = pyoptpp.NLF1.create_constrained(len(x0), eval_cf, eval_cg, x0_vector)
    return nlf1


def _finite_difference_constraint_gradient(x, constraint_func, constraint_index):
    """Compute constraint gradient using finite differences (standalone helper)."""
    eps = 1e-8
    n = len(x)
    grad = np.zeros(n)

    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps

        c_plus = np.atleast_1d(constraint_func(x_plus))
        c_minus = np.atleast_1d(constraint_func(x_minus))

        c_plus_val = c_plus[constraint_index]
        c_minus_val = c_minus[constraint_index]

        grad[i] = (c_plus_val - c_minus_val) / (2 * eps)

    return grad


def convert_nonlinear_constraint(
    scipy_constraint: NonlinearConstraint, x0: npt.NDArray
) -> list[pyoptpp.NonLinearEquation | pyoptpp.NonLinearInequality]:
    """
    Convert a scipy.optimize.NonlinearConstraint to
    OPTPP NonLinearEquation/NonLinearInequality objects.

    Following the OPTPP pattern from hockfcns.C examples
    """
    optpp_constraints = []

    # Get constraint bounds
    lb = np.asarray(scipy_constraint.lb, dtype=float)
    ub = np.asarray(scipy_constraint.ub, dtype=float)

    lb = np.atleast_1d(lb)
    ub = np.atleast_1d(ub)

    # Evaluate constraint at initial point to determine number of constraints
    constraint_values = scipy_constraint.fun(x0)
    constraint_values = np.atleast_1d(constraint_values)
    num_constraints = len(constraint_values)

    for i in range(num_constraints):
        constraint = _create_constraint(
            scipy_constraint.fun, scipy_constraint.jac, x0, i
        )

        if not np.isfinite(lb[i]) and not np.isfinite(ub[i]):
            # Both bounds are infinite - this is not a real constraint
            continue

        if np.isclose(lb[i] - ub[i], 0, atol=1e-12):
            # Equality constraint: lb == ub, so c(x) = lb
            nlp_wrapper = pyoptpp.NLP.create(constraint)
            rhs = pyoptpp.SerialDenseVector(lb[i])
            constraint = pyoptpp.NonLinearEquation.create(nlp_wrapper, rhs)
            optpp_constraints.append(constraint)
            continue

        nlp_wrapper = pyoptpp.NLP.create(constraint)
        lower = pyoptpp.SerialDenseVector(lb[i])
        upper = pyoptpp.SerialDenseVector(ub[i])
        constraint = pyoptpp.NonLinearInequality.create(nlp_wrapper, lower, upper)
        optpp_constraints.append(constraint)

    return optpp_constraints


def convert_linear_constraint(
    scipy_constraint: LinearConstraint,
) -> list[pyoptpp.LinearEquation | pyoptpp.LinearInequality]:
    """
    Convert a scipy.optimize.LinearConstraint to
    OPTPP LinearEquation/LinearInequality objects.
    """
    optpp_constraints = []

    # Get constraint matrix and bounds
    A = np.asarray(scipy_constraint.A, dtype=float)
    A = np.atleast_2d(A)

    num_constraints = A.shape[0]
    for i in range(num_constraints):
        A_row = A[i : i + 1, :]  # Keep as 2D for consistency
        A_matrix = pyoptpp.SerialDenseMatrix(A_row)
        lb = scipy_constraint.lb[i]
        ub = scipy_constraint.ub[i]

        if not np.isfinite(lb) and not np.isfinite(ub):
            # Both bounds are infinite - this is not a real constraint
            continue

        if np.isclose(lb - ub, 0, atol=1e-12):
            # Equality constraint: lb == ub
            rhs = pyoptpp.SerialDenseVector(lb)
            constraint = pyoptpp.LinearEquation.create(A_matrix, rhs)
            optpp_constraints.append(constraint)
            continue

        lower = pyoptpp.SerialDenseVector(lb)
        upper = pyoptpp.SerialDenseVector(ub)
        constraint = pyoptpp.LinearInequality.create(A_matrix, lower, upper)
        optpp_constraints.append(constraint)

    return optpp_constraints


def convert_bound_constraint(bounds: Bounds, x0_length: int) -> pyoptpp.BoundConstraint:
    # OPTPP uses a large number for infinity
    optpp_inf = 1.0e30

    lb = np.asarray(bounds.lb, dtype=float)
    ub = np.asarray(bounds.ub, dtype=float)
    lb[np.isneginf(lb)] = -optpp_inf
    ub[np.isposinf(ub)] = optpp_inf
    lb_vec = pyoptpp.SerialDenseVector(lb)
    ub_vec = pyoptpp.SerialDenseVector(ub)
    return pyoptpp.BoundConstraint.create(x0_length, lb_vec, ub_vec)
