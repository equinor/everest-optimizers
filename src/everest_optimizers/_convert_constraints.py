import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, approx_fprime

from everest_optimizers import pyoptpp


def _create_constraint(constraint_func, constraint_jac, x0, constraint_index):
    def evaluate_constraint_function(x):
        x_np = np.array(x.to_numpy(), copy=True)
        c_values = np.atleast_1d(constraint_func(x_np))
        result = c_values[constraint_index]
        return np.array([result])

    def evaluate_constraint_gradient(x):
        x_np = np.array(x.to_numpy(), copy=True)
        if constraint_jac is not None:
            jac = constraint_jac(x_np)
            jac = np.atleast_2d(jac)
            grad_row = jac[constraint_index, :]
        else:
            grad_row = approx_fprime(
                x_np,
                lambda x_arr: np.atleast_1d(constraint_func(x_arr))[constraint_index],
                epsilon=1e-8,
            )
        return grad_row.reshape(len(x0), 1)

    x0_vector = pyoptpp.SerialDenseVector(x0)
    nlf1 = pyoptpp.NLF1(
        len(x0),
        evaluate_constraint_function,
        evaluate_constraint_gradient,
        x0_vector,
        True,
    )
    return nlf1


def convert_nonlinear_constraint(
    scipy_constraint: NonlinearConstraint, x0: npt.NDArray
) -> list[pyoptpp.NonLinearEquation | pyoptpp.NonLinearInequality]:
    """
    Convert a scipy.optimize.NonlinearConstraint to
    OPTPP NonLinearEquation/NonLinearInequality objects.

    Following the OPTPP pattern from hockfcns.C examples
    """
    optpp_constraints: list[
        pyoptpp.NonLinearEquation | pyoptpp.NonLinearInequality
    ] = []

    # Get constraint bounds
    lower_bounds = np.atleast_1d(scipy_constraint.lb).astype(float)
    upper_bounds = np.atleast_1d(scipy_constraint.ub).astype(float)

    # Evaluate constraint at initial point to determine number of constraints
    constraint_values = scipy_constraint.fun(x0)
    num_constraints = len(constraint_values)

    for i in range(num_constraints):
        constraint = _create_constraint(
            scipy_constraint.fun, scipy_constraint.jac, x0, i
        )

        if not np.isfinite(lower_bounds[i]) and not np.isfinite(upper_bounds[i]):
            # Not a real constraint as both bounds are infinite
            continue

        elif np.isclose(lower_bounds[i] - upper_bounds[i], 0, atol=1e-12):
            # Equality constraint: lb == ub
            nlp = pyoptpp.NLP(constraint)
            rhs = pyoptpp.SerialDenseVector(lower_bounds[i])
            optpp_constraints.append(pyoptpp.NonLinearEquation(nlp, rhs))

        else:
            nlp = pyoptpp.NLP(constraint)
            lower = pyoptpp.SerialDenseVector(lower_bounds[i])
            upper = pyoptpp.SerialDenseVector(upper_bounds[i])
            optpp_constraints.append(pyoptpp.NonLinearInequality(nlp, lower, upper))

    return optpp_constraints


def convert_linear_constraint(
    scipy_constraint: LinearConstraint,
) -> list[pyoptpp.LinearEquation | pyoptpp.LinearInequality]:
    """
    Convert a scipy.optimize.LinearConstraint to
    OPTPP LinearEquation/LinearInequality objects.
    """
    optpp_constraints: list[pyoptpp.LinearEquation | pyoptpp.LinearInequality] = []

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
            # Not a real constraint as both bounds are infinite
            continue

        elif np.isclose(lb - ub, 0, atol=1e-12):
            # Equality constraint: lb == ub
            rhs = pyoptpp.SerialDenseVector(lb)
            optpp_constraints.append(pyoptpp.LinearEquation(A_matrix, rhs))

        else:
            lower = pyoptpp.SerialDenseVector(lb)
            upper = pyoptpp.SerialDenseVector(ub)
            optpp_constraints.append(pyoptpp.LinearInequality(A_matrix, lower, upper))

    return optpp_constraints


def convert_bound_constraint(bounds: Bounds, x0_length: int) -> pyoptpp.BoundConstraint:
    lb_vec = pyoptpp.SerialDenseVector(np.asarray(bounds.lb, dtype=float))
    ub_vec = pyoptpp.SerialDenseVector(np.asarray(bounds.ub, dtype=float))
    return pyoptpp.BoundConstraint(x0_length, lb_vec, ub_vec)
