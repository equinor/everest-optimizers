import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

from everest_optimizers import pyoptpp


def _create_constraint_nlf1(
    constraint_func, constraint_jac, x0_ref, constraint_index, is_negated=False
):
    # Create callback functions for constraint evaluation
    def eval_cf(x):
        x_np = np.array(x.to_numpy(), copy=True)
        try:
            c_values = np.atleast_1d(constraint_func(x_np))
            result = c_values[constraint_index]

            if is_negated:
                result = -result

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

            if is_negated:
                grad_row = -grad_row

            return grad_row.reshape(len(x0_ref), 1)
        except Exception as e:
            raise RuntimeError(
                f"Error evaluating nonlinear constraint gradient: {e}"
            ) from e

    x0_vector = pyoptpp.SerialDenseVector(x0_ref)
    nlf1_base = pyoptpp.NLF1.create_constrained(
        len(x0_ref), eval_cf, eval_cg, x0_vector
    )
    return nlf1_base


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


def convert_nonlinear_constraint(scipy_constraint: NonlinearConstraint, x0):
    """
    Convert a scipy.optimize.NonlinearConstraint to OPTPP NonLinearEquation/NonLinearInequality objects.

    Following the OPTPP pattern from hockfcns.C examples

    Parameters:
    -----------
    scipy_constraint : scipy.optimize.NonlinearConstraint
        The scipy constraint to convert
    x0 : np.ndarray
        Initial point (needed for constraint function evaluation)

    Returns:
    --------
    list
        List of OPTPP nonlinear constraint objects
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
        if not np.isfinite(lb[i]) and not np.isfinite(ub[i]):
            # Both bounds are infinite - this is not a real constraint
            continue

        if np.isclose(lb[i] - ub[i], 0, atol=1e-12):
            # Equality constraint: lb == ub, so c(x) = lb
            constraint_nlf1 = _create_constraint_nlf1(
                scipy_constraint.fun, scipy_constraint.jac, x0, i, is_negated=False
            )
            nlp_wrapper = pyoptpp.NLP.create(constraint_nlf1)
            rhs = pyoptpp.SerialDenseVector(np.array(lb[i]))
            constraint = pyoptpp.NonLinearEquation.create(nlp_wrapper, rhs, 1)
            optpp_constraints.append(constraint)
            continue

        if np.isfinite(lb[i]):
            constraint_nlf1 = _create_constraint_nlf1(
                scipy_constraint.fun, scipy_constraint.jac, x0, i, is_negated=False
            )
            nlp_wrapper = pyoptpp.NLP.create(constraint_nlf1)
            rhs = pyoptpp.SerialDenseVector(np.array(lb[i]))
            constraint = pyoptpp.NonLinearInequality.create(nlp_wrapper, rhs, 1)
            optpp_constraints.append(constraint)

        if np.isfinite(ub[i]):
            constraint_nlf1 = _create_constraint_nlf1(
                scipy_constraint.fun, scipy_constraint.jac, x0, i, is_negated=True
            )
            nlp_wrapper = pyoptpp.NLP.create(constraint_nlf1)
            rhs = pyoptpp.SerialDenseVector(np.array(-ub[i]))
            constraint = pyoptpp.NonLinearInequality.create(nlp_wrapper, rhs, 1)
            optpp_constraints.append(constraint)

    return optpp_constraints


def convert_linear_constraint(scipy_constraint: LinearConstraint):
    """
    Convert a scipy.optimize.LinearConstraint to OPTPP LinearEquation/LinearInequality objects.

    Parameters:
    -----------
    scipy_constraint : scipy.optimize.LinearConstraint
        The scipy constraint to convert

    Returns:
    --------
    list
        List of OPTPP constraint objects (LinearEquation and/or LinearInequality)
    """
    optpp_constraints = []

    # Get constraint matrix and bounds
    A = np.asarray(scipy_constraint.A, dtype=float)
    A = np.atleast_2d(A)

    num_constraints = A.shape[0]
    for i in range(num_constraints):
        A_row = A[i : i + 1, :]  # Keep as 2D for consistency
        lb = scipy_constraint.lb[i]
        ub = scipy_constraint.ub[i]

        if not np.isfinite(lb) and not np.isfinite(ub):
            # Both bounds are infinite - this is not a real constraint
            continue

        if np.isclose(lb - ub, 0, atol=1e-12):
            # Equality constraint: lb == ub
            A_matrix = pyoptpp.SerialDenseMatrix(A_row)
            rhs = pyoptpp.SerialDenseVector(np.array([lb]))
            constraint = pyoptpp.LinearEquation.create(A_matrix, rhs)
            optpp_constraints.append(constraint)
            continue

        # Lower bound: Ax >= lb  =>  Ax >= lb (OPTPP standard form)
        if np.isfinite(lb):
            A_matrix = pyoptpp.SerialDenseMatrix(A_row)
            rhs = pyoptpp.SerialDenseVector(np.array([lb]))
            constraint = pyoptpp.LinearInequality.create(A_matrix, rhs)
            optpp_constraints.append(constraint)

        # Upper bound: Ax <= ub  =>  -Ax >= -ub (OPTPP standard form)
        if np.isfinite(ub):
            A_neg_matrix = pyoptpp.SerialDenseMatrix(-A_row)
            rhs = pyoptpp.SerialDenseVector(np.array([-ub]))
            constraint = pyoptpp.LinearInequality.create(A_neg_matrix, rhs)
            optpp_constraints.append(constraint)

    return optpp_constraints
