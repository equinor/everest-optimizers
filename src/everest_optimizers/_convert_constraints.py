import numpy as np
import numpy.typing as npt
from scipy.optimize import LinearConstraint, NonlinearConstraint

from everest_optimizers import pyoptpp


def _convert_nonlinear_constraint(
    scipy_constraint: NonlinearConstraint, x0: npt.NDArray[np.float64]
):
    """
    Convert a scipy.optimize.NonlinearConstraint to OPTPP NonLinearEquation/NonLinearInequality objects.

    Following the OPTPP pattern from hockfcns.C examples:
    1. Create NLF1 objects with proper C++ constraint function signatures
    2. Wrap in NLP handles
    3. Create NonLinearEquation/NonLinearInequality from NLP handles

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

    # Create constraint NLF1 objects following OPTPP patterns from hockfcns.C
    # Pattern: NLP* constraint_nlp = new NLP(new NLF1(n, 1, constraint_func, init_func))
    def create_constraint_nlf1(
        constraint_func, constraint_jac, x0_ref, constraint_index, is_negated=False
    ):
        """Create an NLF1 object for a specific constraint following OPTPP hockfcns.C patterns."""

        class ConstraintNLF1(pyoptpp.NLF1):
            def __init__(self, n_vars):
                # Follow OPTPP NLF1 constructor pattern from hockfcns.C
                super().__init__(n_vars)
                self.n_vars = n_vars
                self.constraint_func = constraint_func
                self.constraint_jac = constraint_jac
                self.constraint_index = constraint_index
                self.x0_ref = x0_ref
                self.is_negated = is_negated

                # Set initial point following OPTPP pattern
                init_vector = pyoptpp.SerialDenseVector(x0_ref)
                self.setX(init_vector)
                self.setIsExpensive(True)

            def evalF(self, x):
                """Main objective function - not used in constraint NLF1 (similar to hockfcns.C)."""
                return 0.0

            def evalG(self, x):
                """Main objective gradient - not used in constraint NLF1 (similar to hockfcns.C)."""
                return np.zeros(self.n_vars)

            def evalCF(self, x):
                """Evaluate constraint function(s) - OPTPP calls this for constraints (like eqn_hs6 in hockfcns.C)."""
                x_np = np.array(x.to_numpy(), copy=True)
                try:
                    # Evaluate the scipy constraint function
                    c_values = self.constraint_func(x_np)
                    c_values = np.atleast_1d(c_values)

                    # For single constraint, return only the relevant constraint
                    if self.constraint_index < len(c_values):
                        result = c_values[self.constraint_index]
                    else:
                        result = c_values[0]  # fallback

                    # Apply negation if needed (for upper bound inequalities)
                    if self.is_negated:
                        result = -result

                    # Return as numpy array - let pybind11 handle the conversion
                    return np.array([result])
                except Exception as e:
                    raise RuntimeError(
                        f"Error evaluating nonlinear constraint: {e}"
                    ) from e

            def evalCG(self, x):
                """Evaluate constraint gradient (like eqn_hs6 gradient in hockfcns.C)."""
                x_np = np.array(x.to_numpy(), copy=True)
                try:
                    if self.constraint_jac is not None:
                        # Use provided Jacobian
                        jac = self.constraint_jac(x_np)
                        jac = np.atleast_2d(jac)

                        # For single constraint, get the relevant row
                        if self.constraint_index < jac.shape[0]:
                            grad_row = jac[self.constraint_index, :]
                        else:
                            grad_row = jac[0, :]  # fallback
                    else:
                        # Use finite differences
                        grad_row = self._finite_difference_constraint_gradient(x_np)

                    # Apply negation if needed
                    if self.is_negated:
                        grad_row = -grad_row

                    # Return as 2D numpy array - let pybind11 handle the conversion
                    return grad_row.reshape(self.n_vars, 1)

                except Exception as e:
                    raise RuntimeError(
                        f"Error evaluating nonlinear constraint gradient: {e}"
                    ) from e

            def _finite_difference_constraint_gradient(self, x):
                """Compute constraint gradient using finite differences."""
                eps = 1e-8
                n = len(x)
                grad = np.zeros(n)

                # Evaluate at base point
                c0 = self.constraint_func(x)
                c0 = np.atleast_1d(c0)

                # Get the relevant constraint value
                if self.constraint_index < len(c0):
                    c0_val = c0[self.constraint_index]
                else:
                    c0_val = c0[0]

                # Finite difference for each variable
                for i in range(n):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    x_minus = x.copy()
                    x_minus[i] -= eps

                    c_plus = np.atleast_1d(self.constraint_func(x_plus))
                    c_minus = np.atleast_1d(self.constraint_func(x_minus))

                    # Get relevant constraint values
                    if self.constraint_index < len(c_plus):
                        c_plus_val = c_plus[self.constraint_index]
                        c_minus_val = c_minus[self.constraint_index]
                    else:
                        c_plus_val = c_plus[0]
                        c_minus_val = c_minus[0]

                    grad[i] = (c_plus_val - c_minus_val) / (2 * eps)

                return grad

        return ConstraintNLF1(len(x0_ref))

    # Process each constraint
    for i in range(num_constraints):
        lb_i = lb[i] if i < len(lb) else lb[-1]
        ub_i = ub[i] if i < len(ub) else ub[-1]

        # Determine constraint type based on bounds
        if np.isfinite(lb_i) and np.isfinite(ub_i):
            if np.abs(lb_i - ub_i) < 1e-12:
                # Equality constraint: lb == ub, so c(x) = lb
                # Follow OPTPP pattern from hockfcns.C: new NLP(new NLF1(...))
                constraint_nlf1 = create_constraint_nlf1(
                    scipy_constraint.fun, scipy_constraint.jac, x0, i, is_negated=False
                )
                nlp_wrapper = pyoptpp.NLP(constraint_nlf1)

                # For OPTPP, equality constraint is c(x) - rhs = 0
                rhs = pyoptpp.SerialDenseVector(np.array([lb_i]))
                eq_constraint = pyoptpp.NonLinearEquation(nlp_wrapper, rhs, 1)
                optpp_constraints.append(eq_constraint)
            else:
                # Double-sided inequality: lb <= c(x) <= ub
                # Split into two constraints following OPTPP standard form (Ax >= b)

                if np.isfinite(lb_i):
                    # c(x) >= lb  =>  c(x) - lb >= 0
                    # Follow OPTPP pattern from hockfcns.C: new NLP(new NLF1(...))
                    constraint_nlf1 = create_constraint_nlf1(
                        scipy_constraint.fun,
                        scipy_constraint.jac,
                        x0,
                        i,
                        is_negated=False,
                    )
                    nlp_wrapper = pyoptpp.NLP(constraint_nlf1)
                    rhs_lower = pyoptpp.SerialDenseVector(np.array([lb_i]))
                    ineq_lower = pyoptpp.NonLinearInequality(nlp_wrapper, rhs_lower, 1)
                    optpp_constraints.append(ineq_lower)

                if np.isfinite(ub_i):
                    # c(x) <= ub  =>  -c(x) + ub >= 0 (OPTPP standard form)
                    # Follow OPTPP pattern from hockfcns.C: new NLP(new NLF1(...))
                    constraint_nlf1 = create_constraint_nlf1(
                        scipy_constraint.fun,
                        scipy_constraint.jac,
                        x0,
                        i,
                        is_negated=True,
                    )
                    nlp_wrapper = pyoptpp.NLP(constraint_nlf1)
                    rhs_upper = pyoptpp.SerialDenseVector(np.array([-ub_i]))
                    ineq_upper = pyoptpp.NonLinearInequality(nlp_wrapper, rhs_upper, 1)
                    optpp_constraints.append(ineq_upper)

        elif np.isfinite(lb_i) and not np.isfinite(ub_i):
            # One-sided inequality: c(x) >= lb
            # Follow OPTPP pattern from hockfcns.C: new NLP(new NLF1(...))
            constraint_nlf1 = create_constraint_nlf1(
                scipy_constraint.fun, scipy_constraint.jac, x0, i, is_negated=False
            )
            nlp_wrapper = pyoptpp.NLP(constraint_nlf1)
            rhs = pyoptpp.SerialDenseVector(np.array([lb_i]))
            ineq_constraint = pyoptpp.NonLinearInequality(nlp_wrapper, rhs, 1)
            optpp_constraints.append(ineq_constraint)

        elif not np.isfinite(lb_i) and np.isfinite(ub_i):
            # One-sided inequality: c(x) <= ub  =>  -c(x) + ub >= 0 (OPTPP standard form)
            # Follow OPTPP pattern from hockfcns.C: new NLP(new NLF1(...))
            constraint_nlf1 = create_constraint_nlf1(
                scipy_constraint.fun, scipy_constraint.jac, x0, i, is_negated=True
            )
            nlp_wrapper = pyoptpp.NLP(constraint_nlf1)
            rhs = pyoptpp.SerialDenseVector(np.array([-ub_i]))
            ineq_constraint = pyoptpp.NonLinearInequality(nlp_wrapper, rhs, 1)
            optpp_constraints.append(ineq_constraint)

        else:
            # Both bounds are infinite - this is not a real constraint
            continue

    return optpp_constraints


def _convert_linear_constraint(scipy_constraint: LinearConstraint):
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
            constraint = pyoptpp.LinearEquation(A_matrix, rhs)
            optpp_constraints.append(constraint)
            continue

        if np.isfinite(lb):
            A_matrix = pyoptpp.SerialDenseMatrix(A_row)
            rhs = pyoptpp.SerialDenseVector(np.array([lb]))
            constraint = pyoptpp.LinearInequality(A_matrix, rhs)
            optpp_constraints.append(constraint)

        if np.isfinite(ub):
            A_neg_matrix = pyoptpp.SerialDenseMatrix(-A_row)
            rhs = pyoptpp.SerialDenseVector(np.array([-ub]))
            constraint = pyoptpp.LinearInequality(A_neg_matrix, rhs)
            optpp_constraints.append(constraint)

    return optpp_constraints
