#!/usr/bin/env python3

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult

from everest_optimizers import pyoptpp


class _OptQNewtonProblem:
    """Problem definition for OptQNewton optimizer."""

    def __init__(
        self,
        fun: Callable,
        x0: np.ndarray,
        args: tuple,
        jac: Callable | None = None,
        pyopttpp_module=None,
    ):
        self.fun = fun
        self.x0 = np.asarray(x0, dtype=float)
        self.args = args
        self.jac = jac

        self.nfev = 0
        self.njev = 0
        self.current_x = None
        self.current_f = None
        self.current_g = None

        # Create the NLF1 problem
        self.nlf1_problem = self._create_nlf1_problem()

    def _create_nlf1_problem(self):
        """Create the NLF1 problem for OPTPP."""

        class OptQNewtonNLF1(pyoptpp.NLF1):
            def __init__(self, parent_problem):
                super().__init__(len(parent_problem.x0))
                self.parent = parent_problem

                # Set initial point
                init_vector = pyoptpp.SerialDenseVector(parent_problem.x0)
                self.setX(init_vector)
                self.setIsExpensive(True)

            def evalF(self, x):
                """Evaluate objective function."""
                x_np = np.array(x.to_numpy(), copy=True)
                self.parent.current_x = x_np

                try:
                    f_val = self.parent.fun(x_np, *self.parent.args)
                    self.parent.current_f = float(f_val)
                    self.parent.nfev += 1
                    return self.parent.current_f
                except Exception as e:
                    raise RuntimeError(
                        f"Error evaluating objective function: {e}"
                    ) from e

            def evalG(self, x):
                """Evaluate gradient."""
                x_np = np.array(x.to_numpy(), copy=True)

                if self.parent.jac is not None:
                    try:
                        grad = self.parent.jac(x_np, *self.parent.args)
                        grad_np = np.asarray(grad, dtype=float)
                        self.parent.current_g = grad_np
                        self.parent.njev += 1
                        return grad_np
                    except Exception as e:
                        raise RuntimeError(f"Error evaluating gradient: {e}") from e
                else:
                    # Use finite differences for gradient
                    grad = self._finite_difference_gradient(x_np)
                    self.parent.current_g = grad
                    return grad

            def _finite_difference_gradient(self, x):
                """Compute gradient using finite differences."""
                eps = 1e-8
                grad = np.zeros_like(x)

                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    x_minus = x.copy()
                    x_minus[i] -= eps

                    f_plus = self.parent.fun(x_plus, *self.parent.args)
                    f_minus = self.parent.fun(x_minus, *self.parent.args)

                    grad[i] = (f_plus - f_minus) / (2 * eps)
                    self.parent.nfev += 2

                return grad

        return OptQNewtonNLF1(self)


def _minimize_optqnewton(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    jac: Callable | None = None,
    bounds: Any | None = None,
    constraints: Any | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """
    Minimize a scalar function using optpp_q_newton optimizer.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : ndarray
        Initial guess.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives.
    jac : callable, optional
        Method for computing the gradient vector.
    bounds : sequence, optional
        Bounds on variables (not supported by optpp_q_newton).
    constraints : dict or list, optional
        Constraints definition (not supported by optpp_q_newton).
    options : dict, optional
        Solver options including:
        - 'search_strategy': 'TrustRegion', 'LineSearch', or 'TrustPDS'
        - 'tr_size': Trust region size
        - 'debug': Enable debug output
        - 'output_file': Output file for debugging

    Returns
    -------
    OptimizeResult
        The optimization result.
    """
    # Convert inputs
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")

    if bounds is not None:
        raise NotImplementedError("optpp_q_newton does not support bounds")

    if constraints is not None:
        raise NotImplementedError("optpp_q_newton does not support constraints")

    # Set up options
    if options is None:
        options = {}

    search_strategy = options.get("search_strategy", "TrustRegion")
    tr_size = options.get("tr_size", 100.0)
    debug = options.get("debug", False)
    output_file = options.get("output_file", None)

    # Create problem
    problem = _OptQNewtonProblem(fun, x0, args, jac)

    # Create optimizer
    optimizer = pyoptpp.OptQNewton(problem.nlf1_problem)

    # Set search strategy
    if search_strategy == "TrustRegion":
        optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustRegion)
    elif search_strategy == "LineSearch":
        optimizer.setSearchStrategy(pyoptpp.SearchStrategy.LineSearch)
    elif search_strategy == "TrustPDS":
        optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustPDS)
    else:
        raise ValueError(f"Unknown search strategy: {search_strategy}")

    # Set trust region size
    optimizer.setTRSize(tr_size)

    # Set debug mode
    if debug:
        optimizer.setDebug()

    # Set output file
    if output_file:
        optimizer.setOutputFile(output_file, 0)

    # Run optimization
    try:
        optimizer.optimize()

        # Get results
        solution_vector = problem.nlf1_problem.getXc()
        x_final = solution_vector.to_numpy()
        f_final = problem.nlf1_problem.getF()

        # Create result
        result = OptimizeResult(
            x=x_final,
            fun=f_final,
            nfev=problem.nfev,
            njev=problem.njev,
            nit=0,  # optpp_q_newton doesn't provide iteration count
            success=True,
            status=0,
            message="Optimization terminated successfully",
            jac=problem.current_g if problem.current_g is not None else None,
        )

        optimizer.cleanup()
        return result

    except Exception as e:
        optimizer.cleanup()
        return OptimizeResult(
            x=x0,
            fun=None,
            nfev=problem.nfev,
            njev=problem.njev,
            nit=0,
            success=False,
            status=1,
            message=f"Optimization failed: {e!s}",
            jac=None,
        )


def _minimize_optconstrqnewton(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    jac: Callable | None = None,
    bounds: Any = None,
    constraints: Any = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """
    Minimize a scalar function with constraints using OptConstrQNewton.
    """
    # Convert inputs
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")

    if bounds is None and constraints is None:
        raise ValueError(
            "Either bounds or constraints must be provided for constrained optimization"
        )

    # Set up options
    if options is None:
        options = {}

    search_strategy = options.get("search_strategy", "TrustRegion")
    tr_size = options.get("tr_size", 100.0)
    debug = options.get("debug", False)
    output_file = options.get("output_file", None)

    # The problem definition doesn't need to know about constraints, as they are handled by the C++ part.
    problem = _OptQNewtonProblem(fun, x0, args, jac)

    # Process constraints
    constraint_list = []
    if bounds is not None:
        lb = np.asarray(bounds.lb, dtype=float)
        ub = np.asarray(bounds.ub, dtype=float)
        # OPTPP uses a large number for infinity
        inf = 1.0e30
        lb[np.isneginf(lb)] = -inf
        ub[np.isposinf(ub)] = inf
        bound_constraint = pyoptpp.BoundConstraint(
            len(x0), pyoptpp.SerialDenseVector(lb), pyoptpp.SerialDenseVector(ub)
        )
        constraint_list.append(bound_constraint)

    if constraints is not None:
        # Handle various constraint types
        if hasattr(constraints, "__iter__") and not isinstance(constraints, dict):
            # List of constraints
            for constraint in constraints:
                if (
                    hasattr(constraint, "A")
                    and hasattr(constraint, "lb")
                    and hasattr(constraint, "ub")
                ):
                    # LinearConstraint from scipy
                    A_matrix = pyoptpp.SerialDenseMatrix(constraint.A)
                    if np.allclose(constraint.lb, constraint.ub):
                        # Equality constraint: lb == ub
                        linear_eq = pyoptpp.LinearEquation(
                            A_matrix, pyoptpp.SerialDenseVector(constraint.lb)
                        )
                        constraint_list.append(linear_eq)
                    else:
                        # Inequality constraint: lb <= Ax <= ub
                        # For now, only handle Ax >= lb case (lb finite, ub infinite)
                        if (
                            np.isfinite(constraint.lb).all()
                            and np.isinf(constraint.ub).all()
                        ):
                            # Convert Ax >= lb to Ax - lb >= 0
                            linear_ineq = pyoptpp.LinearInequality(
                                A_matrix, pyoptpp.SerialDenseVector(constraint.lb)
                            )
                            constraint_list.append(linear_ineq)
                        else:
                            raise NotImplementedError(
                                "Only linear equality constraints (lb == ub) and one-sided inequalities "
                                "(Ax >= lb with infinite upper bounds) are currently supported."
                            )
                else:
                    raise ValueError(f"Unknown constraint type: {type(constraint)}")
        else:
            # Single constraint
            if (
                hasattr(constraints, "A")
                and hasattr(constraints, "lb")
                and hasattr(constraints, "ub")
            ):
                # LinearConstraint from scipy
                A_matrix = pyoptpp.SerialDenseMatrix(constraints.A)
                if np.allclose(constraints.lb, constraints.ub):
                    # Equality constraint: lb == ub
                    linear_eq = pyoptpp.LinearEquation(
                        A_matrix, pyoptpp.SerialDenseVector(constraints.lb)
                    )
                    constraint_list.append(linear_eq)
                else:
                    # Inequality constraint: lb <= Ax <= ub
                    # For now, only handle Ax >= lb case (lb finite, ub infinite)
                    if (
                        np.isfinite(constraints.lb).all()
                        and np.isinf(constraints.ub).all()
                    ):
                        # Convert Ax >= lb to Ax - lb >= 0
                        linear_ineq = pyoptpp.LinearInequality(
                            A_matrix, pyoptpp.SerialDenseVector(constraints.lb)
                        )
                        constraint_list.append(linear_ineq)
                    else:
                        raise NotImplementedError(
                            "Only linear equality constraints (lb == ub) and one-sided inequalities "
                            "(Ax >= lb with infinite upper bounds) are currently supported."
                        )
            else:
                raise ValueError(f"Unknown constraint type: {type(constraints)}")

    if not constraint_list:
        raise ValueError("No valid constraints were processed.")

    # Create C++ compound constraint object
    if (
        len(constraint_list) == 1
        and hasattr(constraint_list[0], "__class__")
        and "BoundConstraint" in str(constraint_list[0].__class__)
    ):
        # Use the bounds-only helper for backwards compatibility
        lb = np.asarray(bounds.lb, dtype=float)
        ub = np.asarray(bounds.ub, dtype=float)
        # OPTPP uses a large number for infinity
        inf = 1.0e30
        lb[np.isneginf(lb)] = -inf
        ub[np.isposinf(ub)] = inf
        cc_ptr = pyoptpp.create_compound_constraint(lb, ub)
    else:
        # For general constraints, use the constraint list approach
        try:
            # Wrap each constraint in a Constraint handle
            wrapped_constraints = []
            for constr in constraint_list:
                wrapped_constraint = pyoptpp.create_constraint(constr)
                wrapped_constraints.append(wrapped_constraint)
            cc_ptr = pyoptpp.create_compound_constraint(wrapped_constraints)
        except (AttributeError, TypeError) as e:
            # Fall back to bounds-only if LinearConstraint classes are not available
            if bounds is not None:
                lb = np.asarray(bounds.lb, dtype=float)
                ub = np.asarray(bounds.ub, dtype=float)
                inf = 1.0e30
                lb[np.isneginf(lb)] = -inf
                ub[np.isposinf(ub)] = inf
                cc_ptr = pyoptpp.create_compound_constraint(lb, ub)
            else:
                raise NotImplementedError(
                    "Linear constraints are not available in the current build. Only bounds constraints are supported."
                ) from e

    # Attach constraints to the NLF1 problem
    problem.nlf1_problem.setConstraints(cc_ptr)
    optimizer = pyoptpp.OptConstrQNewton(problem.nlf1_problem)

    # Set search strategy
    if search_strategy == "TrustRegion":
        optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustRegion)
    elif search_strategy == "LineSearch":
        optimizer.setSearchStrategy(pyoptpp.SearchStrategy.LineSearch)
    elif search_strategy == "TrustPDS":
        optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustPDS)
    else:
        raise ValueError(f"Unknown search strategy: {search_strategy}")

    # Set trust region size
    optimizer.setTRSize(tr_size)

    # Set debug mode
    if debug:
        optimizer.setDebug()

    # Set output file
    if output_file:
        optimizer.setOutputFile(output_file, 0)

    # Run optimization
    try:
        optimizer.optimize()
        solution_vector = problem.nlf1_problem.getXc()
        x_final = solution_vector.to_numpy()
        # Ensure caller sees feasible result if bounds are provided
        if bounds is not None:
            x_final = np.minimum(np.maximum(x_final, bounds.lb), bounds.ub)
        f_final = problem.nlf1_problem.getF()
        result = OptimizeResult(
            x=x_final,
            fun=f_final,
            nfev=problem.nfev,
            njev=problem.njev,
            nit=0,
            success=True,
            status=0,
            message="Optimization terminated successfully",
            jac=problem.current_g if problem.current_g is not None else None,
        )
        optimizer.cleanup()
        return result

    except Exception as e:
        optimizer.cleanup()
        return OptimizeResult(
            x=x0,
            fun=None,
            nfev=problem.nfev,
            njev=problem.njev,
            nit=0,
            success=False,
            status=1,
            message=f"Optimization failed: {e!s}",
            jac=None,
        )
