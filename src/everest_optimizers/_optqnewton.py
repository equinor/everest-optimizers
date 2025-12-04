import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult

from everest_optimizers import pyoptpp
from everest_optimizers._convert_constraints import (
    convert_bound_constraint,
    convert_linear_constraint,
)


class _OptQNewtonProblem:
    def __init__(
        self,
        fun: Callable,
        x0: np.ndarray,
        args: tuple,
        jac: Callable[..., npt.NDArray[np.float64]] | None = None,
        callback: Callable | None = None,
    ):
        self.fun = fun
        self.x0 = np.asarray(x0, dtype=float)
        self.args = args
        self.jac = jac
        self.callback = callback

        self.nfev = 0
        self.njev = 0
        self.current_x = None
        self.current_f = None
        self.current_g = None

        self.nlf1_problem = self._create_nlf1_problem()

    def _create_nlf1_problem(self):
        """Create the NLF1 problem for OPTPP using C++ CallbackNLF1."""

        # Create callback functions for objective evaluation
        def eval_f(x):
            x_np = np.array(x.to_numpy(), copy=True)
            self.current_x = x_np

            try:
                f_val = self.fun(x_np, *self.args)
                self.current_f = float(f_val)
                self.nfev += 1

                if self.callback is not None:
                    try:
                        self.callback(x_np)
                    except Exception as cb_err:
                        warnings.warn(
                            f"Callback function raised exception: {cb_err}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

                return self.current_f
            except Exception as e:
                raise RuntimeError(f"Error evaluating objective function: {e}") from e

        def eval_g(x):
            x_np = np.array(x.to_numpy(), copy=True)

            if self.jac is not None:
                try:
                    grad = self.jac(x_np, *self.args)
                    grad_np = np.asarray(grad, dtype=float)
                    self.current_g = grad_np
                    self.njev += 1
                    return grad_np
                except Exception as e:
                    raise RuntimeError(f"Error evaluating gradient: {e}") from e
            else:
                # Use finite differences for gradient if no jacobian is supplied
                grad = self._finite_difference_gradient(x_np)
                self.current_g = grad
                return grad

        x0_vector = pyoptpp.SerialDenseVector(self.x0)
        nlf1 = pyoptpp.NLF1.create(len(self.x0), eval_f, eval_g, x0_vector)
        return nlf1

    def _finite_difference_gradient(self, x):
        """Compute gradient using finite differences."""
        eps = 1e-8
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps

            f_plus = self.fun(x_plus, *self.args)
            f_minus = self.fun(x_minus, *self.args)

            grad[i] = (f_plus - f_minus) / (2 * eps)
            self.nfev += 2

        return grad


def minimize_optqnewton(
    fun: Callable,
    x0: npt.NDArray,
    args: tuple = (),
    jac: Callable[..., npt.NDArray[np.float64]] | None = None,
    bounds: Bounds | None = None,
    constraints: list[LinearConstraint | NonlinearConstraint] | None = None,
    callback: Any | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """
    Minimize a scalar function using optpp_q_newton optimizer.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    x0 : ndarray
        Initial guess. Must be 1d.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives.
    jac : callable, optional
        Method for computing the gradient vector.
    bounds : sequence, optional
        Bounds on variables (not supported by optpp_q_newton).
    constraints : list, optional
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
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")

    if bounds is not None:
        raise NotImplementedError("optpp_q_newton does not support bounds")

    if constraints is not None:
        raise NotImplementedError("optpp_q_newton does not support constraints")

    if options is None:
        options = {}

    search_strategy = options.get("search_strategy", "TrustRegion")
    tr_size = options.get("tr_size", 100.0)
    debug = options.get("debug", False)
    output_file = options.get("output_file", None)

    problem = _OptQNewtonProblem(fun, x0, args, jac, callback)
    optimizer = pyoptpp.OptQNewton(problem.nlf1_problem)

    match search_strategy:
        case "TrustRegion":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustRegion)
        case "LineSearch":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.LineSearch)
        case "TrustPDS":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustPDS)
        case other:
            raise ValueError(
                f"Unknown search strategy: {other}. Valid options: TrustRegion, LineSearch, TrustPDS"
            )

    optimizer.setTRSize(tr_size)
    if debug:
        optimizer.setDebug()
    if output_file:
        optimizer.setOutputFile(output_file, 0)

    try:
        optimizer.optimize()

        solution_vector = problem.nlf1_problem.getXc()
        x_final = solution_vector.to_numpy()
        f_final = problem.nlf1_problem.getF()

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


def minimize_optconstrqnewton(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    jac: Callable[..., npt.NDArray[np.float64]] | None = None,
    bounds: Bounds | None = None,
    constraints: list[LinearConstraint | NonlinearConstraint] | None = None,
    callback: Any | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """
    Minimize a scalar function with constraints using OptConstrQNewton.
    """
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")

    if bounds is None and constraints is None:
        raise ValueError(
            "Either bounds or constraints must be provided for constrained optimization"
        )

    if options is None:
        options = {}

    search_strategy = options.get("search_strategy", "TrustRegion")
    tr_size = options.get("tr_size", 100.0)
    debug = options.get("debug", False)
    output_file = options.get("output_file", None)

    problem = _OptQNewtonProblem(fun, x0, args, jac, callback)

    constraint_list = []
    if bounds is not None:
        constraint_list.append(convert_bound_constraint(bounds, len(x0)))

    if constraints is not None:
        for constraint in constraints:
            if np.isfinite(constraint.lb) + np.isfinite(constraint.ub) != 1:
                raise NotImplementedError(
                    "Only linear equality constraints (lb == ub) and one-sided inequalities "
                    "(Ax >= lb with infinite upper bounds) are currently supported."
                )
            optpp_constraint = convert_linear_constraint(constraint)
            constraint_list.extend(optpp_constraint)

    cc_ptr = pyoptpp.create_compound_constraint(constraint_list)

    problem.nlf1_problem.setConstraints(cc_ptr)
    optimizer = pyoptpp.OptConstrQNewton(problem.nlf1_problem)

    if search_strategy == "TrustRegion":
        optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustRegion)
    elif search_strategy == "LineSearch":
        optimizer.setSearchStrategy(pyoptpp.SearchStrategy.LineSearch)
    elif search_strategy == "TrustPDS":
        optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustPDS)
    else:
        raise ValueError(f"Unknown search strategy: {search_strategy}")

    optimizer.setTRSize(tr_size)
    if debug:
        optimizer.setDebug()
    if output_file:
        optimizer.setOutputFile(output_file, 0)

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
