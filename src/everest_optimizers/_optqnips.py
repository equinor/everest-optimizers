from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, OptimizeResult

from everest_optimizers import pyoptpp
from everest_optimizers._convert_constraints import (
    convert_bound_constraint,
    convert_linear_constraint,
    convert_nonlinear_constraint,
)


def minimize_optqnips(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    jac: Callable[..., npt.NDArray[np.float64]] | None = None,
    bounds: Bounds | None = None,
    constraints: list[LinearConstraint | NonlinearConstraint] | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """
    This implementation supports all the parameters documented in the Dakota
    quasi-Newton methods documentation.

    options only supports max_step
    """
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")

    if bounds is None and constraints is None:
        raise ValueError("Either bounds or constraints must be provided for OptQNIPS")

    if options is None:
        options = {}

    # Standard optimization parameters
    debug = options.get("debug", False)
    output_file = options.get("output_file", None)

    search_method = options.get("search_method", "trust_region")
    merit_function = options.get("merit_function", "argaez_tapia")

    # Interior-point specific parameters with Dakota defaults based on merit function
    match merit_function.lower():
        case "el_bakry":
            default_centering = 0.2
            default_step_to_boundary = 0.8
        case "argaez_tapia":
            default_centering = 0.2
            default_step_to_boundary = 0.99995
        case "van_shanno":
            default_centering = 0.1
            default_step_to_boundary = 0.95
        case _:
            default_centering = 0.2
            default_step_to_boundary = 0.95

    centering_parameter = options.get("centering_parameter", default_centering)
    steplength_to_boundary = options.get(
        "steplength_to_boundary", default_step_to_boundary
    )

    # Standard optimization control parameters
    max_iterations = options.get("max_iterations", 100)
    max_function_evaluations = options.get("max_function_evaluations", 1000)
    convergence_tolerance = options.get("convergence_tolerance", 1e-4)
    gradient_tolerance = options.get("gradient_tolerance", 1e-4)
    constraint_tolerance = options.get("constraint_tolerance", 1e-6)
    max_step = options.get("max_step", 1000.0)

    # Speculative gradients (not implemented but recognized)
    speculative = options.get("speculative", False)

    # Legacy parameters for backward compatibility
    mu = options.get("mu", 0.1)
    tr_size = options.get("tr_size", max_step)
    gradient_multiplier = options.get("gradient_multiplier", 0.1)
    search_pattern_size = options.get("search_pattern_size", 64)

    class OptQNIPSProblem:
        def __init__(self, fun, x0, args, jac):
            self.fun = fun
            self.x0 = np.asarray(x0, dtype=float)
            self.args = args
            self.jac = jac

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
                """Evaluate objective function - called by C++."""
                x_np = np.array(x.to_numpy(), copy=True)
                self.current_x = x_np

                try:
                    f_val = self.fun(x_np, *self.args)
                    self.current_f = float(f_val)
                    self.nfev += 1
                    return self.current_f
                except Exception as e:
                    raise RuntimeError(
                        f"Error evaluating objective function: {e}"
                    ) from e

            def eval_g(x):
                """Evaluate gradient - called by C++."""
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
                    grad = self._finite_difference_gradient(x_np)
                    self.current_g = grad
                    return grad

            # Use C++ factory to create NLF1 - fully C++-managed, no ownership conflicts
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

    constraint_objects = []

    if bounds is not None:
        constraint_objects.append(convert_bound_constraint(bounds, len(x0)))

    if constraints is not None:
        for constraint in constraints:
            if isinstance(constraint, LinearConstraint):
                optpp_constraints = convert_linear_constraint(constraint)
                constraint_objects.extend(optpp_constraints)
            elif isinstance(constraint, NonlinearConstraint):
                optpp_constraints = convert_nonlinear_constraint(constraint, x0)
                constraint_objects.extend(optpp_constraints)
            else:
                raise ValueError(f"Unsupported constraint type: {type(constraint)}")

    if constraint_objects:
        cc_ptr = pyoptpp.create_compound_constraint(constraint_objects)
    else:
        raise ValueError("OptQNIPS requires at least bounds constraints")

    problem = OptQNIPSProblem(fun, x0, args, jac)
    problem.nlf1_problem.setConstraints(cc_ptr)
    optimizer = pyoptpp.OptQNIPS(problem.nlf1_problem)

    match search_method.lower():
        case "trust_region" | "trustregion":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustRegion)
        case "line_search" | "linesearch":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.LineSearch)
        case "trust_pds" | "trustpds":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustPDS)
        case _:
            raise ValueError(
                f"Unknown search method: {search_method}. Valid options: trust_region, line_search, trust_pds"
            )

    # Set trust region parameters
    optimizer.setTRSize(tr_size)
    optimizer.setGradMult(gradient_multiplier)
    optimizer.setSearchSize(search_pattern_size)

    # Set OptQNIPS-specific parameters
    optimizer.setMu(mu)
    optimizer.setCenteringParameter(centering_parameter)
    optimizer.setStepLengthToBdry(steplength_to_boundary)

    match merit_function.lower():
        case "el_bakry":
            optimizer.setMeritFcn(pyoptpp.MeritFcn.NormFmu)
        case "argaez_tapia":
            optimizer.setMeritFcn(pyoptpp.MeritFcn.ArgaezTapia)
        case "van_shanno":
            optimizer.setMeritFcn(pyoptpp.MeritFcn.VanShanno)
        case "norm_fmu":
            optimizer.setMeritFcn(pyoptpp.MeritFcn.NormFmu)
        case merit_fn:
            raise ValueError(
                f"Unknown merit function: {merit_fn}. Valid options: el_bakry, argaez_tapia, van_shanno"
            )

    # Set optimization control parameters
    optimizer.setMaxIter(max_iterations)
    optimizer.setMaxFeval(max_function_evaluations)
    optimizer.setFcnTol(convergence_tolerance)
    optimizer.setGradTol(gradient_tolerance)
    optimizer.setConTol(constraint_tolerance)

    if "max_step" in options:
        optimizer.setTRSize(max_step)
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
            nit=0,  # OptQNIPS doesn't provide iteration count directly
            success=True,
            status=0,
            message="OptQNIPS optimization terminated successfully",
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
            message=f"OptQNIPS optimization failed: {e!s}",
            jac=None,
        )
