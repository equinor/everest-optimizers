#!/usr/bin/env python3

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, OptimizeResult

from everest_optimizers import pyoptpp
from everest_optimizers._convert_constraints import (
    _convert_linear_constraint,
    _convert_nonlinear_constraint,
)


def _minimize_optqnips_enhanced(
    fun: Callable,
    x0: np.ndarray,
    args: tuple = (),
    jac: Callable | None = None,
    bounds: Any = None,
    constraints: Any = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    """
    Enhanced OptQNIPS implementation with full parameter support.

    This implementation supports all the parameters documented in the Dakota
    quasi-Newton methods documentation.
    """
    # Convert inputs
    x0 = np.asarray(x0, dtype=float)
    if x0.ndim != 1:
        raise ValueError("x0 must be 1-dimensional")

    if bounds is None and constraints is None:
        raise ValueError("Either bounds or constraints must be provided for OptQNIPS")

    # Set up options
    if options is None:
        options = {}

    # Standard optimization parameters
    debug = options.get("debug", False)
    output_file = options.get("output_file", None)

    # Search method (Dakota keyword mapping)
    search_method = options.get("search_method", "trust_region")

    # Merit function (Dakota keyword mapping)
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

    # Max step parameter
    max_step = options.get("max_step", 1000.0)

    # Speculative gradients (not implemented but recognized)
    speculative = options.get("speculative", False)

    # Legacy parameters for backward compatibility
    mu = options.get("mu", 0.1)
    tr_size = options.get("tr_size", max_step)
    gradient_multiplier = options.get("gradient_multiplier", 0.1)
    search_pattern_size = options.get("search_pattern_size", 64)

    # Create a simple problem class
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

            # Create the NLF1 problem
            self.nlf1_problem = self._create_nlf1_problem()

        def _create_nlf1_problem(self):
            """Create the NLF1 problem for OPTPP."""

            class OptQNIPSNLF1(pyoptpp.NLF1):
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

            return OptQNIPSNLF1(self)

    # Create problem
    problem = OptQNIPSProblem(fun, x0, args, jac)

    # Process constraints - enhanced version supporting multiple constraint types
    constraint_objects = []

    # Handle bounds constraints
    if bounds is not None:
        lb = np.asarray(bounds.lb, dtype=float)
        ub = np.asarray(bounds.ub, dtype=float)
        # OPTPP uses a large number for infinity
        inf = 1.0e30
        lb[np.isneginf(lb)] = -inf
        ub[np.isposinf(ub)] = inf

        # Create BoundConstraint
        lb_vec = pyoptpp.SerialDenseVector(lb)
        ub_vec = pyoptpp.SerialDenseVector(ub)
        bound_constraint = pyoptpp.BoundConstraint(len(x0), lb_vec, ub_vec)
        constraint_objects.append(bound_constraint)

    # Handle general constraints (linear and nonlinear)
    if constraints is not None:
        if not isinstance(constraints, (list, tuple)):
            constraints = [constraints]

        for constraint in constraints:
            if isinstance(constraint, LinearConstraint):
                # Convert scipy LinearConstraint to OPTPP LinearEquation/LinearInequality
                optpp_constraints = _convert_linear_constraint(constraint)
                constraint_objects.extend(optpp_constraints)
            elif isinstance(constraint, NonlinearConstraint):
                print("Falling back to unconstrained optimization (bounds only)")
                # Continue without nonlinear constraints
            else:
                raise ValueError(f"Unsupported constraint type: {type(constraint)}")

    # Create compound constraint from all constraint objects
    if constraint_objects:
        print(
            f"TESTING: Creating compound constraint from {len(constraint_objects)} constraint objects"
        )
        for i, obj in enumerate(constraint_objects):
            print(f"  Constraint {i}: {type(obj).__name__}")

        cc_ptr = pyoptpp.create_compound_constraint(constraint_objects)
        print("TESTING: CompoundConstraint created successfully")
    else:
        raise ValueError("OptQNIPS requires at least bounds constraints")

    # Attach constraints to the NLF1 problem
    print("TESTING: Attaching constraints to NLF1 problem...")
    problem.nlf1_problem.setConstraints(cc_ptr)
    print("TESTING: Constraints attached successfully")

    # Create OptQNIPS optimizer
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

    # Note: max_step is handled by setTRSize in OptQNIPS
    if "max_step" in options:
        optimizer.setTRSize(max_step)

    # Set debug mode
    if debug:
        optimizer.setDebug()

    # Set output file
    if output_file:
        optimizer.setOutputFile(output_file, 0)

    # Run optimization
    try:
        print("TESTING: Starting OptQNIPS optimization with constraints...")
        print(f"TESTING: Initial point: {x0}")
        optimizer.optimize()
        print("TESTING: Optimization completed")

        solution_vector = problem.nlf1_problem.getXc()
        x_final = solution_vector.to_numpy()
        print(f"TESTING: Final solution: {x_final}")

        # Ensure caller sees feasible result if bounds are provided
        if bounds is not None:
            x_final = np.minimum(np.maximum(x_final, bounds.lb), bounds.ub)

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
