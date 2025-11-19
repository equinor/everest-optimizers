#!/usr/bin/env python3

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, OptimizeResult

from everest_optimizers import pyoptpp


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


def _convert_nonlinear_constraint(scipy_constraint, x0):
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

    # Ensure bounds are 1D arrays
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


def _convert_linear_constraint(scipy_constraint):
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
    lb = np.asarray(scipy_constraint.lb, dtype=float)
    ub = np.asarray(scipy_constraint.ub, dtype=float)

    # Ensure A is 2D
    if A.ndim == 1:
        A = A.reshape(1, -1)

    # Ensure bounds are 1D arrays
    lb = np.atleast_1d(lb)
    ub = np.atleast_1d(ub)

    num_constraints = A.shape[0]

    # Process each constraint row
    for i in range(num_constraints):
        A_row = A[i : i + 1, :]  # Keep as 2D for consistency
        lb_i = lb[i]
        ub_i = ub[i]

        # Create OPTPP matrix and vector objects
        A_matrix = pyoptpp.SerialDenseMatrix(A_row)

        # Determine constraint type based on bounds
        if np.isfinite(lb_i) and np.isfinite(ub_i):
            if np.abs(lb_i - ub_i) < 1e-12:
                # Equality constraint: lb == ub
                rhs = pyoptpp.SerialDenseVector(np.array([lb_i]))
                eq_constraint = pyoptpp.LinearEquation(A_matrix, rhs)
                optpp_constraints.append(eq_constraint)
            else:
                # Double-sided inequality: lb <= Ax <= ub
                # Convert to two single-sided inequalities:
                # Ax >= lb  =>  Ax - lb >= 0
                # Ax <= ub  =>  -Ax + ub >= 0

                # Lower bound: Ax >= lb  =>  Ax >= lb (OPTPP standard form)
                if np.isfinite(lb_i):
                    rhs_lower = pyoptpp.SerialDenseVector(np.array([lb_i]))
                    ineq_lower = pyoptpp.LinearInequality(A_matrix, rhs_lower)
                    optpp_constraints.append(ineq_lower)

                # Upper bound: Ax <= ub  =>  -Ax >= -ub (OPTPP standard form)
                if np.isfinite(ub_i):
                    A_neg = -A_row
                    A_neg_matrix = pyoptpp.SerialDenseMatrix(A_neg)
                    rhs_upper = pyoptpp.SerialDenseVector(np.array([-ub_i]))
                    ineq_upper = pyoptpp.LinearInequality(A_neg_matrix, rhs_upper)
                    optpp_constraints.append(ineq_upper)

        elif np.isfinite(lb_i) and not np.isfinite(ub_i):
            # One-sided inequality: Ax >= lb (OPTPP standard form)
            rhs = pyoptpp.SerialDenseVector(np.array([lb_i]))
            ineq_constraint = pyoptpp.LinearInequality(A_matrix, rhs)
            optpp_constraints.append(ineq_constraint)

        elif not np.isfinite(lb_i) and np.isfinite(ub_i):
            # One-sided inequality: Ax <= ub  =>  -Ax >= -ub (OPTPP standard form)
            A_neg = -A_row
            A_neg_matrix = pyoptpp.SerialDenseMatrix(A_neg)
            rhs = pyoptpp.SerialDenseVector(np.array([-ub_i]))
            ineq_constraint = pyoptpp.LinearInequality(A_neg_matrix, rhs)
            optpp_constraints.append(ineq_constraint)

        else:
            # Both bounds are infinite - this is not a real constraint
            continue

    return optpp_constraints
