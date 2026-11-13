"""Option handling and algorithm selection for the `optpp_q_newton` method."""

from collections.abc import Callable
from numbers import Integral, Real
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from ._evaluation import Evaluator
from ._optpp import (
    Bounds,
    LinearConstraints,
    run_bcq_newton,
    run_q_newton,
    run_q_nips,
)
from ._problem import NLF1Problem

# Bounds at or beyond this magnitude count as absent.
_BIG_BOUND: Final = 1.0e30

_INT_OPTIONS: Final = {"max_iterations", "max_function_evaluations"}
_FLOAT_OPTIONS: Final = {
    "convergence_tolerance",
    "gradient_tolerance",
    "max_step",
    "steplength_to_boundary",
    "centering_parameter",
}
_CHOICE_OPTIONS: Final = {
    "search_method": (
        "value_based_line_search",
        "gradient_based_line_search",
        "trust_region",
    ),
    "merit_function": ("el_bakry", "argaez_tapia", "van_shanno"),
    "output": ("debug", "verbose", "normal", "quiet", "silent"),
}
_FLAG_OPTIONS: Final = {"speculative"}
_STR_OPTIONS: Final = {"output_file"}

_DEFAULTS: Final[dict[str, Any]] = {
    "max_iterations": 100,
    "max_function_evaluations": 1000,
    "convergence_tolerance": 1.0e-4,
    "gradient_tolerance": 1.0e-4,
    "max_step": 1000.0,
    "merit_function": "argaez_tapia",
}

# Merit function: (OPT++ name, steplength_to_boundary, centering_parameter).
_MERIT_FUNCTIONS: Final = {
    "el_bakry": ("norm_fmu", 0.8, 0.2),
    "argaez_tapia": ("argaez_tapia", 0.99995, 0.2),
    "van_shanno": ("van_shanno", 0.95, 0.1),
}


def validate_options(options: dict[str, Any]) -> dict[str, Any]:
    """Check `optpp_q_newton` options given as a dictionary.

    Args:
        options: The options, keyed by option name.

    Returns:
        A copy of the options, with numeric values converted to `int` or
        `float`.

    Raises:
        ValueError: If an option is unknown or has an invalid value.

    """
    unknown = set(options) - set(_CHECKS)
    if unknown:
        msg = f"Unsupported option(s) for optpp_q_newton: {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    for key, value in options.items():
        is_valid, expected = _CHECKS[key]
        if not is_valid(value):
            msg = f"Option {key} must be {expected}, got: {value!r}"
            raise ValueError(msg)
    return {key: _normalize(key, value) for key, value in options.items()}


def _normalize(key: str, value: object) -> object:
    if key in _INT_OPTIONS:
        assert isinstance(value, Integral)
        return int(value)
    if key in _FLOAT_OPTIONS:
        assert isinstance(value, Real)
        return float(value)
    return value


def _is_count(value: object) -> bool:
    return (
        isinstance(value, Integral) and not isinstance(value, bool) and int(value) >= 0
    )


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_fraction(value: object) -> bool:
    if not _is_number(value):
        return False
    assert isinstance(value, Real)
    if value < 0:
        return False
    return bool(value <= 1)


def _choice_check(key: str) -> tuple[Callable[[object], bool], str]:
    choices = _CHOICE_OPTIONS[key]
    return (lambda value: value in choices), f"one of {', '.join(choices)}"


_CHECKS: Final[dict[str, tuple[Callable[[object], bool], str]]] = (
    dict.fromkeys(_INT_OPTIONS, (_is_count, "a non-negative integer"))
    | dict.fromkeys(_FLOAT_OPTIONS, (_is_number, "a number"))
    | {"steplength_to_boundary": (_is_fraction, "a number between 0 and 1 inclusive")}
    | {key: _choice_check(key) for key in _CHOICE_OPTIONS}
    | dict.fromkeys(_FLAG_OPTIONS, (lambda value: type(value) is bool, "a boolean"))
    | dict.fromkeys(_STR_OPTIONS, (lambda value: type(value) is str, "a string"))
)


def minimize_optpp_q_newton(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    evaluate: Evaluator,
    x0: NDArray[np.float64],
    bounds: Bounds | None,
    linear_constraints: LinearConstraints | None,
    nonlinear_equalities: NDArray[np.bool_],
    options: dict[str, Any] | None,
) -> tuple[NDArray[np.float64], int]:
    """Minimize a function with the `optpp_q_newton` method.

    Args:
        evaluate:             The combined objective and nonlinear constraint evaluator.
        x0:                   The initial point.
        bounds:               The variable bounds.
        linear_constraints:   The linear constraints, with at least one row.
        nonlinear_equalities: Marks the nonlinear constraints that are equalities.
        options:              The options, keyed by option name.

    Returns:
        The final point and the OPT++ termination code.

    """
    method_options = _DEFAULTS | validate_options(options or {})

    finite_bounds = (
        bounds if bounds is not None and _has_finite_bounds(bounds) else None
    )
    has_nonlinear = bool(nonlinear_equalities.size)
    problem = NLF1Problem(evaluate, x0, nonlinear_equalities)

    if linear_constraints is not None or has_nonlinear:
        return run_q_nips(
            problem,
            finite_bounds,
            linear_constraints,
            _qnips_options(method_options, has_nonlinear=has_nonlinear),
        )
    if finite_bounds is not None:
        return run_bcq_newton(
            problem, finite_bounds, _newton_options(method_options, has_bounds=True)
        )
    return run_q_newton(problem, _newton_options(method_options, has_bounds=False))


def _has_finite_bounds(bounds: Bounds) -> bool:
    lower, upper = bounds
    return bool(np.any(lower > -_BIG_BOUND) or np.any(upper < _BIG_BOUND))


def _common_options(
    method_options: dict[str, Any], *, mode_override: bool
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "debug": method_options.get("output") == "debug",
        "max_iter": method_options["max_iterations"],
        "fcn_tol": method_options["convergence_tolerance"],
        "max_step": method_options["max_step"],
        "mode_override": mode_override,
    }
    if "output_file" in method_options:
        options["output_file"] = method_options["output_file"]
    return options


def _newton_options(
    method_options: dict[str, Any], *, has_bounds: bool
) -> dict[str, Any]:
    search_method = method_options.get("search_method")
    match search_method:
        case "value_based_line_search" | "gradient_based_line_search":
            search_strategy = "line_search"
        case None if has_bounds:
            search_strategy = "line_search"
        case _:
            search_strategy = "trust_region"
    mode_override = (
        method_options.get("speculative", False)
        or search_method == "gradient_based_line_search"
    )
    is_expensive = search_method == "value_based_line_search" or (
        search_method is None and has_bounds
    )
    options = _common_options(method_options, mode_override=mode_override) | {
        "max_feval": method_options["max_function_evaluations"],
        "grad_tol": method_options["gradient_tolerance"],
        "search_strategy": search_strategy,
        "is_expensive": is_expensive,
    }
    if search_strategy == "trust_region":
        options["tr_size"] = method_options["max_step"]
    return options


def _qnips_options(
    method_options: dict[str, Any], *, has_nonlinear: bool
) -> dict[str, Any]:
    merit_fcn, steplength, centering = _MERIT_FUNCTIONS[
        method_options["merit_function"]
    ]
    mode_override = method_options.get("speculative", False) or has_nonlinear
    return _common_options(method_options, mode_override=mode_override) | {
        "max_feval": method_options["max_function_evaluations"],
        "merit_fcn": merit_fcn,
        "step_length_to_bdry": method_options.get("steplength_to_boundary", steplength),
        "centering_parameter": method_options.get("centering_parameter", centering),
    }
