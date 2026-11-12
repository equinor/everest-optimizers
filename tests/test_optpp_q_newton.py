from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from everest_optimizers import EvaluationMode, _optpp_q_newton, minimize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from everest_optimizers import EvaluationResult
    from everest_optimizers._optpp import Bounds, LinearConstraints


def evaluate(x: NDArray[np.float64], mode: EvaluationMode) -> EvaluationResult:
    delta = x - [2.0, -1.0]
    values = np.array([delta @ delta]) if mode & EvaluationMode.VALUES else None
    gradients = 2 * delta[np.newaxis, :] if mode & EvaluationMode.GRADIENTS else None
    return values, gradients


DEFAULTS = {
    "debug": False,
    "max_iter": 100,
    "fcn_tol": 1e-4,
    "max_step": 1000.0,
}

RUNNERS = ("run_q_newton", "run_bcq_newton", "run_lbfgs", "run_q_nips")

LINEAR_CONSTRAINTS = (np.array([[1.0, 1.0]]), np.array([-np.inf]), np.array([1.0]))


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, tuple[Any, ...]]]:
    recorded: list[tuple[str, tuple[Any, ...]]] = []

    def make_runner(name: str) -> Any:
        def runner(*args: Any) -> tuple[NDArray[np.float64], int]:
            recorded.append((name, args))
            return np.zeros(0), 0

        return runner

    for name in RUNNERS:
        monkeypatch.setattr(_optpp_q_newton, name, make_runner(name))
    return recorded


def run(
    calls: list[tuple[str, tuple[Any, ...]]],
    *,
    size: int = 2,
    bounds: Bounds | None = None,
    linear_constraints: LinearConstraints | None = None,
    nonlinear_equalities: NDArray[np.bool_] | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[str, tuple[Any, ...]]:
    minimize(
        np.zeros(size),
        evaluate=evaluate,
        bounds=bounds,
        linear_constraints=linear_constraints,
        nonlinear_equalities=nonlinear_equalities,
        options=options,
    )
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"max_iterations": 1.5}, "must be a non-negative integer"),
        ({"max_iterations": True}, "must be a non-negative integer"),
        ({"max_step": "1"}, "must be a number"),
        ({"merit_function": "norm_fmu"}, "must be one of"),
        ({"speculative": 1}, "must be a boolean"),
        ({"max_iter": 10}, "Unsupported option"),
    ],
)
def test_minimize_rejects_invalid_option_dict(
    options: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        minimize(np.zeros(2), evaluate=evaluate, options=options)


@pytest.mark.parametrize(
    "step_length",
    [
        -0.1,
        1.1,
        np.nan,
        np.inf,
        -np.inf,
        pytest.param(Fraction(2**53 + 1, 2**53), id="fraction-just-above-one"),
        True,
    ],
)
def test_minimize_rejects_invalid_steplength_to_boundary(
    calls: list[tuple[str, tuple[Any, ...]]], step_length: object
) -> None:
    with pytest.raises(
        ValueError, match="steplength_to_boundary must be a number between 0 and 1"
    ):
        minimize(
            np.zeros(2),
            evaluate=evaluate,
            linear_constraints=LINEAR_CONSTRAINTS,
            options={"steplength_to_boundary": step_length},
        )
    assert calls == []


@pytest.mark.parametrize("step_length", [0.0, 1.0, np.float32(0.5), Fraction(1, 2)])
def test_minimize_accepts_valid_steplength_to_boundary(
    calls: list[tuple[str, tuple[Any, ...]]], step_length: object
) -> None:
    _, args = run(
        calls,
        linear_constraints=LINEAR_CONSTRAINTS,
        options={"steplength_to_boundary": step_length},
    )
    assert args[-1]["step_length_to_bdry"] == step_length


@pytest.mark.parametrize(
    ("size", "bounds", "linear_constraints"),
    [
        (2, None, None),
        (2, (np.zeros(2), np.ones(2)), None),
        (100, None, None),
        (2, None, LINEAR_CONSTRAINTS),
    ],
)
def test_minimize_rejects_tr_pds(
    calls: list[tuple[str, tuple[Any, ...]]],
    size: int,
    bounds: Bounds | None,
    linear_constraints: LinearConstraints | None,
) -> None:
    with pytest.raises(ValueError, match="Option search_method must be one of"):
        minimize(
            np.zeros(size),
            evaluate=evaluate,
            bounds=bounds,
            linear_constraints=linear_constraints,
            options={"search_method": "tr_pds"},
        )
    assert calls == []


def test_minimize_unconstrained_problem_uses_q_newton(
    calls: list[tuple[str, tuple[Any, ...]]],
) -> None:
    name, args = run(calls, bounds=(np.full(2, -np.inf), np.full(2, np.inf)))
    assert name == "run_q_newton"
    assert args[-1] == DEFAULTS | {
        "mode_override": False,
        "max_feval": 1000,
        "grad_tol": 1e-4,
        "search_strategy": "trust_region",
        "is_expensive": False,
        "tr_size": 1000.0,
    }


def test_minimize_ignores_bounds_beyond_large_magnitude(
    calls: list[tuple[str, tuple[Any, ...]]],
) -> None:
    name, _ = run(calls, bounds=(np.full(2, -1e30), np.full(2, 1e30)))
    assert name == "run_q_newton"


def test_minimize_bounds_use_bcq_newton(
    calls: list[tuple[str, tuple[Any, ...]]],
) -> None:
    lower = np.array([0.0, -np.inf])
    upper = np.full(2, np.inf)
    name, args = run(calls, bounds=(lower, upper))
    assert name == "run_bcq_newton"
    np.testing.assert_array_equal(args[1][0], lower)
    np.testing.assert_array_equal(args[1][1], upper)
    assert args[-1] == DEFAULTS | {
        "mode_override": False,
        "max_feval": 1000,
        "grad_tol": 1e-4,
        "search_strategy": "line_search",
        "is_expensive": True,
    }


def test_minimize_large_problem_uses_lbfgs(
    calls: list[tuple[str, tuple[Any, ...]]],
) -> None:
    name, args = run(calls, size=100)
    assert name == "run_lbfgs"
    assert args[-1] == DEFAULTS | {
        "mode_override": False,
        "grad_tol": 1e-4,
        "is_expensive": False,
    }


def test_minimize_just_below_large_scale_uses_q_newton(
    calls: list[tuple[str, tuple[Any, ...]]],
) -> None:
    name, _ = run(calls, size=99)
    assert name == "run_q_newton"


def test_minimize_linear_constraints_use_q_nips(
    calls: list[tuple[str, tuple[Any, ...]]],
) -> None:
    name, args = run(
        calls,
        bounds=(np.full(2, -np.inf), np.full(2, np.inf)),
        linear_constraints=LINEAR_CONSTRAINTS,
        options={"search_method": "gradient_based_line_search"},
    )
    assert name == "run_q_nips"
    assert args[1] is None
    for actual, expected in zip(args[2], LINEAR_CONSTRAINTS, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert args[-1] == DEFAULTS | {
        "mode_override": False,
        "max_feval": 1000,
        "merit_fcn": "argaez_tapia",
        "step_length_to_bdry": 0.99995,
        "centering_parameter": 0.2,
    }


def test_minimize_without_linear_constraint_rows_does_not_use_q_nips(
    calls: list[tuple[str, tuple[Any, ...]]],
) -> None:
    name, _ = run(
        calls, linear_constraints=(np.zeros((0, 2)), np.zeros(0), np.zeros(0))
    )
    assert name == "run_q_newton"


def test_minimize_nonlinear_constraints_use_q_nips_with_mode_override(
    calls: list[tuple[str, tuple[Any, ...]]],
) -> None:
    name, args = run(calls, nonlinear_equalities=np.array([False]))
    assert name == "run_q_nips"
    assert args[-1]["mode_override"] is True


@pytest.mark.parametrize(
    ("merit_function", "expected"),
    [
        ("el_bakry", ("norm_fmu", 0.8, 0.2)),
        ("argaez_tapia", ("argaez_tapia", 0.99995, 0.2)),
        ("van_shanno", ("van_shanno", 0.95, 0.1)),
    ],
)
def test_minimize_merit_function_sets_defaults(
    calls: list[tuple[str, tuple[Any, ...]]],
    merit_function: str,
    expected: tuple[str, float, float],
) -> None:
    _, args = run(
        calls,
        bounds=(np.zeros(2), np.ones(2)),
        linear_constraints=LINEAR_CONSTRAINTS,
        options={"merit_function": merit_function},
    )
    settings = args[-1]
    assert (
        settings["merit_fcn"],
        settings["step_length_to_bdry"],
        settings["centering_parameter"],
    ) == expected


def test_minimize_explicit_interior_point_options_win(
    calls: list[tuple[str, tuple[Any, ...]]],
) -> None:
    _, args = run(
        calls,
        linear_constraints=LINEAR_CONSTRAINTS,
        options={
            "merit_function": "van_shanno",
            "steplength_to_boundary": 0.5,
            "centering_parameter": 0.3,
        },
    )
    settings = args[-1]
    assert (settings["step_length_to_bdry"], settings["centering_parameter"]) == (
        0.5,
        0.3,
    )


@pytest.mark.parametrize(
    ("search_method", "bounded", "strategy", "is_expensive", "mode_override"),
    [
        ("value_based_line_search", False, "line_search", True, False),
        ("gradient_based_line_search", False, "line_search", False, True),
        ("gradient_based_line_search", True, "line_search", False, True),
        ("trust_region", False, "trust_region", False, False),
        ("trust_region", True, "trust_region", False, False),
    ],
)
def test_minimize_search_method_maps_to_strategy(  # ruff: ignore[too-many-positional-arguments]
    calls: list[tuple[str, tuple[Any, ...]]],
    search_method: str,
    bounded: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
    strategy: str,
    is_expensive: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
    mode_override: bool,  # ruff: ignore[boolean-type-hint-positional-argument]
) -> None:
    _, args = run(
        calls,
        bounds=(np.zeros(2), np.ones(2)) if bounded else None,
        options={"search_method": search_method, "max_step": 5},
    )
    settings = args[-1]
    assert settings["search_strategy"] == strategy
    assert settings["is_expensive"] is is_expensive
    assert settings["mode_override"] is mode_override
    assert settings.get("tr_size") == (5.0 if strategy == "trust_region" else None)


def test_minimize_maps_common_options(
    calls: list[tuple[str, tuple[Any, ...]]],
) -> None:
    _, args = run(
        calls,
        options={
            "max_iterations": 3,
            "max_function_evaluations": 4,
            "convergence_tolerance": 1e-8,
            "gradient_tolerance": 1e-9,
            "speculative": True,
            "output": "debug",
            "output_file": "opt.out",
        },
    )
    assert args[-1] == DEFAULTS | {
        "debug": True,
        "max_iter": 3,
        "fcn_tol": 1e-8,
        "mode_override": True,
        "output_file": "opt.out",
        "max_feval": 4,
        "grad_tol": 1e-9,
        "search_strategy": "trust_region",
        "is_expensive": False,
        "tr_size": 1000.0,
    }
