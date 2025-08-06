"""Tests for expected solutions using ropt BasicOptimizer.

This module mirrors the structure of
`test_optimizer_convergence.py` but instead of comparing the two
optimizers against each other, it checks that the *ropt* optimizer
converges to the analytically expected solution for both constrained
and unconstrained problems.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any
import sys
import os

import numpy as np
import pytest
from numpy.typing import NDArray
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import BasicOptimizer

# Add the source directory to the path to find everest_optimizers (for type hints)
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

_Function = Callable[[NDArray[np.float64]], float]

# -----------------------------------------------------------------------------
# Pytest option/collection helpers – keep identical behaviour as other tests.
# -----------------------------------------------------------------------------

def pytest_addoption(parser: Any) -> Any:  # noqa: D401 – same style as original
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="run tests with external optimizers",
    )


def pytest_collection_modifyitems(config: Any, items: Sequence[Any]) -> None:  # noqa: D401
    if not config.getoption("--run-external"):
        skip_external = pytest.mark.skip(reason="need --run-external option to run")
        for item in items:
            if "external" in item.keywords:
                item.add_marker(skip_external)


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _function_runner(
    variables: NDArray[np.float64],
    evaluator_context: EvaluatorContext,
    functions: list[_Function],
) -> EvaluatorResult:
    """Run objective / constraint functions mimicking the *ropt* interface."""
    objective_count = evaluator_context.config.objectives.weights.size
    constraint_count = (
        0
        if evaluator_context.config.nonlinear_constraints is None
        else evaluator_context.config.nonlinear_constraints.lower_bounds.size
    )
    objective_results = np.zeros((variables.shape[0], objective_count), dtype=np.float64)
    constraint_results = (
        np.zeros((variables.shape[0], constraint_count), dtype=np.float64)
        if constraint_count > 0
        else None
    )
    for eval_idx in range(evaluator_context.realizations.size):
        if evaluator_context.active[eval_idx]:
            for idx in range(objective_count):
                function = functions[idx]
                objective_results[eval_idx, idx] = function(variables[eval_idx, :])
            for idx in range(constraint_count):
                function = functions[idx + objective_count]
                assert constraint_results is not None
                constraint_results[eval_idx, idx] = function(variables[eval_idx, :])
    return EvaluatorResult(objectives=objective_results, constraints=constraint_results)


def _compute_distance_squared(
    variables: NDArray[np.float64], target: NDArray[np.float64]
) -> float:
    return float(((variables - target) ** 2).sum())


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(name="test_functions", scope="session")
def fixture_test_functions() -> tuple[_Function, _Function]:
    return (
        partial(_compute_distance_squared, target=np.array([0.5, 0.5, 0.5])),
        partial(_compute_distance_squared, target=np.array([-1.5, -1.5, 0.5])),
    )


@pytest.fixture(scope="session")
def evaluator(test_functions: Any) -> Callable[[list[_Function]], Any]:  # noqa: D401
    def _evaluator(test_functions: list[_Function] = test_functions) -> Any:
        return partial(_function_runner, functions=test_functions)

    return _evaluator


# Various starting points (same as convergence tests)
initial_values_1: list[float] = [0.0, 0.0, 0.1]
initial_values_2: list[float] = [1.0, -1.0, 0.5]
initial_values_3: list[float] = [-0.5, 0.5, -0.2]


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    """Standard configuration for the EnOpt optimizer."""
    return {
        "variables": {
            "variable_count": 3,
            "perturbation_magnitudes": 0.01,
        },
        "optimizer": {
            "method": "optpp_q_newton",
            "tolerance": 1e-6,
        },
        "objectives": {
            "weights": [0.75, 0.25],
        },
    }


# -----------------------------------------------------------------------------
# Analytical expected solutions helpers
# -----------------------------------------------------------------------------

expected_unconstrained = np.array([0.0, 0.0, 0.5], dtype=np.float64)


def _project_to_bounds(
    x: NDArray[np.float64], lower: list[float] | NDArray[np.float64], upper: list[float] | NDArray[np.float64]
) -> NDArray[np.float64]:
    lower_a = np.asarray(lower, dtype=np.float64)
    upper_a = np.asarray(upper, dtype=np.float64)
    return np.minimum(np.maximum(x, lower_a), upper_a)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.external
@pytest.mark.parametrize("initial_values", [initial_values_1, initial_values_2, initial_values_3])
def test_ropt_unconstrained_expected(
    enopt_config: dict[str, Any],
    evaluator: Callable[[list[_Function]], Any],
    initial_values: list[float],
) -> None:
    """ropt optimizer should converge to analytical solution for unconstrained case."""
    ropt_optimizer = BasicOptimizer(enopt_config, evaluator())
    ropt_result = ropt_optimizer.run(initial_values)
    ropt_solution = ropt_result.variables
    assert ropt_solution is not None
    np.testing.assert_allclose(ropt_solution, expected_unconstrained, rtol=1e-2, atol=1e-2)


@pytest.mark.external
@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"),
    [
        ([-1.0, -1.0, -1.0], [1.0, 1.0, 0.2]),
        ([0.1, 0.1, 0.1], [1.0, 1.0, 1.0]),
        ([-0.2, -0.2, 0.6], [0.2, 0.2, 1.0]),
    ],
)
def test_ropt_constrained_expected(
    enopt_config: dict[str, Any],
    evaluator: Callable[[list[_Function]], Any],
    lower_bounds: list[float],
    upper_bounds: list[float],
) -> None:
    """ropt optimizer should converge to projected analytical solution for constrained case."""
    ropt_config = enopt_config.copy()
    ropt_config["variables"].update({"lower_bounds": lower_bounds, "upper_bounds": upper_bounds})
    ropt_optimizer = BasicOptimizer(ropt_config, evaluator())
    ropt_result = ropt_optimizer.run(initial_values_1)
    ropt_solution = ropt_result.variables
    assert ropt_solution is not None

    expected = _project_to_bounds(expected_unconstrained, lower_bounds, upper_bounds)
    np.testing.assert_allclose(ropt_solution, expected, rtol=1e-2, atol=1e-2)
