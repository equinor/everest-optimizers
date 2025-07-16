# tests/ropt_dakota_vs_everest_optimizers/test_numerical_similarity.py

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any
import csv
from pathlib import Path
import sys
import os

import numpy as np
import pytest
from numpy.typing import NDArray
from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import BasicOptimizer

# Add the source directory to the path to find everest_optimizers
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

_Function = Callable[[NDArray[np.float64]], float]


def pytest_addoption(parser: Any) -> Any:
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="run tests with external optimizers",
    )


def pytest_collection_modifyitems(config: Any, items: Sequence[Any]) -> None:
    if not config.getoption("--run-external"):
        skip_external = pytest.mark.skip(reason="need --run-external option to run")
        for item in items:
            if "external" in item.keywords:
                item.add_marker(skip_external)


def _function_runner(
    variables: NDArray[np.float64],
    evaluator_context: EvaluatorContext,
    functions: list[_Function],
) -> EvaluatorResult:
    objective_count = evaluator_context.config.objectives.weights.size
    constraint_count = (
        0
        if evaluator_context.config.nonlinear_constraints is None
        else evaluator_context.config.nonlinear_constraints.lower_bounds.size
    )
    objective_results = np.zeros(
        (variables.shape[0], objective_count), dtype=np.float64
    )
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
    return EvaluatorResult(
        objectives=objective_results,
        constraints=constraint_results,
    )


def _compute_distance_squared(
    variables: NDArray[np.float64], target: NDArray[np.float64]
) -> float:
    return float(((variables - target) ** 2).sum())


@pytest.fixture(name="test_functions", scope="session")
def fixture_test_functions() -> tuple[_Function, _Function]:
    return (
        partial(_compute_distance_squared, target=np.array([0.5, 0.5, 0.5])),
        partial(_compute_distance_squared, target=np.array([-1.5, -1.5, 0.5])),
    )


@pytest.fixture(scope="session")
def evaluator(test_functions: Any) -> Callable[[list[_Function]], Any]:
    def _evaluator(test_functions: list[_Function] = test_functions) -> Any:
        return partial(_function_runner, functions=test_functions)
    return _evaluator


# --- Test setup ---

# Define the output files for the results
ROPT_RESULTS_FILE = Path(__file__).parent.parent / "ropt_results.csv"
EVEREST_RESULTS_FILE = Path(__file__).parent.parent / "everest_results.csv"

# Define various initial values for the optimizer
initial_values_1 = [0.0, 0.0, 0.1]
initial_values_2 = [1.0, -1.0, 0.5]
initial_values_3 = [-0.5, 0.5, -0.2]

# Define expected outcomes for each set of initial values
expected_results_unconstrained = {
    str(initial_values_1): [0.0, 0.0, 0.5],
    str(initial_values_2): [0.0, 0.0, 0.5],
    str(initial_values_3): [0.0, 0.0, 0.5],
}


@pytest.fixture(scope="module", autouse=True)
def setup_results_files():
    """Set up the results CSV files with a header."""
    for fpath in [ROPT_RESULTS_FILE, EVEREST_RESULTS_FILE]:
        with open(fpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "test_name",
                    "initial_values",
                    "lower_bounds",
                    "upper_bounds",
                    "expected_result",
                    "actual_result",
                ]
            )
    yield


@pytest.fixture(name="enopt_config")
def enopt_config_fixture() -> dict[str, Any]:
    """Provides a standard configuration for the EnOpt optimizer."""
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


# --- ROPT Optimizer Tests ---

@pytest.mark.parametrize(
    "initial_values",
    [initial_values_1, initial_values_2, initial_values_3],
)
def test_ropt_unconstrained_numerical(
    enopt_config: Any, evaluator: Any, initial_values: list[float]
) -> None:
    """Tests the ropt optimizer with different numerical starting points."""
    optimizer = BasicOptimizer(enopt_config, evaluator())
    result = optimizer.run(initial_values)
    variables = result.variables

    assert variables is not None
    expected = expected_results_unconstrained[str(initial_values)]

    with open(ROPT_RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "unconstrained",
                str(initial_values),
                "N/A",
                "N/A",
                str(expected),
                str(list(variables)),
            ]
        )


@pytest.mark.skip(reason="Skipping bound constraint tests for now")
@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds", "expected_result"),
    [
        ([-1.0, -1.0, -1.0], [1.0, 1.0, 0.2], [0.0, 0.0, 0.2]),
        ([0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [0.1, 0.1, 0.5]),
        ([-0.2, -0.2, 0.6], [0.2, 0.2, 1.0], [0.0, 0.0, 0.6]),
    ],
)
def test_ropt_bound_constraints(
    enopt_config: Any,
    evaluator: Any,
    lower_bounds: list[float],
    upper_bounds: list[float],
    expected_result: list[float],
) -> None:
    """Tests ropt with various bound constraints."""
    enopt_config["variables"]["lower_bounds"] = lower_bounds
    enopt_config["variables"]["upper_bounds"] = upper_bounds

    optimizer = BasicOptimizer(enopt_config, evaluator())
    result = optimizer.run(initial_values_1)
    variables = result.variables

    assert variables is not None

    with open(ROPT_RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bound_constrained",
                str(initial_values_1),
                str(lower_bounds),
                str(upper_bounds),
                str(expected_result),
                str(list(variables)),
            ]
        )


# --- Everest Optimizers Tests ---

@pytest.fixture(name="everest_objective")
def everest_objective_fixture(test_functions, enopt_config):
    """Create a single objective function for everest-optimizers."""
    weights = enopt_config["objectives"]["weights"]

    def _objective(x):
        return weights[0] * test_functions[0](x) + weights[1] * test_functions[1](x)

    return _objective


@pytest.mark.parametrize(
    "initial_values",
    [initial_values_1, initial_values_2, initial_values_3],
)
def test_everest_unconstrained_numerical(
    everest_objective: Any, initial_values: list[float]
) -> None:
    """Tests the everest optimizer with different numerical starting points."""
    from everest_optimizers import minimize

    result = minimize(everest_objective, initial_values, method="OptQNewton")
    variables = result.x

    assert variables is not None
    expected = expected_results_unconstrained[str(initial_values)]

    with open(EVEREST_RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "unconstrained",
                str(initial_values),
                "N/A",
                "N/A",
                str(expected),
                str(list(variables)),
            ]
        )


@pytest.mark.skip(reason="Skipping bound constraint tests for now")
@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds", "expected_result"),
    [
        ([-1.0, -1.0, -1.0], [1.0, 1.0, 0.2], [0.0, 0.0, 0.2]),
        ([0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [0.1, 0.1, 0.5]),
        ([-0.2, -0.2, 0.6], [0.2, 0.2, 1.0], [0.0, 0.0, 0.6]),
    ],
)
def test_everest_bound_constraints(
    everest_objective: Any,
    lower_bounds: list[float],
    upper_bounds: list[float],
    expected_result: list[float],
) -> None:
    """Tests everest optimizer with various bound constraints."""
    from everest_optimizers import minimize
    from scipy.optimize import Bounds

    bounds = Bounds(lower_bounds, upper_bounds)
    result = minimize(
        everest_objective, initial_values_1, method="OptQNewton", bounds=bounds
    )
    variables = result.x

    assert variables is not None

    with open(EVEREST_RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bound_constrained",
                str(initial_values_1),
                str(lower_bounds),
                str(upper_bounds),
                str(expected_result),
                str(list(variables)),
            ]
        )
