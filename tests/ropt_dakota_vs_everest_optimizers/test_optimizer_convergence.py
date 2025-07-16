# tests/ropt_dakota_vs_everest_optimizers/test_optimizer_convergence.py

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


# Define various initial values for the optimizer
initial_values_1 = [0.0, 0.0, 0.1]
initial_values_2 = [1.0, -1.0, 0.5]
initial_values_3 = [-0.5, 0.5, -0.2]


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
def test_unconstrained_convergence(
    enopt_config: Any, 
    evaluator: Any, 
    everest_objective: Any, 
    initial_values: list[float]
) -> None:
    """Tests that ropt and everest-optimizers converge to a similar solution."""
    # Run ropt optimizer
    ropt_optimizer = BasicOptimizer(enopt_config, evaluator())
    ropt_result = ropt_optimizer.run(initial_values)
    ropt_solution = ropt_result.variables
    assert ropt_solution is not None

    # Run everest-optimizer
    from everest_optimizers import minimize
    everest_result = minimize(everest_objective, initial_values, method="optpp_q_newton")
    everest_solution = everest_result.x
    assert everest_solution is not None

    # Compare solutions
    np.testing.assert_allclose(
        ropt_solution, everest_solution, rtol=1e-2, atol=1e-2
    )

@pytest.mark.skip(reason="Skipping bound constraint tests for now")
@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"),
    [
        ([-1.0, -1.0, -1.0], [1.0, 1.0, 0.2]),
        ([0.1, 0.1, 0.1], [1.0, 1.0, 1.0]),
        ([-0.2, -0.2, 0.6], [0.2, 0.2, 1.0]),
    ],
)
def test_constrained_convergence(
    enopt_config: Any,
    evaluator: Any,
    everest_objective: Any,
    lower_bounds: list[float],
    upper_bounds: list[float],
) -> None:
    """Tests that ropt and everest-optimizers converge to a similar solution for constrained problems."""
    from everest_optimizers import minimize
    from scipy.optimize import Bounds

    # Run ropt optimizer
    ropt_config = enopt_config.copy()
    ropt_config["variables"]["lower_bounds"] = lower_bounds
    ropt_config["variables"]["upper_bounds"] = upper_bounds
    ropt_optimizer = BasicOptimizer(ropt_config, evaluator())
    ropt_result = ropt_optimizer.run(initial_values_1)
    ropt_solution = ropt_result.variables
    assert ropt_solution is not None

    # Run everest-optimizer
    bounds = Bounds(lower_bounds, upper_bounds)
    everest_result = minimize(
        everest_objective, initial_values_1, method="optpp_q_newton", bounds=bounds
    )
    everest_solution = everest_result.x
    assert everest_solution is not None

    # Compare solutions
    np.testing.assert_allclose(ropt_solution, everest_solution, rtol=1e-2, atol=1e-2)
