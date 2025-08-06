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
from scipy.optimize import Bounds

from everest_optimizers import minimize

src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

_Function = Callable[[NDArray[np.float64]], float]

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

def objective(x: NDArray[np.float64]) -> float:
    return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2 + (x[2] - 3.0) ** 2


def objective_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([2 * (x[0] - 1.0), 2 * (x[1] - 2.0), 2 * (x[2] - 3.0)])


@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"),
    [
        ([-1.0, -1.0, -1.0], [1.0, 1.0, 0.2]),
        ([0.1, 0.1, 0.1], [1.0, 1.0, 1.0]),
        ([-0.2, -0.2, 0.6], [0.2, 0.2, 1.0]),
    ],
)
def test_ropt_vs_everest_bounds(
    enopt_config: dict[str, Any],
    evaluator: Callable[[list[_Function]], Any],
    lower_bounds: list[float],
    upper_bounds: list[float],
) -> None:
    initial_values = np.array([0.0, 0.0, 0.0])

    # --- ropt optimizer ---
    ropt_config = enopt_config.copy()
    ropt_config["variables"].update({"lower_bounds": lower_bounds, "upper_bounds": upper_bounds})
    ropt_config["objectives"] = {"weights": [1.0]}
    ropt_optimizer = BasicOptimizer(ropt_config, evaluator([objective]))
    ropt_result = ropt_optimizer.run(initial_values)
    ropt_solution = ropt_result.variables
    assert ropt_solution is not None

    # --- everest_optimizers ---
    bounds = Bounds(lower_bounds, upper_bounds)
    res_everest = minimize(
        objective,
        initial_values,
        method='optpp_constr_q_newton',
        jac=objective_grad,
        bounds=bounds,
    )
    assert res_everest.success

    np.testing.assert_allclose(ropt_solution, res_everest.x, rtol=1e-2, atol=1e-2)
