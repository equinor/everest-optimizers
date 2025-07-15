# test/ropt-dakota-vs-everest-optimizers/test_ropt_dakota_vs_everest_optimizers.py

import numpy as np
import pytest
from ropt.config import EnOptConfig
from functools import partial

from ropt.evaluator import EvaluatorContext, EvaluatorResult
from ropt.plan import BasicOptimizer

from everest_optimizers import minimize


def _function_runner(
    variables: np.ndarray,
    evaluator_context: EvaluatorContext,
    functions: list[callable],
) -> EvaluatorResult:
    objective_count = evaluator_context.config.objectives.weights.size
    objective_results = np.zeros(
        (variables.shape[0], objective_count), dtype=np.float64
    )
    for eval_idx in range(evaluator_context.realizations.size):
        if evaluator_context.active[eval_idx]:
            for idx in range(objective_count):
                function = functions[idx]
                objective_results[eval_idx, idx] = function(variables[eval_idx, :])
    return EvaluatorResult(
        objectives=objective_results,
        constraints=None,
    )


def rosenbrock(x: np.ndarray) -> float:
    """The Rosenbrock function."""
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)


@pytest.mark.parametrize(
    "start_point",
    [
        np.array([0.0, 0.0]),
        np.array([-1.2, 1.0]),
    ],
)
def test_optqnewton_comparison(start_point: np.ndarray) -> None:
    """Compare OptQNewton results from Dakota and Everest Optimizers."""
    # Run with everest-optimizers
    everest_result = minimize(
        rosenbrock, start_point, method="OptQNewton", options={"max_iterations": 100}
    )

    # Run with ropt-dakota
    dakota_config_dict = {
        "variables": {
            "variable_count": len(start_point),
        },
        "optimizer": {
            "method": "optpp_q_newton",
            "options": ["max_iterations = 100", "convergence_tolerance = 1e-8"],
        },
        "objectives": {
            "weights": [1.0],
        },
    }
    dakota_config = EnOptConfig.model_validate(dakota_config_dict)
    evaluator = partial(_function_runner, functions=[rosenbrock])
    dakota_optimizer = BasicOptimizer(dakota_config, evaluator)
    dakota_optimizer.run(start_point)

    ropt_results = dakota_optimizer.results
    assert ropt_results is not None
    assert ropt_results.functions is not None
    assert ropt_results.functions.objectives is not None

    dakota_solution = dakota_optimizer.variables
    assert dakota_solution is not None
    dakota_fun = ropt_results.functions.objectives[0]

    # Verify solutions are the same
    assert np.allclose(dakota_solution, everest_result.x, atol=1e-4), \
        f"Dakota solution {dakota_solution} is not close to Everest solution {everest_result.x}"
    assert np.allclose(dakota_fun, everest_result.fun, atol=1e-4), \
        f"Dakota fun {dakota_fun} is not close to Everest fun {everest_result.fun}"

    # Compare the results from both optimizers
    assert np.allclose(everest_result.x, dakota_solution, atol=1e-5)
    assert np.allclose(everest_result.fun, dakota_fun, atol=1e-5)

