"""Tests for expected solutions using scipy.optimize.minimize BFGS.

This module mirrors the structure of the original test but checks that the
scipy.optimize.minimize optimizer converges to the analytically expected
solution for both constrained and unconstrained problems.
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
from scipy.optimize import minimize, Bounds

# Add the source directory to the path to find everest_optimizers (for type hints)
# This part is kept for structural consistency, though not strictly necessary for this version.
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

_Function = Callable[[NDArray[np.float64]], float]

# -----------------------------------------------------------------------------
# Pytest option/collection helpers – keep identical behaviour as other tests.
# -----------------------------------------------------------------------------

def pytest_addoption(parser: Any) -> Any:  # noqa: D401 – same style as original
    """Adds a command line option to run tests with external optimizers."""
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="run tests with external optimizers",
    )


def pytest_collection_modifyitems(config: Any, items: Sequence[Any]) -> None:  # noqa: D401
    """Skips tests marked as 'external' unless --run-external is specified."""
    if not config.getoption("--run-external"):
        skip_external = pytest.mark.skip(reason="need --run-external option to run")
        for item in items:
            if "external" in item.keywords:
                item.add_marker(skip_external)


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _objective_function(
    x: NDArray[np.float64], functions: list[_Function], weights: list[float]
) -> float:
    """Computes the weighted sum of objective functions for SciPy."""
    objective_value = 0.0
    for func, weight in zip(functions, weights):
        objective_value += func(x) * weight
    return objective_value


def _compute_distance_squared(
    variables: NDArray[np.float64], target: NDArray[np.float64]
) -> float:
    """Computes the squared Euclidean distance from a target point."""
    return float(((variables - target) ** 2).sum())


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(name="test_functions", scope="session")
def fixture_test_functions() -> tuple[_Function, _Function]:
    """Provides the two objective functions to be minimized."""
    return (
        partial(_compute_distance_squared, target=np.array([0.5, 0.5, 0.5])),
        partial(_compute_distance_squared, target=np.array([-1.5, -1.5, 0.5])),
    )


# Various starting points (same as convergence tests)
initial_values_1: list[float] = [0.0, 0.0, 0.1]
initial_values_2: list[float] = [1.0, -1.0, 0.5]
initial_values_3: list[float] = [-0.5, 0.5, -0.2]


@pytest.fixture(name="optimizer_config")
def optimizer_config_fixture() -> dict[str, Any]:
    """Standard configuration for the optimization problem."""
    # This config is simplified but maintains the key parts from the original.
    return {
        "optimizer": {
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
    """Projects a point to be within the given lower and upper bounds."""
    lower_a = np.asarray(lower, dtype=np.float64)
    upper_a = np.asarray(upper, dtype=np.float64)
    return np.minimum(np.maximum(x, lower_a), upper_a)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.external
@pytest.mark.parametrize("initial_values", [initial_values_1, initial_values_2, initial_values_3])
def test_scipy_unconstrained_expected(
    optimizer_config: dict[str, Any],
    test_functions: list[_Function],
    initial_values: list[float],
) -> None:
    """SciPy optimizer should converge to analytical solution for unconstrained case."""
    objective_func = partial(
        _objective_function,
        functions=test_functions,
        weights=optimizer_config["objectives"]["weights"],
    )

    result = minimize(
        objective_func,
        x0=np.array(initial_values),
        method="BFGS",
        tol=optimizer_config["optimizer"]["tolerance"],
    )
    scipy_solution = result.x
    assert scipy_solution is not None
    np.testing.assert_allclose(scipy_solution, expected_unconstrained, rtol=1e-2, atol=1e-2)


@pytest.mark.external
@pytest.mark.parametrize("initial_values", [initial_values_1, initial_values_2, initial_values_3])
@pytest.mark.parametrize(
    ("lower_bounds", "upper_bounds"),
    [
        ([-1.0, -1.0, -1.0], [1.0, 1.0, 0.2]),
        ([0.1, 0.1, 0.1], [1.0, 1.0, 1.0]),
        ([-0.2, -0.2, 0.6], [0.2, 0.2, 1.0]),
        ([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]),
        ([0.0, 0.0, 0.0], [0.1, 0.1, 0.1]),
        ([-0.5, -0.5, 0.4], [0.5, 0.5, 0.6]),
        ([0.4, 0.4, 0.4], [0.6, 0.6, 0.6]),
    ],
)
def test_scipy_constrained_expected(
    optimizer_config: dict[str, Any],
    test_functions: list[_Function],
    initial_values: list[float],
    lower_bounds: list[float],
    upper_bounds: list[float],
) -> None:
    """SciPy optimizer should converge to projected analytical solution for constrained case."""
    objective_func = partial(
        _objective_function,
        functions=test_functions,
        weights=optimizer_config["objectives"]["weights"],
    )

    scipy_bounds = Bounds(lower_bounds, upper_bounds)

    # Note: The 'BFGS' method in SciPy does not support bounds. The 'L-BFGS-B'
    # method is used instead as it is a closely related algorithm that does.
    result = minimize(
        objective_func,
        x0=np.array(initial_values),
        method="L-BFGS-B",
        bounds=scipy_bounds,
        options={"ftol": optimizer_config["optimizer"]["tolerance"]},
    )
    scipy_solution = result.x
    assert scipy_solution is not None

    expected = _project_to_bounds(expected_unconstrained, lower_bounds, upper_bounds)
    np.testing.assert_allclose(scipy_solution, expected, rtol=1e-2, atol=1e-2)