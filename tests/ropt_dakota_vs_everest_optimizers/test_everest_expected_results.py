"""Tests for expected solutions using everest_optimizers.minimize.

This mirrors `test_optimizer_convergence.py` but validates that the
*everest-optimizers* solver itself converges to analytically expected
solutions for both unconstrained and constrained problems. It does **not**
compare against *ropt*; instead, it asserts against hard-coded expected
results.
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Any
import os
import sys

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import Bounds

# Add the source directory to the path to find everest_optimizers
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from everest_optimizers import minimize  # noqa: E402  pylint: disable=C0413

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

_Function = Callable[[NDArray[np.float64]], float]


def _compute_distance_squared(
    variables: NDArray[np.float64], target: NDArray[np.float64]
) -> float:
    return float(((variables - target) ** 2).sum())


# Objective function combines two distance-squared metrics with weights.
weights = np.array([0.75, 0.25], dtype=np.float64)
function_1: _Function = partial(_compute_distance_squared, target=np.array([0.5, 0.5, 0.5]))
function_2: _Function = partial(_compute_distance_squared, target=np.array([-1.5, -1.5, 0.5]))


def objective(x: NDArray[np.float64]) -> float:
    return weights[0] * function_1(x) + weights[1] * function_2(x)


# -----------------------------------------------------------------------------
# Expected analytical solutions
# -----------------------------------------------------------------------------

expected_unconstrained = np.array([0.0, 0.0, 0.5], dtype=np.float64)


def _project_to_bounds(
    x: NDArray[np.float64], lower: list[float] | NDArray[np.float64], upper: list[float] | NDArray[np.float64]
) -> NDArray[np.float64]:
    lower_a = np.asarray(lower, dtype=np.float64)
    upper_a = np.asarray(upper, dtype=np.float64)
    return np.minimum(np.maximum(x, lower_a), upper_a)


# -----------------------------------------------------------------------------
# Test data
# -----------------------------------------------------------------------------

initial_values_1: list[float] = [0.0, 0.0, 0.1]
initial_values_2: list[float] = [1.0, -1.0, 0.5]
initial_values_3: list[float] = [-0.5, 0.5, -0.2]
initial_values_4: list[float] = [2.0, 2.0, 2.0]
initial_values_5: list[float] = [-2.0, -2.0, -2.0]


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("initial_values", [initial_values_1, initial_values_2, initial_values_3, initial_values_4, initial_values_5])
def test_everest_unconstrained_expected(initial_values: list[float]) -> None:
    """everest-optimizers should match analytical solution in unconstrained case."""
    res = minimize(objective, initial_values, method="optpp_q_newton")
    np.testing.assert_allclose(res.x, expected_unconstrained, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("initial_values", [initial_values_1, initial_values_2, initial_values_3, initial_values_4, initial_values_5])
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
        ([-1.0, -1.0, 0.45], [1.0, 1.0, 0.55]),
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ([-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]),
    ],
)
@pytest.mark.skip
def test_everest_constrained_expected(
    initial_values: list[float],
    lower_bounds: list[float],
    upper_bounds: list[float],
) -> None:
    """everest-optimizers should match projected analytical solution in constrained case."""
    bounds = Bounds(lower_bounds, upper_bounds)
    res = minimize(objective, initial_values, method="optpp_constr_q_newton", bounds=bounds)
    expected = _project_to_bounds(expected_unconstrained, lower_bounds, upper_bounds)
    np.testing.assert_allclose(res.x, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    ("initial_values", "lower_bounds", "upper_bounds"),
    [
        ([0.0, 0.0, 0.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 0.2]),
        ([0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0]),
        ([0.0, 0.0, 0.8], [-0.2, -0.2, 0.6], [0.2, 0.2, 1.0]),
        ([0.0, 0.0, 0.0], [-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]),
        ([0.05, 0.05, 0.05], [0.0, 0.0, 0.0], [0.1, 0.1, 0.1]),
        ([0.0, 0.0, 0.5], [-0.5, -0.5, 0.4], [0.5, 0.5, 0.6]),
        ([0.5, 0.5, 0.5], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]),
        ([0.0, 0.0, 0.5], [-1.0, -1.0, 0.45], [1.0, 1.0, 0.55]),
        ([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ([0.0, 0.0, 0.0], [-0.1, -0.1, -0.1], [0.1, 0.1, 0.1]),
    ],
)
def test_everest_constrained_expected_feasible_start(
    initial_values: list[float],
    lower_bounds: list[float],
    upper_bounds: list[float],
) -> None:
    """everest-optimizers should match projected analytical solution in constrained case with feasible start."""
    # Make sure initial point is feasible
    for i, val in enumerate(initial_values):
        assert lower_bounds[i] <= val <= upper_bounds[i]

    bounds = Bounds(lower_bounds, upper_bounds)
    res = minimize(objective, initial_values, method="optpp_constr_q_newton", bounds=bounds)
    expected = _project_to_bounds(expected_unconstrained, lower_bounds, upper_bounds)
    np.testing.assert_allclose(res.x, expected, rtol=1e-1, atol=1e-1)
