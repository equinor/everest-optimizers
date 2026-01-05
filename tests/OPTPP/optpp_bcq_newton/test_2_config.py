"""Test suite for everest_optimizers.minimize() with method='optpp_bcq_newton'

Testing the OptQNIPS (Quasi-Newton Interior-Point Solver) method from everest_optimizers.minimize().
In Dakota OPTPP this optimization algorithm is referred to as OptQNIPS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import Bounds

from everest_optimizers import minimize


# --- Fixed Problem Definition ---
def objective(x: NDArray[np.float64]) -> float:
    return (x[0] - 2.0) ** 2 + (x[1] + 1.0) ** 2


def objective_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([2 * (x[0] - 2.0), 2 * (x[1] + 1.0)])


X0 = np.array([0.0, 0.0])
BOUNDS = Bounds([0, -np.inf], [np.inf, np.inf])
EXPECTED_SOLUTION = np.array([2.0, -1.0])

# --- Tests for different options ---


@pytest.mark.parametrize("search_method", ["line_search", "trust_pds"])
def test_search_strategy_options(tmp_path: Path, monkeypatch: Any, search_method: str):
    """Test that the optimizer runs with different search strategy settings."""
    monkeypatch.chdir(tmp_path)
    options = {"search_method": search_method}
    result = minimize(
        objective,
        X0,
        method="optpp_bcq_newton",
        jac=objective_grad,
        bounds=BOUNDS,  # type: ignore[arg-type]
        options=options,
    )
    assert result.success
    np.testing.assert_allclose(result.x, EXPECTED_SOLUTION, rtol=1e-4, atol=1e-4)


def test_search_strategy_options_trust_region():
    """Test that the trust-region method is not supported."""
    options = {"search_method": "trust_region"}
    with pytest.raises(
        ValueError,
        match="OptBCQNewton does not support the 'trust_region' search method",
    ):
        minimize(
            objective,
            X0,
            method="optpp_bcq_newton",
            jac=objective_grad,
            bounds=BOUNDS,  # type: ignore[arg-type]
            options=options,
        )


@pytest.mark.parametrize("tolerance", [1e-4, 1e-6, 1e-8])
def test_convergence_tolerance_options(tolerance: float):
    """Test that the optimizer runs with different convergence tolerance settings."""
    options = {"convergence_tolerance": tolerance, "max_iterations": 100000}
    result = minimize(
        objective,
        X0,
        method="optpp_bcq_newton",
        jac=objective_grad,
        bounds=BOUNDS,  # type: ignore[arg-type]
        options=options,
    )
    assert result.success
    np.testing.assert_allclose(result.x, EXPECTED_SOLUTION, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "tolerance",
    [
        1,
        1e-2,
        1e-4,
        1e-6,
        1e-8,
        1e-10,
    ],
)  # TODO: investigate if this tolerance parameter is handled correctly
def test_gradient_tolerance_options(tolerance: float):
    """Test that the optimizer runs with different gradient tolerance settings."""
    options = {"gradient_tolerance": tolerance, "max_iterations": 10000}
    result = minimize(
        objective,
        X0,
        method="optpp_bcq_newton",
        jac=objective_grad,
        bounds=BOUNDS,  # type: ignore[arg-type]
        options=options,
    )
    assert result.success
    np.testing.assert_allclose(result.x, EXPECTED_SOLUTION, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("max_iterations", [10, 100, 1000])
def test_max_iterations_option(max_iterations: int):
    """Test that the optimizer respects the max_iterations setting."""
    options = {"max_iterations": max_iterations}
    result = minimize(
        objective,
        X0,
        method="optpp_bcq_newton",
        jac=objective_grad,
        bounds=BOUNDS,  # type: ignore[arg-type]
        options=options,
    )
    assert result.success
    np.testing.assert_allclose(result.x, EXPECTED_SOLUTION, rtol=1e-3, atol=1e-3)


def test_too_low_max_iterations():
    """Too low max_iterations should mean it numerically does not converge"""
    options = {"max_iterations": 1}
    result = minimize(
        objective,
        X0,
        method="optpp_bcq_newton",
        jac=objective_grad,
        bounds=BOUNDS,  # type: ignore[arg-type]
        options=options,
    )
    assert result.success  # algorithm terminates successfully
    assert not np.allclose(
        result.x, EXPECTED_SOLUTION, rtol=1e-3, atol=1e-3
    )  # but fails to converge to the expected solution


@pytest.mark.parametrize("debug_flag", [True, False])
def test_debug_option(debug_flag: bool):
    """Test that the optimizer runs with different debug flag settings."""
    options = {"debug": debug_flag}
    result = minimize(
        objective,
        X0,
        method="optpp_bcq_newton",
        jac=objective_grad,
        bounds=BOUNDS,  # type: ignore[arg-type]
        options=options,
    )
    assert result.success
    np.testing.assert_allclose(result.x, EXPECTED_SOLUTION, rtol=1e-4, atol=1e-4)


def test_output_file_no_default_file(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    assert not Path("OPT_DEFAULT.out").exists()
    minimize(
        objective,
        X0,
        bounds=BOUNDS,  # type: ignore[arg-type]
        method="optpp_bcq_newton",
        jac=objective_grad,
    )
    assert not Path("OPT_DEFAULT.out").exists()


def test_output_file_exists(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    assert not Path("foo.out").exists()
    minimize(
        objective,
        X0,
        bounds=BOUNDS,  # type: ignore[arg-type]
        method="optpp_bcq_newton",
        jac=objective_grad,
        options={"output_file": "foo.out"},
    )
    assert Path("foo.out").exists()
    assert not Path("OPT_DEFAULT.out").exists()
