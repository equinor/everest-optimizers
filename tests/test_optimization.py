#!/usr/bin/env python3
"""Everest Optimizers Python bindings test.

Demonstrates working pybind11 bindings with:
- Python function optimization
- Quasi-Newton algorithm
- Dictionary result format
"""
# Everest Optimizers Python Bindings Test

import everest_optimizers
import pytest


# Test 1: Quadratic function
def test_optimization_quadratic() -> None:
    """Test optimization on a quadratic function."""

    def quadratic(x: float) -> float:
        return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

    x0 = [0.0, 0.0]
    result = everest_optimizers.optimize_python_func(quadratic, x0)

    x_sol = result["x"]
    fval = result["fun"]

    # Use pytest.approx for approximate equality checks
    assert x_sol[0] == pytest.approx(1.0, abs=1e-6)
    assert x_sol[1] == pytest.approx(2.0, abs=1e-6)
    assert fval == pytest.approx(0.0, abs=1e-12)


# Test 2: Rosenbrock function
def test_optimization_rosenbrock() -> None:
    """Test optimization on a rosenbrock function."""

    def rosenbrock(x: float) -> float:
        return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    x0 = [1.2, 1.2]
    result = everest_optimizers.optimize_python_func(rosenbrock, x0)

    assert "x" in result
    assert "fun" in result


# Test 3: Higher dimensional
def test_optimization_sphere() -> None:
    """Test optimization on a high-dimensional sphere function."""

    def sphere(x: float) -> float:
        return sum(xi**2 for xi in x)

    x0 = [2.0, -1.0, 3.0, -2.0, 1.0]
    result = everest_optimizers.optimize_python_func(sphere, x0)
    assert result["success"]
