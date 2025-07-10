#!/usr/bin/env python3
# tests/OptQNewton/test_OptQNewton_expanded.py
"""
Test the expanded functionalities of the OptQNewton optimizer from the pyopttpp module.
Use pytest to run the tests.
"""

import sys
import os
import numpy as np
import pytest
import tempfile
import shutil

# Add the build directory to the Python path
pyopttpp_build_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src/",
        "dakota-packages",
        "OPTPP",
        "build",
        "python",
    )
)
if pyopttpp_build_dir not in sys.path:
    sys.path.insert(0, pyopttpp_build_dir)

# Try to import pyopttpp but handle the case where it's not available
pyopttpp_available = False
try:
    import pyopttpp

    pyopttpp_available = True
except ImportError:
    pass

# Helper class for Rosenbrock function
if pyopttpp_available:

    class Rosenbrock(pyopttpp.NLF1):
        """Rosenbrock function for optimization testing."""

        def __init__(self, ndim, x_init_np):
            super().__init__(ndim)
            init_vector = pyopttpp.SerialDenseVector(x_init_np)
            self.setX(init_vector)

        def evalF(self, x):
            """Evaluates the Rosenbrock function."""
            x_np = np.array(x.to_numpy(), copy=True)
            return sum(
                100.0 * (x_np[1:] - x_np[:-1] ** 2.0) ** 2.0 + (1 - x_np[:-1]) ** 2.0
            )

        def evalG(self, x):
            """Evaluates the gradient of the Rosenbrock function."""
            x_np = np.array(x.to_numpy(), copy=True)
            grad = np.zeros_like(x_np)
            xm = x_np[1:-1]
            xm_m1 = x_np[:-2]
            xm_p1 = x_np[2:]
            grad[1:-1] = (
                200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
            )
            grad[0] = -400 * x_np[0] * (x_np[1] - x_np[0] ** 2) - 2 * (1 - x_np[0])
            grad[-1] = 200 * (x_np[-1] - x_np[-2] ** 2)
            return grad


@pytest.mark.skipif(not pyopttpp_available, reason="pyopttpp module not available")
def test_optqnewton_linesearch():
    """Test OptQNewton with the LineSearch strategy on the Rosenbrock function."""
    ndim = 2
    rosen_problem = Rosenbrock(ndim, np.array([-1.2, 1.0]))

    optimizer = pyopttpp.OptQNewton(rosen_problem)
    optimizer.setSearchStrategy(pyopttpp.SearchStrategy.LineSearch)

    optimizer.optimize()
    optimizer.cleanup()

    solution_np = rosen_problem.getXc().to_numpy()
    expected_solution = np.array([1.0, 1.0])

    assert np.allclose(solution_np, expected_solution, rtol=1e-4), (
        f"LineSearch solution {solution_np} is not close to expected {expected_solution}"
    )


@pytest.mark.skipif(not pyopttpp_available, reason="pyopttpp module not available")
def test_optqnewton_trustregion():
    """Test OptQNewton with the TrustRegion strategy on the Rosenbrock function."""
    ndim = 2
    rosen_problem = Rosenbrock(ndim, np.array([-1.2, 1.0]))

    optimizer = pyopttpp.OptQNewton(rosen_problem)
    optimizer.setSearchStrategy(pyopttpp.SearchStrategy.TrustRegion)

    optimizer.optimize()
    optimizer.cleanup()

    solution_np = rosen_problem.getXc().to_numpy()
    expected_solution = np.array([1.0, 1.0])

    assert np.allclose(solution_np, expected_solution, rtol=1e-4), (
        f"TrustRegion solution {solution_np} is not close to expected {expected_solution}"
    )


@pytest.mark.skipif(not pyopttpp_available, reason="pyopttpp module not available")
def test_optqnewton_trustpds():
    """Test OptQNewton with the TrustPDS strategy."""
    ndim = 2
    rosen_problem = Rosenbrock(ndim, np.array([-1.2, 1.0]))

    optimizer = pyopttpp.OptQNewton(rosen_problem)
    optimizer.setSearchStrategy(pyopttpp.SearchStrategy.TrustPDS)

    optimizer.optimize()
    optimizer.cleanup()

    solution_np = rosen_problem.getXc().to_numpy()
    expected_solution = np.array([1.0, 1.0])

    assert np.allclose(solution_np, expected_solution, rtol=1e-4), (
        f"TrustPDS solution {solution_np} is not close to expected {expected_solution}"
    )


@pytest.mark.skipif(not pyopttpp_available, reason="pyopttpp module not available")
def test_optqnewton_output_file():
    """Test if OptQNewton can write to an output file."""
    ndim = 2
    rosen_problem = Rosenbrock(ndim, np.array([-1.2, 1.0]))

    optimizer = pyopttpp.OptQNewton(rosen_problem)

    # Create a temporary directory to store the output file
    temp_dir = tempfile.mkdtemp()
    output_file = os.path.join(temp_dir, "opt_output.txt")

    try:
        optimizer.setOutputFile(output_file, 0)
        optimizer.optimize()
        optimizer.cleanup()

        assert os.path.exists(output_file), "Output file was not created."
        with open(output_file, "r") as f:
            content = f.read()
            assert len(content) > 0, "Output file is empty."
            assert "OPT++" in content, "Output file does not contain expected content."
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


@pytest.mark.skipif(not pyopttpp_available, reason="pyopttpp module not available")
def test_optqnewton_set_debug():
    """Test the setDebug() method by redirecting output to a file."""
    ndim = 2
    rosen_problem = Rosenbrock(ndim, np.array([-1.2, 1.0]))
    optimizer = pyopttpp.OptQNewton(rosen_problem)

    temp_dir = tempfile.mkdtemp()
    output_file = os.path.join(temp_dir, "debug_output.txt")

    try:
        optimizer.setOutputFile(output_file, 0)
        optimizer.setDebug()
        optimizer.optimize()
        optimizer.cleanup()

        with open(output_file, "r") as f:
            content = f.read()
            assert "OPT++ version" in content
    finally:
        shutil.rmtree(temp_dir)


@pytest.mark.skipif(not pyopttpp_available, reason="pyopttpp module not available")
def test_optqnewton_runs_without_error():
    """Test that the optimizer can be invoked and cleaned up without error."""
    ndim = 2
    rosen_problem = Rosenbrock(ndim, np.array([-1.2, 1.0]))
    optimizer = pyopttpp.OptQNewton(rosen_problem)

    try:
        # The primary purpose of this test is to ensure that the optimize()
        # and cleanup() methods can be called without raising an exception.
        optimizer.optimize()
        optimizer.cleanup()
    except Exception as e:
        pytest.fail(f"OptQNewton raised an exception during execution: {e}")


@pytest.mark.skipif(not pyopttpp_available, reason="pyopttpp module not available")
def test_serial_dense_vector():
    """Test the functionalities of the SerialDenseVector class."""
    # Create a vector from a NumPy array
    np_array = np.array([1.0, 2.5, -3.0])
    vector = pyopttpp.SerialDenseVector(np_array)

    # Test length
    assert len(vector) == 3, f"Expected length 3, but got {len(vector)}"

    # Test __getitem__
    assert np.isclose(vector[1], 2.5), (
        f"Expected element 1 to be 2.5, but got {vector[1]}"
    )

    # Test __setitem__
    vector[1] = 9.9
    assert np.isclose(vector[1], 9.9), (
        f"Expected element 1 to be 9.9 after setitem, but got {vector[1]}"
    )

    # Test conversion back to NumPy
    modified_np_array = vector.to_numpy()
    expected_array = np.array([1.0, 9.9, -3.0])
    assert np.allclose(modified_np_array, expected_array), (
        f"Expected numpy array {expected_array}, but got {modified_np_array}"
    )


class RosenbrockWithHessian(pyopttpp.NLF1):
    """Rosenbrock function implementation that also provides the analytical Hessian."""

    def __init__(self, ndim, x_init):
        super().__init__(ndim)
        self.x_init = x_init
        self.setX(pyopttpp.SerialDenseVector(self.x_init))

    def evalF(self, x):
        return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

    def evalG(self, x):
        g = np.zeros(len(x))
        g[0] = -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0])
        g[1] = 200.0 * (x[1] - x[0] ** 2)
        return pyopttpp.SerialDenseVector(g)

    def evalH(self, x, H):
        H[0, 0] = 1200.0 * x[0] ** 2 - 400.0 * x[1] + 2.0
        H[0, 1] = -400.0 * x[0]
        H[1, 0] = -400.0 * x[0]
        H[1, 1] = 200.0
        return H


@pytest.mark.skipif(not pyopttpp_available, reason="pyopttpp module not available")
def test_optqnewton_with_hessian():
    """Test OptQNewton with a user-supplied analytical Hessian."""
    ndim = 2
    x_init = np.array([-1.2, 1.0])
    problem = RosenbrockWithHessian(ndim, x_init)
    optimizer = pyopttpp.OptQNewton(problem)

    optimizer.optimize()

    solution = problem.getXc().to_numpy()
    expected_solution = np.array([1.0, 1.0])

    assert np.allclose(solution, expected_solution, atol=1e-4), (
        f"Solution {solution} is not close to expected {expected_solution}"
    )

    optimizer.cleanup()
