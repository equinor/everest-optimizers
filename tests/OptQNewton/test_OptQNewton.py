#!/usr/bin/env python3
# tests/OptQNewton/test_OptQNewton.py
"""
Test the OptQNewton optimizer from the pyopttpp module.
Use pytest to run the tests.
"""

import sys
import os
import numpy as np
import pytest

# Add the build directory to the Python path
# This allows importing the pyopttpp module from its build location
pyopttpp_build_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "dakota-packages",
        "OPTPP",
        "build",
        "python",
    )
)
sys.path.insert(0, pyopttpp_build_dir)

# Try to import pyopttpp but don't fail if not available
# This allows pytest to at least collect the test
pyopttpp_available = False
try:
    import pyopttpp

    pyopttpp_available = True
except ImportError:
    pass  # The error message will be shown when the test is run


def test_pyopttpp_import():
    """Test if the pyopttpp module can be imported successfully."""
    if not pyopttpp_available:
        # Provide detailed error messages
        print(
            f"Could not import pyopttpp from {pyopttpp_build_dir}. Make sure the module is built."
        )

        # Check if the file exists and Python version
        so_file_found = False
        py_version_mismatch = False
        current_py_version = f"{sys.version_info.major}{sys.version_info.minor}"

        if os.path.exists(pyopttpp_build_dir):
            for f in os.listdir(pyopttpp_build_dir):
                if f.startswith("pyopttpp") and f.endswith(".so"):
                    print(f"Found shared object file: {f}")
                    so_file_found = True
                    # Check if the file is for a different Python version
                    if f"cpython-{current_py_version}" not in f:
                        py_version_mismatch = True
                        print(
                            "WARNING: Python version mismatch. Module built for a different Python version."
                        )
                        print(
                            f"Current Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                        )
                        print(
                            "Module appears to be built for a different Python version."
                        )
                    break

        if not so_file_found:
            print(
                "Could not find the pyopttpp shared object file in the build directory."
            )
            print(
                "Make sure you've built the module according to the instructions in "
                "dakota-packages/OPTPP/python/README.md"
            )

        if py_version_mismatch:
            print(
                "\nSOLUTION: Run pytest with the same Python version that the module was built for."
            )
            print("Example: python3.11 -m pytest tests/OptQNewton/test_OptQNewton.py")

        pytest.skip("pyopttpp module not available")

    # If we get here, the import was successful
    assert pyopttpp_available, "pyopttpp module should be available"


def test_optqnewton_rosenbrock():
    """Test OptQNewton optimizer with the Rosenbrock function."""
    # Skip test if pyopttpp import failed
    if not pyopttpp_available:
        pytest.skip("pyopttpp module not available")

    # Define Rosenbrock class inside the test to avoid import errors during collection
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

    # Create an instance of the Rosenbrock problem
    ndim = 2
    rosen_problem = Rosenbrock(ndim, np.array([-1.2, 1.0]))
    rosen_problem.setIsExpensive(True)

    # Create the optimizer
    optimizer = pyopttpp.OptQNewton(rosen_problem)
    optimizer.setTRSize(100.0)

    # Run the optimization
    optimizer.optimize()
    optimizer.cleanup()

    # Get results
    solution_vector = rosen_problem.getXc()
    solution_np = solution_vector.to_numpy()
    final_value = rosen_problem.getF()

    # The known solution for Rosenbrock is (1, 1)
    expected_solution = np.array([1.0, 1.0])
    assert np.allclose(solution_np, expected_solution, rtol=1e-4), (
        f"Solution {solution_np} is not close to expected {expected_solution}"
    )

    print(f"Final value: {final_value}")
    print(f"Solution: {solution_np}")
    print("Test passed!")


if __name__ == "__main__":
    test_pyopttpp_import()
    if pyopttpp_available:
        test_optqnewton_rosenbrock()
