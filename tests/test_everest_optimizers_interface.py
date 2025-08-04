#!/usr/bin/env python3
# tests/OptQNewton/test_everest_optimizers_interface.py

import pytest
import numpy as np
import sys
import os

# Add the source directory to the path
src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add pyopttpp path
pyopttpp_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "dakota-packages", "OPTPP", "build", "python"
)
if pyopttpp_path not in sys.path:
    sys.path.insert(0, pyopttpp_path)


class TestEverestOptimizersInterface:
    """Test the everest_optimizers interface and compatibility."""

    def test_package_import(self):
        """Test that everest_optimizers can be imported."""
        import everest_optimizers

        assert hasattr(everest_optimizers, "minimize")
        assert everest_optimizers.__version__ == "0.1.0"
        assert everest_optimizers.__all__ == ["minimize"]

    def test_minimize_function_signature(self):
        """Test that minimize function has correct signature."""
        from everest_optimizers import minimize
        import inspect

        sig = inspect.signature(minimize)
        expected_params = [
            "fun",
            "x0",
            "args",
            "method",
            "jac",
            "hess",
            "hessp",
            "bounds",
            "constraints",
            "tol",
            "callback",
            "options",
        ]

        actual_params = list(sig.parameters.keys())
        assert actual_params == expected_params

        # Check default values
        assert sig.parameters["args"].default == ()
        assert sig.parameters["method"].default == "optpp_q_newton"
        assert sig.parameters["jac"].default is None
        assert sig.parameters["options"].default is None

    def test_scipy_compatibility(self):
        """Test that the interface matches scipy.optimize.minimize."""
        from everest_optimizers import minimize
        from scipy.optimize import minimize as scipy_minimize
        import inspect

        everest_sig = inspect.signature(minimize)
        scipy_sig = inspect.signature(scipy_minimize)

        # Check that all scipy parameters are present
        scipy_params = set(scipy_sig.parameters.keys())
        everest_params = set(everest_sig.parameters.keys())

        assert scipy_params.issubset(everest_params), "Missing scipy parameters"

    def test_unsupported_method_error(self):
        """Test that unsupported methods raise appropriate errors."""
        from everest_optimizers import minimize

        def dummy_func(x):
            return x[0] ** 2

        with pytest.raises(ValueError, match="Unknown method"):
            minimize(dummy_func, [1.0], method="UnsupportedMethod")

    def test_unsupported_features_error(self):
        """Test that unsupported features raise appropriate errors."""
        from everest_optimizers import minimize

        def dummy_func(x):
            return x[0] ** 2

        # Test bounds
        with pytest.raises(
            NotImplementedError, match="optpp_q_newton does not support bounds"
        ):
            minimize(dummy_func, [1.0], method="optpp_q_newton", bounds=[(0, 1)])

        # Test constraints
        with pytest.raises(
            NotImplementedError, match="optpp_q_newton does not support constraints"
        ):
            minimize(
                dummy_func,
                [1.0],
                method="optpp_q_newton",
                constraints={"type": "eq", "fun": lambda x: x[0]},
            )

        # Test callback
        with pytest.raises(
            NotImplementedError,
            match="Callback function not implemented for optpp_q_newton",
        ):
            minimize(
                dummy_func, [1.0], method="optpp_q_newton", callback=lambda x: None
            )

    def test_input_validation(self):
        """Test input validation."""
        from everest_optimizers import minimize

        def dummy_func(x):
            return x[0] ** 2

        # Test 2D x0
        with pytest.raises(ValueError, match="x0 must be 1-dimensional"):
            minimize(dummy_func, [[1.0, 2.0]], method="optpp_q_newton")

    def test_return_type(self):
        """Test that minimize returns OptimizeResult."""
        from everest_optimizers import minimize
        from scipy.optimize import OptimizeResult

        def quadratic(x):
            return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

        result = minimize(quadratic, [0.0, 0.0], method="optpp_q_newton")

        assert isinstance(result, OptimizeResult)
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert hasattr(result, "nfev")
        assert hasattr(result, "njev")

    def test_case_insensitive_method(self):
        """Test that method parameter is case insensitive."""
        from everest_optimizers import minimize

        def quadratic(x):
            return x[0] ** 2 + x[1] ** 2

        # Should work with different cases
        result1 = minimize(quadratic, [1.0, 1.0], method="Optpp_q_newton")
        result2 = minimize(quadratic, [1.0, 1.0], method="optpp_q_newton")
        result3 = minimize(quadratic, [1.0, 1.0], method="OptPP_Q_Newton")

        assert result1.success
        assert result2.success
        assert result3.success

    def test_args_parameter(self):
        """Test that args parameter works correctly."""
        from everest_optimizers import minimize

        def func_with_args(x, a, b):
            return a * (x[0] - 1) ** 2 + b * (x[1] - 2) ** 2

        result = minimize(
            func_with_args, [0.0, 0.0], args=(2, 3), method="optpp_q_newton"
        )

        assert result.success
        assert np.allclose(result.x, [1.0, 2.0], rtol=1e-3)

    def test_list_input_conversion(self):
        """Test that list inputs are converted to numpy arrays."""
        from everest_optimizers import minimize

        def quadratic(x):
            return (x[0] - 1) ** 2 + (x[1] - 2) ** 2

        # Test with list input
        result = minimize(quadratic, [0.0, 0.0], method="optpp_q_newton")

        assert result.success
        assert isinstance(result.x, np.ndarray)
        assert np.allclose(result.x, [1.0, 2.0], rtol=1e-3)
