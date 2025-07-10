"""Simple test for the everest_optimizers module."""

import everest_optimizers
import pyopttpp


def test_test_optpp() -> None:
    """Test import everest_optimizers."""
    assert (
        everest_optimizers.test_optpp()
        == "Everest Optimizers OptQNewton binding compilation successful!"
    )


def test_pyopttpp_import() -> None:
    assert hasattr(pyopttpp, "SerialDenseVector")


def test_pyopttpp_serial_dense_vector() -> None:
    # Create a SerialDenseVector of size 5
    vec = pyopttpp.SerialDenseVector(5)
    assert len(vec) == 5
    # Check setting and getting an item
    vec[0] = 3.14
    assert abs(vec[0] - 3.14) < 1e-12
