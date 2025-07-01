"""Simple test for the everest_optimizers module."""

import everest_optimizers


def test_test_optpp() -> None:
    """Test import everest_optimizers."""
    assert (
        everest_optimizers.test_optpp()
        == "Everest Optimizers OptQNewton binding compilation successful!"
    )
