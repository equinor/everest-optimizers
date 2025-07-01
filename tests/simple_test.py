import everest_optimizers


def test_test_optpp() -> None:
    assert (
        everest_optimizers.test_optpp()
        == "Everest Optimizers OptQNewton binding compilation successful!"
    )
