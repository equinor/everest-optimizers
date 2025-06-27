import everest_optimizers


def test_add():
    assert 1 + 1 == 2


def test_test_optpp():
    assert (
        everest_optimizers.test_optpp()
        == "Everest Optimizers OptQNewton binding compilation successful!"
    )
