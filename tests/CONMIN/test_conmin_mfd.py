from everest_optimizers.minimize import minimize
import numpy as np
import pytest

def test_rosen_suzuki_constraints_satisfied():
    
    def obj(x):
        return (
            x[0]**2 - 5*x[0] +
            x[1]**2 - 5*x[1] +
            2*x[2]**2 - 21*x[2] +
            x[3]**2 + 7*x[3] + 50
        )

    def constraint1(x): return x[0]**2 + x[0] + x[1]**2 - x[1] + x[2]**2 + x[2] + x[3]**2 - x[3] - 8
    def constraint2(x): return x[0]**2 - x[0] + 2*x[1]**2 + x[2]**2 + 2*x[3]**2 - x[3] - 10
    def constraint3(x): return 2*x[0]**2 + 2*x[0] + x[1]**2 - x[1] + x[2]**2 - x[3] - 5

    x0 = np.array([1.0, 1.0, 1.0, 1.0])

    constraints = [
        {"type": "ineq", "fun": constraint1},
        {"type": "ineq", "fun": constraint2},
        {"type": "ineq", "fun": constraint3},
    ]

    bounds = [(-10.0, 10.0)] * 4

    result = minimize(
        fun=obj,
        x0=x0,
        method="conmin_mfd",
        bounds=bounds,
        constraints=constraints,
        options={"ITMAX": 100}
    )

    # Expected solution
    expected_x = np.array([0.0, 1.0, 2.0, -1.0])
    expected_fun = 6.0

    assert result.success
    assert np.allclose(result.x, expected_x, atol=1e-2)
    assert np.isclose(result.fun, expected_fun, atol=1e-2)

@pytest.mark.parametrize("x0", [
    np.array([-4.0, -4.0, -4.0, -4.0]),
    np.array([4.0, 4.0, 4.0, 4.0]),
    np.array([0.0, 0.0, 0.0, 0.0]),
    np.array([0.1, 0.9, 2.1, -1.2])
])
def test_rosen_suzuki_multiple_initial_guesses(x0):
    def obj(x):
        return (
            x[0]**2 - 5*x[0] +
            x[1]**2 - 5*x[1] +
            2*x[2]**2 - 21*x[2] +
            x[3]**2 + 7*x[3] + 50
        )

    def constraint1(x): return x[0]**2 + x[0] + x[1]**2 - x[1] + x[2]**2 + x[2] + x[3]**2 - x[3] - 8
    def constraint2(x): return x[0]**2 - x[0] + 2*x[1]**2 + x[2]**2 + 2*x[3]**2 - x[3] - 10
    def constraint3(x): return 2*x[0]**2 + 2*x[0] + x[1]**2 - x[1] + x[2]**2 - x[3] - 5

    constraints = [
        {"type": "ineq", "fun": constraint1},
        {"type": "ineq", "fun": constraint2},
        {"type": "ineq", "fun": constraint3},
    ]
    bounds = [(-10.0, 10.0)] * 4

    result = minimize(
        fun=obj,
        x0=x0,
        method="conmin_mfd",
        bounds=bounds,
        constraints=constraints,
        options={"ITMAX": 100}
    )

    expected_x = np.array([0.0, 1.0, 2.0, -1.0])
    expected_fun = 6.0

    assert result.success
    assert np.allclose(result.x, expected_x, atol=1e-2)
    assert np.isclose(result.fun, expected_fun, atol=1e-2)