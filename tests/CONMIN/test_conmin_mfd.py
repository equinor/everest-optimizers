import numpy as np
from everest_optimizers import minimize

def rosen_suzuki_obj(x):
    # OBJ = x1^2 - 5*x1 + x2^2 - 5*x2 + 2*x3^2 - 21*x3 + x4^2 + 7*x4 + 50
    return (
        x[0]**2 - 5*x[0] +
        x[1]**2 - 5*x[1] +
        2*x[2]**2 - 21*x[2] +
        x[3]**2 + 7*x[3] + 50
    )

def constraint1(x):
    # G1 = x1^2 + x1 + x2^2 - x2 + x3^2 + x3 + x4^2 - x4 - 8 <= 0
    return x[0]**2 + x[0] + x[1]**2 - x[1] + x[2]**2 + x[2] + x[3]**2 - x[3] - 8

def constraint2(x):
    # G2 = x1^2 - x1 + 2*x2^2 + x3^2 + 2*x4^2 - x4 - 10 <= 0
    return x[0]**2 - x[0] + 2*x[1]**2 + x[2]**2 + 2*x[3]**2 - x[3] - 10

def constraint3(x):
    # G3 = 2*x1^2 + 2*x1 + x2^2 - x2 + x3^2 - x4 - 5 <= 0
    return 2*x[0]**2 + 2*x[0] + x[1]**2 - x[1] + x[2]**2 - x[3] - 5

def test_conmin_rosen_suzuki():
    x0 = np.array([1.0, 1.0, 1.0, 1.0])  # initial guess

    constraints = [
        {'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3},
    ]

    res = minimize(
        rosen_suzuki_obj,
        x0,
        method='conmin_mfd',
        jac=None,
        constraints=constraints,
        options={'maxiter': 40, 'iprint': 2}
    )

    expected_x = np.array([0.0, 1.0, 2.0, -1.0])
    expected_fun = 6.0

    assert res.success, f"Optimization failed: {res.message}"
    assert np.allclose(res.x, expected_x, atol=1e-2), f"x not close to expected: {res.x}"
    assert np.isclose(res.fun, expected_fun, atol=1e-2), f"Function value not close to expected: {res.fun}"
