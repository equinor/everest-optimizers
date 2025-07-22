from everest_optimizers import minimize
import numpy as np

def quad(x):
    return np.dot(x, x)  # f(x) = x^2

def grad_quad(x):
    return 2 * x  # grad f(x) = 2x

def test_conmin_quadratic_minimization():
    x0 = np.array([5.0, -3.0])
    res = minimize(quad, x0, method='conmin_mfd', jac=grad_quad)

    # Expected minimum is at x = [0.0, 0.0]
    expected_x = np.zeros_like(x0)
    expected_fun = 0.0

    assert res.success, f"Optimization failed: {res.message}"
    assert np.allclose(res.x, expected_x, atol=1e-5), f"x not close to zero: {res.x}"
    assert np.isclose(res.fun, expected_fun, atol=1e-8), f"Function value not close to 0: {res.fun}"
