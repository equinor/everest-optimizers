from everest_optimizers import minimize
import numpy as np

def quad(x):
    return np.dot(x, x)  # f(x) = x^2

def grad_quad(x):
    return 2 * x  # grad f(x) = 2x
  
def constraint_eq(x):
    return x[0] + x[1] - 1  # equals 0 when constraint is satisfied

def jac_constraint_eq(x):
    return np.array([1.0, 1.0])  # gradient of constraint

def test_conmin_quadratic_minimization():
    x0 = np.array([5.0, -3.0])
    bounds = [(-10, 10), (-10, 10)]  # bounds for x0 and x1
    constraints = [{'type': 'eq', 'fun': constraint_eq, 'jac': jac_constraint_eq}]
    
    res = minimize(quad, x0, method='conmin_mfd', jac=grad_quad, bounds=bounds, constraints=constraints)

    # Expected minimum is at x = [0.0, 0.0]
    expected_x = np.zeros_like(x0)
    expected_fun = 0.0

    assert res.success, f"Optimization failed: {res.message}"
    assert np.allclose(res.x, expected_x, atol=1e-5), f"x not close to zero: {res.x}"
    assert np.isclose(res.fun, expected_fun, atol=1e-8), f"Function value not close to 0: {res.fun}"
