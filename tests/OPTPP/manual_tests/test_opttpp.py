#!/usr/bin/env python3

import numpy as np

from everest_optimizers import pyoptpp


def _rosenbrock_value(x_np: np.ndarray) -> float:
    """Rosenbrock function value for a numpy array input."""
    return float(
        np.sum(100.0 * (x_np[1:] - x_np[:-1] ** 2.0) ** 2.0 + (1 - x_np[:-1]) ** 2.0)
    )


def _rosenbrock_grad(x_np: np.ndarray) -> np.ndarray:
    """Gradient of the Rosenbrock function for a numpy array input."""
    grad = np.zeros_like(x_np)
    if x_np.size == 2:
        # Analytical 2D gradient
        grad[0] = -400 * x_np[0] * (x_np[1] - x_np[0] ** 2) - 2 * (1 - x_np[0])
        grad[1] = 200 * (x_np[1] - x_np[0] ** 2)
    else:
        xm = x_np[1:-1]
        xm_m1 = x_np[:-2]
        xm_p1 = x_np[2:]
        grad[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
        grad[0] = -400 * x_np[0] * (x_np[1] - x_np[0] ** 2) - 2 * (1 - x_np[0])
        grad[-1] = 200 * (x_np[-1] - x_np[-2] ** 2)
    return grad


def main():
    """Main function to run the optimization test."""
    print("--- Testing OPTPP Python Wrapper ---")

    # Create an instance of the Rosenbrock problem
    ndim = 2
    x0_np = np.array([-1.2, 1.0])

    # Create Python callbacks for the objective and gradient following the
    # _OptQNewtonProblem pattern used in the library (NLF1.create factory).
    def eval_f(x):
        x_np = np.array(x.to_numpy(), copy=True)
        return _rosenbrock_value(x_np)

    def eval_g(x):
        x_np = np.array(x.to_numpy(), copy=True)
        return _rosenbrock_grad(x_np)

    x0_vector = pyoptpp.SerialDenseVector(x0_np)
    nlf1_obj = pyoptpp.NLF1.create(ndim, eval_f, eval_g, x0_vector)

    # No update function needed for this simple case
    optimizer = pyoptpp.OptQNewton(nlf1_obj)
    optimizer.setDebug()
    optimizer.setOutputFile("test_opttpp.out")
    optimizer.setTRSize(100.0)

    print("Running optimization...")
    optimizer.optimize()
    optimizer.printStatus("Optimization finished.")
    optimizer.cleanup()

    # Check results
    solution_vector = nlf1_obj.getXc()
    solution_np = solution_vector.to_numpy()
    final_value = nlf1_obj.getF()

    print(f"Solution: {solution_np}")
    print(f"Function value: {final_value}")

    # The known solution is (1, 1)
    expected_solution = np.array([1.0, 1.0])
    assert np.allclose(solution_np, expected_solution), (
        f"Test failed: Solution {solution_np} is not close to {expected_solution}"
    )

    print("\n--- Test Passed! ---")


if __name__ == "__main__":
    main()
