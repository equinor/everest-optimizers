#!/usr/bin/env python3

import numpy as np

from everest_optimizers import pyoptpp


class Rosenbrock(pyoptpp.NLF1):
    """Rosenbrock function for optimization testing."""

    def __init__(self, ndim, x_init_np):
        super().__init__(ndim)
        init_vector = pyoptpp.SerialDenseVector(x_init_np)
        self.setX(init_vector)

    def evalF(self, x):
        """Evaluates the Rosenbrock function."""
        x_np = np.array(x.to_numpy(), copy=True)
        return sum(
            100.0 * (x_np[1:] - x_np[:-1] ** 2.0) ** 2.0 + (1 - x_np[:-1]) ** 2.0
        )

    def evalG(self, x):
        """Evaluates the gradient of the Rosenbrock function."""
        x_np = np.array(x.to_numpy(), copy=True)
        grad = np.zeros_like(x_np)
        xm = x_np[1:-1]
        xm_m1 = x_np[:-2]
        xm_p1 = x_np[2:]
        grad[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
        grad[0] = -400 * x_np[0] * (x_np[1] - x_np[0] ** 2) - 2 * (1 - x_np[0])
        grad[-1] = 200 * (x_np[-1] - x_np[-2] ** 2)
        return grad


def main():
    """Main function to run the optimization test."""
    print("--- Testing OPTPP Python Wrapper ---")

    # Create an instance of the Rosenbrock problem
    ndim = 2
    rosen_problem = Rosenbrock(ndim, np.array([-1.2, 1.0]))
    rosen_problem.setIsExpensive(True)

    # No update function needed for this simple case
    optimizer = pyoptpp.OptQNewton(rosen_problem)
    optimizer.setDebug()
    optimizer.setOutputFile("test_opttpp.out")
    optimizer.setTRSize(100.0)

    print("Running optimization...")
    optimizer.optimize()
    optimizer.printStatus("Optimization finished.")
    optimizer.cleanup()

    # Check results
    solution_vector = rosen_problem.getXc()
    solution_np = solution_vector.to_numpy()
    final_value = rosen_problem.getF()

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
