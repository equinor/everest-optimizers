#!/usr/bin/env python3
# tests/OptQNewton/compare_problems.py
"""
Compare the performance of OptQNewton on different problem formulations.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the build directory to the Python path
pyopttpp_build_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..',
    'dakota-packages', 'OPTPP', 'build', 'python'
))
if pyopttpp_build_dir not in sys.path:
    sys.path.insert(0, pyopttpp_build_dir)

# Try to import pyopttpp but handle the case where it's not available
pyopttpp_available = False
try:
    import pyopttpp
    pyopttpp_available = True
except ImportError:
    logging.warning("pyopttpp module not available, skipping tests.")

# Helper classes for Rosenbrock function
if pyopttpp_available:
    class Rosenbrock(pyopttpp.NLF1):
        """Standard Rosenbrock function implementation."""
        def __init__(self, ndim, x_init_np):
            super().__init__(ndim)
            self.path = [x_init_np]
            self.setX(pyopttpp.SerialDenseVector(x_init_np))

        def evalF(self, x):
            x_np = np.array(x.to_numpy(), copy=True)
            self.path.append(x_np)
            return sum(100.0 * (x_np[1:] - x_np[:-1]**2.0)**2.0 + (1 - x_np[:-1])**2.0)

        def evalG(self, x):
            x_np = np.array(x.to_numpy(), copy=True)
            grad = np.zeros_like(x_np)
            xm = x_np[1:-1]
            xm_m1 = x_np[:-2]
            xm_p1 = x_np[2:]
            grad[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
            grad[0] = -400 * x_np[0] * (x_np[1] - x_np[0]**2) - 2 * (1 - x_np[0])
            grad[-1] = 200 * (x_np[-1] - x_np[-2]**2)
            return grad

    class RosenbrockWithHessian(Rosenbrock):
        """Rosenbrock function that also provides the analytical Hessian."""
        def evalH(self, x, H):
            x_np = np.array(x.to_numpy(), copy=True)
            H[0, 0] = 1200.0 * x_np[0]**2 - 400.0 * x_np[1] + 2.0
            H[0, 1] = -400.0 * x_np[0]
            H[1, 0] = -400.0 * x_np[0]
            H[1, 1] = 200.0
            return H

def run_optimization(problem):
    """Run optimization for a given problem."""
    optimizer = pyopttpp.OptQNewton(problem)
    optimizer.setOutputFile(os.devnull) # Suppress output file
    optimizer.optimize()
    optimizer.cleanup()
    solution_np = problem.getXc().to_numpy()
    path = problem.path
    return solution_np, path

def compare_problem_setups():
    """Compare the optimizer performance with different setups."""
    if not pyopttpp_available:
        return

    start_point_1 = np.array([-1.2, 1.0])
    start_point_2 = np.array([2.0, 2.0])

    problems = {
        f"Standard (start={start_point_1})": Rosenbrock(2, start_point_1),
        f"With Hessian (start={start_point_1})": RosenbrockWithHessian(2, start_point_1),
        f"Standard (start={start_point_2})": Rosenbrock(2, start_point_2),
    }
    
    results = {}
    for name, problem in problems.items():
        logging.info(f"Running: {name}")
        try:
            solution, path = run_optimization(problem)
            results[name] = {'solution': solution, 'path': path}
            logging.info(f"  -> Solution: {solution}")
        except Exception as e:
            logging.error(f"  -> FAILED with error: {e}")

    # Plotting
    plt.figure(figsize=(12, 9))
    plt.title('Comparison of OptQNewton Setups')
    
    # Rosenbrock function contour
    x_range = np.linspace(-2.5, 2.5, 400)
    y_range = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x_range, y_range)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2
    plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='gray', alpha=0.5)
    
    colors = ['r', 'g', 'b']
    for i, (name, result) in enumerate(results.items()):
        path = np.array(result['path'])
        plt.plot(path[:, 0], path[:, 1], 'o-', label=name, alpha=0.7, color=colors[i])
        start_point = path[0]
        plt.plot(start_point[0], start_point[1], 'X', color=colors[i], markersize=12, label=f'_start_{name}')

    plt.plot(1, 1, 'm*', markersize=20, label='Global Optimum')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.savefig('problem_setups_comparison.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    compare_problem_setups()
