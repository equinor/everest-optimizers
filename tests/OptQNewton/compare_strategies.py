#!/usr/bin/env python3
# tests/OptQNewton/compare_strategies.py
"""
Compare the performance of different search strategies for OptQNewton.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import tempfile

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

# Helper class for Rosenbrock function
if pyopttpp_available:
    class Rosenbrock(pyopttpp.NLF1):
        """Rosenbrock function for optimization testing."""
        def __init__(self, ndim, x_init_np):
            super().__init__(ndim)
            self.path = [x_init_np]
            init_vector = pyopttpp.SerialDenseVector(x_init_np)
            self.setX(init_vector)

        def evalF(self, x):
            """Evaluates the Rosenbrock function."""
            x_np = np.array(x.to_numpy(), copy=True)
            self.path.append(x_np)
            return sum(100.0 * (x_np[1:] - x_np[:-1]**2.0)**2.0 + (1 - x_np[:-1])**2.0)

        def evalG(self, x):
            """Evaluates the gradient of the Rosenbrock function."""
            x_np = np.array(x.to_numpy(), copy=True)
            grad = np.zeros_like(x_np)
            xm = x_np[1:-1]
            xm_m1 = x_np[:-2]
            xm_p1 = x_np[2:]
            grad[1:-1] = 200 * (xm - xm_m1**2) - 400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm)
            grad[0] = -400 * x_np[0] * (x_np[1] - x_np[0]**2) - 2 * (1 - x_np[0])
            grad[-1] = 200 * (x_np[-1] - x_np[-2]**2)
            return grad

def run_optimization(strategy, start_point):
    """Run optimization for a given strategy and starting point."""
    ndim = len(start_point)
    rosen_problem = Rosenbrock(ndim, start_point)
    
    optimizer = pyopttpp.OptQNewton(rosen_problem)
    optimizer.setSearchStrategy(strategy)
    
    # Create a temporary file to capture the output
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log') as tmp_file:
        log_filename = tmp_file.name

    optimizer.setOutputFile(log_filename)
    optimizer.setDebug()

    optimizer.optimize()
    optimizer.cleanup()

    # Parse the output file for metrics
    iterations = -1
    func_evals = -1
    try:
        with open(log_filename, 'r') as f:
            lines = f.readlines()
            # The final summary line appears after 'checkConvg'
            for i, line in enumerate(reversed(lines)):
                if 'checkConvg:' in line:
                    # The summary line is a few lines before this, starting with iteration count
                    summary_line_index = len(lines) - i - 2
                    if summary_line_index >= 0:
                        summary_line = lines[summary_line_index]
                        parts = summary_line.split()
                        if len(parts) >= 2:
                            iterations = int(parts[0])
                            # For LineSearch, function evals are on the same summary line
                            if len(parts) >= 4:
                                try:
                                    # Handle both integer and hex float strings
                                    func_evals = int(float.fromhex(parts[3]))
                                except ValueError:
                                    func_evals = int(parts[3])
                            else:
                                # For other strategies, we sum them from the log
                                total_fcn_evals = 0
                                for log_line in lines:
                                    if 'fcn evals=' in log_line:
                                        try:
                                            evals = int(log_line.split('fcn evals=')[1].split(',')[0].strip())
                                            total_fcn_evals += evals
                                        except (ValueError, IndexError):
                                            pass # Ignore lines that don't parse correctly
                                    elif 'No. function evaluations' in log_line:
                                        try:
                                            evals = int(log_line.split('=')[1].strip())
                                            total_fcn_evals += evals
                                        except (ValueError, IndexError):
                                            pass # Ignore lines that don't parse correctly
                                func_evals = total_fcn_evals
                    break
    except (IOError, IndexError, ValueError) as e:
        logging.warning(f"Could not parse log file {log_filename}: {e}")

    os.remove(log_filename)

    solution_np = rosen_problem.getXc().to_numpy()
    path = rosen_problem.path
    
    return solution_np, iterations, func_evals, path

def compare_strategies():
    """Compare the search strategies."""
    if not pyopttpp_available:
        return

    start_points = [
        np.array([-1.2, 1.0]),
        np.array([2.0, 2.0]),
        np.array([-2.0, -2.0]),
        np.array([0.0, 0.0]),
    ]
    
    strategies = {
        "LineSearch": pyopttpp.SearchStrategy.LineSearch,
        "TrustRegion": pyopttpp.SearchStrategy.TrustRegion,
        "TrustPDS": pyopttpp.SearchStrategy.TrustPDS,
    }
    
    results = {}

    for name, strategy in strategies.items():
        results[name] = []
        for start_point in start_points:
            logging.info(f"Running {name} from {start_point}")
            try:
                solution, iters, f_evals, path = run_optimization(strategy, start_point)
                results[name].append({
                    'start': start_point,
                    'solution': solution,
                    'iterations': iters,
                    'func_evals': f_evals,
                    'path': path,
                })
                logging.info(f"  -> Solution: {solution}, Iterations: {iters}, Func Evals: {f_evals}")
            except Exception as e:
                logging.error(f"  -> FAILED with error: {e}")

    # Plotting
    for i, start_point in enumerate(start_points):
        plt.figure(figsize=(10, 8))
        plt.title(f'Optimization Paths from {start_point}')
        
        # Rosenbrock function contour
        x = np.linspace(-3, 3, 400)
        y = np.linspace(-3, 3, 400)
        X, Y = np.meshgrid(x, y)
        Z = (1 - X)**2 + 100 * (Y - X**2)**2
        plt.contour(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='gray')
        
        for name in strategies.keys():
            path = np.array(results[name][i]['path'])
            plt.plot(path[:, 0], path[:, 1], 'o-', label=name, alpha=0.7)
        
        plt.plot(1, 1, 'r*', markersize=15, label='Optimum') # Optimum
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'comparison_from_start_{i}.png')
        plt.show()
        plt.close()

if __name__ == "__main__":
    compare_strategies()

