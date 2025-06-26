#!/usr/bin/env python3
"""
OPTPP Python bindings test

Demonstrates working pybind11 bindings with:
- Python function optimization 
- Quasi-Newton algorithm
- Dictionary result format
"""

import everest_optimizers

def test_optimization():
    print("Everest Optimizers Python Bindings Test")
    print()
    
    # Test 1: Quadratic function
    def quadratic(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    
    print("Test 1: Quadratic function")
    print("f(x,y) = (x-1)² + (y-2)²")
    
    result = everest_optimizers.optimize_python_func(quadratic, [0.0, 0.0])
    print(f"Solution: {result['x']}")
    print(f"Function value: {result['fun']:.2e}")
    print()
    
    # Test 2: Rosenbrock function  
    def rosenbrock(x):
        return 100.0 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    
    print("Test 2: Rosenbrock function")
    print("f(x,y) = 100*(y-x²)² + (1-x)²")
    
    result = everest_optimizers.optimize_python_func(rosenbrock, [-1.2, 1.0])
    print(f"Solution: {result['x']}")
    print(f"Function value: {result['fun']:.2e}")
    print()
    
    # Test 3: Higher dimensional
    def sphere(x):
        return sum(xi**2 for xi in x)
    
    print("Test 3: 5D Sphere function")
    print("f(x) = Σ xᵢ²")
    
    result = everest_optimizers.optimize_python_func(sphere, [2.0, -1.0, 3.0, -2.0, 1.0])
    print(f"Solution: {[f'{x:.4f}' for x in result['x']]}")
    print(f"Function value: {result['fun']:.2e}")

if __name__ == "__main__":
    test_optimization()