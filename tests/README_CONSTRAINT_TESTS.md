# C++ Constraint Tests for OPTPP

This directory contains comprehensive constraint tests implemented at the C++ level to avoid Python/C++ translation issues and potential segfaults when using pybind11.

## Overview

The constraint testing framework consists of:

1. **C++ Implementation** (`dakota-packages/OPTPP/tests/constraint_tests.h/.C`)
   - Complete test implementations in C++
   - Tests for linear equality/inequality constraints
   - Tests for bound constraints
   - Tests for mixed constraint types
   - Placeholder framework for nonlinear constraints

2. **pybind11 Bindings** (`dakota-packages/OPTPP/python/pyopttpp.cpp`)
   - Exposes C++ test functions to Python
   - Binds test result structures
   - Provides utility functions

3. **Python Test Wrapper** (`tests/test_cpp_constraint_tests.py`)
   - pytest-compatible test suite
   - Calls C++ test functions
   - Provides detailed reporting

## Test Categories

### Linear Equality Constraints (LinearEquation)
- **Test 1**: Simple 2D quadratic with `x + y = 3`
- **Test 2**: 3D quadratic with multiple equalities `x + y + z = 6`, `x - y = 1`

### Linear Inequality Constraints (LinearInequality)  
- **Test 3**: 2D quadratic with `x + y <= 2`
- **Test 4**: 3D quadratic with multiple inequalities

### Bound Constraints (BoundConstraint)
- **Test 5**: Simple bounds where optimum is at boundary
- **Test 6**: Asymmetric bounds where optimum is within bounds

### Mixed Linear Constraints (CompoundConstraint)
- **Test 7**: Combination of bounds + equality + inequality constraints

### Nonlinear Constraints (Placeholder)
- **Test 8**: Nonlinear equality `x² + y² = 4` (not implemented)
- **Test 9**: Nonlinear inequality `x² + y² <= 1` (not implemented)  
- **Test 10**: Mixed nonlinear constraints (not implemented)

## Running the Tests

### Via pytest
```bash
cd everest-optimizers
pytest tests/test_cpp_constraint_tests.py -v
```

### Direct Python execution
```bash
cd everest-optimizers
python tests/test_cpp_constraint_tests.py
```

### From C++ (after building)
```python
import pyopttpp

# Run individual tests
result = pyopttpp.run_linear_eq_test1()
print(f"Success: {result.success}")
print(f"Message: {result.message}")

# Run all tests
all_results = pyopttpp.run_all_constraint_tests()
print(f"Passed: {all_results.passed_tests()}/{all_results.total_tests()}")
```

## Building

The constraint tests are automatically included when building the pybind11 module:

```bash
cd dakota-packages/OPTPP/build
cmake ..
make pyopttpp
```

The `constraint_tests.C` file is compiled into the `pyopttpp` module via the updated `python/CMakeLists.txt`.

## Test Structure

Each test follows this pattern:

1. **Initialization Function**: Sets up starting point
2. **Objective Function**: Defines the optimization problem  
3. **Constraint Creation**: Sets up appropriate OPTPP constraint objects
4. **Test Runner**: Creates NLF1 problem, runs OptQNIPS optimizer, validates results

### Test Result Structure
```cpp
struct TestResult {
    bool success;                          // Overall test success
    double final_objective;                // Final objective function value
    SerialDenseVector<int,double> final_point;  // Final optimization point
    double constraint_violation;           // Maximum constraint violation
    int iterations;                        // Number of optimizer iterations  
    std::string message;                   // Status/error message
};
```

## Advantages of C++ Implementation

1. **Avoids Translation Issues**: No Python/C++ data conversion during optimization
2. **Prevents Segfaults**: All memory management handled within C++
3. **Performance**: Direct C++ execution without Python overhead
4. **Debugging**: Easier to debug constraint issues at the C++ level
5. **OPTPP Native**: Uses OPTPP constraint objects directly

## Extending the Tests

To add new constraint tests:

1. Add test functions to `constraint_tests.h/.C`
2. Add pybind11 bindings in `pyopttpp.cpp`
3. Add Python test wrapper in `test_cpp_constraint_tests.py`
4. Update `AllTestResults` structure if needed

## Current Limitations

- Nonlinear constraints require more complex NLP setup with specialized constraint function signatures
- Some advanced constraint features may not be tested yet
- Error handling could be more granular

## Future Work

- Implement nonlinear constraint tests with proper NLP object setup
- Add more sophisticated mixed constraint scenarios  
- Add constraint Jacobian/Hessian testing
- Add infeasible problem detection tests
- Add constraint tolerance sensitivity tests