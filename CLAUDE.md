# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Testing
- Run all tests: `pytest`
- Run specific test file: `pytest path/to/your_file.py`
- Run specific test: `pytest path/to/your_file.py::name_of_test`
- Skip external tests: `pytest -m "not external"`

### Installation
- Install project: `pip install .`
- Optional virtual environment:
  ```bash
  uv venv
  source .venv/bin/activate
  ```

### Linting and Formatting
- **Python**: Use ruff for linting and formatting
  - Format code: `ruff format .`
  - Check all rules: `ruff check --select ALL`
  - Check specific directory: `ruff check src/`
  - Check specific file: `ruff check src/everest_optimizers/file.py`

- **C++**: Use clang-format for formatting
  - Format single file: `clang-format -i your_file.cpp`
  - Format all C++ files: `find . -regex '.*\.\(cpp\|hpp\|cc\|c\|h\)' -exec clang-format -i {} +`

## Architecture Overview

This is a Python optimization library that provides scipy.optimize.minimize-compatible interfaces to OPTPP optimizers. The library implements three main optimization algorithms from the OPTPP C++ library.

### Core Structure
- **Main Interface**: `src/everest_optimizers/minimize.py` - Drop-in replacement for scipy.optimize.minimize
- **Optimizer Implementations**: `src/everest_optimizers/optqnewton.py` - Core optimizer wrappers
- **Enhanced OPTQNIPS**: `src/everest_optimizers/optqnips_impl.py` - Enhanced implementation of OPTQNIPS

### Supported Optimization Methods
1. **optpp_q_newton**: Unconstrained quasi-Newton optimizer
2. **optpp_constr_q_newton**: Constrained quasi-Newton optimizer (supports bounds and linear constraints)
3. **optpp_q_nips**: Quasi-Newton Interior-Point Solver (supports bounds and linear constraints)

### Key Technical Details
- Uses pybind11 to interface with OPTPP C++ library
- OPTPP is located in `dakota-packages/OPTPP/build/python`
- All optimizers create NLF1 problems (first-order methods)
- Constraint handling uses CompoundConstraint objects from OPTPP
- Linear constraints support equality (lb == ub) and one-sided inequalities (Ax >= lb)
- Finite difference gradients available when analytical gradients not provided

### Test Structure
- `tests/OptQNewton/` - Tests for OptQNewton optimizer
- `tests/ropt_dakota_vs_everest_optimizers/` - Comparison tests with other optimizers
- `tests/dakota/` - Dakota integration tests
- External tests marked with `@pytest.mark.external` decorator

### Configuration
- pytest configuration in `pytest.ini` sets `pythonpath = src`
- ruff configuration in `pyproject.toml` ignores S101 (assert statements) in tests
- Build system uses setuptools with pybind11, cmake, and ninja