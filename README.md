# everest-optimizers

The everest-optimizers repository aims to replace the carolina repository by implementing the two algorithms OPTPP_Q_NEWTON and CONMIN_MFD from Dakota. This removes the need of building Dakota through Carolina every time you need to use these two algorithms. Dakota is huge and quite cumbersome to build, so by replacing this dependency we can gain a lot of time.

## Getting Started

### CONMIN

- We have used some python bindings and implementation from pyoptsparse: (https://github.com/mdolab/pyoptsparse/)

- See the following documentation for which input options the implementation have:
(https://mdolab-pyoptsparse.readthedocs-hosted.com/en/latest/optimizers/CONMIN.html)

To call the CONMIN implementation, we go through the python interface that we have made. Both CONMIN and OPTPP should be called using the minimize() function.

Example:

```python
result = minimize(
    fun=obj,
    x0=x0,
    method="conmin_mfd",
    bounds=bounds,
    constraints=constraints,
    options={"ITMAX": 100},
)
```
- See also example usage in the tests. For example: [tests/CONMIN/test_conmin_mfd.py](tests/CONMIN/test_conmin_mfd.py)


## Installation

1. (Optional) We recommend using a virtual enviroment. This can be created and activated by one of the following approaches:

- Using uv:
```bash
uv venv
source .venv/bin/activate
```

- Without uv:
```bash
python3 -m venv venv
source venv/bin/activate
```

2.  Install the project:

```bash
pip install .[test]
```

### Running the tests

To run tests, execute one of the following commands:

- Recommended:

```bash
pytest tests
```

- All tests (this is not recommended as there are existing tests in Trilinos which are not working):
```bash
pytest
```

- One specific test file:

```bash
pytest path/to/your_file.py
```

- One specific test in a file:

```bash
pytest path/to/your_file.py::name_of_test
```

If you want to add print statements / see the print statements in the terminal, you should run with the -s:

```bash
pytest tests -s
```

### Linting and Formatting

Python:

- We use ruff for python linting. To install ruff, use:

```bash
pip install ruff
```

- To run ruff, do the following commands:

```bash
ruff format .
ruff check --select ALL
```

- To only run ruff on a select folder or file, do these commands example):

```bash
ruff format src/
ruff check src/

ruff format src/everest_optimizers_utils/dummy_implementation.py
ruff check src/everest_optimizers_utils/dummy_implementation.py
```

C++:

- We use clang-format for c++ formatting. Install clang-format like this:

```bash
sudo apt install clang-format
```

- To format a c++ or c file (or header file), use

```bash
clang-format -i your_file.cpp
```

- You can run the following command to format all files in your project!

```bash
find . -regex '.*\.\(cpp\|hpp\|cc\|c\|h\)' -exec clang-format -i {} +
```
