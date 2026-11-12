# everest-optimizers

The everest-optimizers repository wraps optimization algorithms from the OPT++
library.

## Getting Started

## Installation

Install using pip (or compatible tools):

```bash
pip install .[test]
```

## Tests

To run tests, execute:

```bash
pytest
```

#### Run pyoptpp tests with address sanitizer on
Configure pyoptpp with address sanitizer
```bash
cmake -Bbuild -S. -DDEBUG_MEMORY=1
```

Build
```bash
cmake --build build
```

Copy the module to the virtual environment. so file might have a slightly
different name depending on OS/python version
```bash
cp build/src/optpp_python/_pyoptpp.cpython-313-x86_64-linux-gnu.so .venv/lib/python3.13/site-packages/everest_optimizers/_pyoptpp.cpython-313-x86_64-linux-gnu.so
```

Run the OPTPP tests. The preload should point to your locally installed
libasan.so.

```bash
LD_PRELOAD=/usr/lib64/libasan.so.8.0.0 pytest "tests" -svvv --cache-clear
```

## Linting and Formatting

Python:

- We use ruff for python linting:

```bash
ruff format .
ruff check --select ALL
```

C++:

- We use clang-format for c++ formatting:

```bash
clang-format -i your_file.cpp
```
