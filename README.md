# everest-optimizers

## Getting Started

### Prerequisites

### Installation

1.  (optional) Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

2.  Install the project:

```bash
pip install .
```

### Running the project

To run tests, execute one of the following commands:

- All tests:
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

- To only run ruff on a select folder or file, do these commands (for example):

```bash
ruff check src/
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
