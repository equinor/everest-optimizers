# everest-optimizers

The everest-optimizers repository wraps optimization algorithms from the OPT++
library.

## Getting Started

## Installation

Install using pip (or compatible tools):

```bash
pip install .[test]
```

## Usage

`minimize` runs the `optpp_q_newton` method. It requires a combined evaluator
function for the objective and any nonlinear constraints.

```python
import numpy as np

from everest_optimizers import EvaluationMode, minimize


def evaluate(x, mode):
    delta = x - [2.0, -1.0]
    values = np.array([delta @ delta]) if mode & EvaluationMode.VALUES else None
    gradients = 2 * delta[np.newaxis, :] if mode & EvaluationMode.GRADIENTS else None
    return values, gradients


x, status = minimize(np.zeros(2), evaluate=evaluate)
print(x)
```

The evaluator receives `(x, mode)` and returns `(values, gradients)`.
`EvaluationMode.VALUES` requests values, `EvaluationMode.GRADIENTS` requests
gradients, and `EvaluationMode.BOTH` requests both. Return `None` for a part
that was not requested.

For `n` variables and `m` nonlinear constraints, `values` has shape `(1 + m,)`
and `gradients` has shape `(1 + m, n)`. The first entry or row belongs to the
objective; the remaining entries or rows belong to the nonlinear constraints.

`minimize` stores every evaluation until it returns. The evaluator is not called
for a request when one stored evaluation of exactly the same point contains
everything that is requested. Otherwise the evaluator is called for the complete
request and its result is stored. Stored evaluations of the same point are not
combined: if a point has one stored evaluation of values and another of
gradients, a request for both calls the evaluator. The evaluator must return the
same result each time it is called for the same point.

Bounds and constraints are given as NumPy arrays:

- `bounds` is a pair `(lower, upper)` of variable bounds, broadcast to the shape
  of the initial point. Infinite values mean that there is no bound.
- `linear_constraints` is a triple `(coefficients, lower, upper)`, with
  `coefficients` of shape `(k, n)` and bounds of shape `(k,)`. A row with equal
  finite bounds is an equality.
- `nonlinear_equalities` is a boolean array with one entry per nonlinear
  constraint. A constraint value `c` must satisfy `c = 0` where the entry is
  `True`, and `c >= 0` otherwise.

The initial point is passed to OPT++ unchanged. An initial point outside the
bounds or constraints may lead to an infeasible result. All arguments except the
initial point are keyword-only.

`minimize` returns a tuple `(x, status)` with the final point and the OPT++
termination code, which is positive if the optimizer converged.

### Options

`minimize` selects and configures an OPT++ algorithm: an interior-point method
for problems with linear or nonlinear constraints, a bound-constrained
quasi-Newton method for problems with finite variable bounds, a quasi-Newton
method for unconstrained problems with fewer than 100 variables, and a
limited-memory BFGS method otherwise. Bounds with a magnitude of `1e30` or more
count as absent.

It accepts the options `max_iterations`, `max_function_evaluations`,
`convergence_tolerance`, `gradient_tolerance`, `max_step`, `search_method`,
`merit_function`, `steplength_to_boundary`, `centering_parameter`,
`speculative`, and `output`, plus `output_file`. Supply these as a dictionary
keyed by option name, such as `{"max_iterations": 50, "speculative": True}`.
Unspecified options use their default values.

`steplength_to_boundary` must be a number between 0 and 1, inclusive.

`"output": "debug"` turns on the OPT++ debug output; the other `output` levels
have no effect. `output_file` names the file that receives the OPT++ output;
without it, the output is discarded.

`search_method` accepts `value_based_line_search`, `gradient_based_line_search`,
and `trust_region`. The `tr_pds` strategy is disabled because its OPT++
implementation retains process-wide state between optimization runs.

`max_function_evaluations` is compared with the number of objective-value
requests that OPT++ makes. OPT++ does not count a request that its most recent
objective evaluation already covers. It does count a request that `minimize`
answers from the stored evaluations described above, so this number can exceed
the number of evaluator calls. OPT++ checks it between iterations, so it is not
a hard cap.

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
