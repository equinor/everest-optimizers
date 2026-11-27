"""The combined objective and nonlinear constraint evaluation interface."""

from collections.abc import Callable
from enum import IntFlag
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray


class EvaluationMode(IntFlag):
    """The parts of the objective and nonlinear constraints to evaluate."""

    VALUES = 1
    GRADIENTS = 2
    BOTH = VALUES | GRADIENTS


EvaluationResult: TypeAlias = tuple[
    NDArray[np.float64] | None, NDArray[np.float64] | None
]
Evaluator: TypeAlias = Callable[[NDArray[np.float64], EvaluationMode], EvaluationResult]
