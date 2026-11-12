from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from everest_optimizers import EvaluationMode, minimize

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from everest_optimizers import EvaluationResult


def _evaluate_quadratic(
    x: NDArray[np.float64], mode: EvaluationMode
) -> EvaluationResult:
    delta = x - [2.0, -1.0]
    values = np.array([delta @ delta]) if mode & EvaluationMode.VALUES else None
    gradients = 2 * delta[np.newaxis, :] if mode & EvaluationMode.GRADIENTS else None
    return values, gradients


def test_evaluator_speculative_requests_values_and_gradients_together() -> None:
    modes: list[EvaluationMode] = []

    def evaluate(x: NDArray[np.float64], mode: EvaluationMode) -> EvaluationResult:
        modes.append(mode)
        return _evaluate_quadratic(x, mode)

    x, status = minimize(np.zeros(2), evaluate=evaluate, options={"speculative": True})

    assert status > 0
    np.testing.assert_allclose(x, [2.0, -1.0], atol=1e-4)
    assert EvaluationMode.BOTH in modes


@pytest.mark.parametrize(
    ("result", "error", "match"),
    [
        (np.zeros(1), TypeError, "must return a pair"),
        ((None, np.zeros((1, 2))), ValueError, "must return values when requested"),
        ((np.zeros(1), None), ValueError, "must return gradients when requested"),
        ((np.zeros(2), np.zeros((1, 2))), ValueError, r"values must have shape \(1,\)"),
        (
            (np.zeros(1), np.zeros((2, 2))),
            ValueError,
            r"gradients must have shape \(1, 2\)",
        ),
    ],
)
def test_evaluator_invalid_results_are_rejected(
    result: object, error: type[Exception], match: str
) -> None:
    def evaluate(_x: NDArray[np.float64], _mode: EvaluationMode) -> EvaluationResult:
        return cast("EvaluationResult", result)

    with pytest.raises(error, match=match):
        minimize(np.zeros(2), evaluate=evaluate, options={"speculative": True})


def test_evaluator_exceptions_are_propagated() -> None:
    def evaluate(_x: NDArray[np.float64], _mode: EvaluationMode) -> EvaluationResult:
        msg = "evaluation failed"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="evaluation failed"):
        minimize(np.zeros(2), evaluate=evaluate)
