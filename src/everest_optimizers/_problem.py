import weakref
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from . import _pyoptpp
from ._evaluation import EvaluationMode, Evaluator

_EvaluateCallbackResult = tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]
_EvaluateCallback = Callable[[NDArray[np.float64], int], _EvaluateCallbackResult]


class _Evaluation(NamedTuple):
    mode: EvaluationMode
    values: NDArray[np.float64] | None
    gradients: NDArray[np.float64] | None


class NLF1Problem:
    def __init__(
        self,
        evaluate: Evaluator,
        x0: NDArray[np.float64],
        nonlinear_equalities: NDArray[np.bool_],
    ) -> None:
        self.evaluate = evaluate
        self.x0 = x0
        self.num_equalities = int(np.count_nonzero(nonlinear_equalities))
        self.constraint_order = np.concatenate(
            (
                np.flatnonzero(nonlinear_equalities),
                np.flatnonzero(~nonlinear_equalities),
            )
        )
        self._evaluations: dict[tuple[float, ...], list[_Evaluation]] = {}
        self._evaluate_callback = self._create_evaluate_callback()
        self.nlf1_problem = _pyoptpp.ObjectiveNLF1(
            x0.size,
            self.constraint_order.size,
            self._evaluate_callback,
            _pyoptpp.SerialDenseVector(x0),
        )

    def create_nonlinear_constraint(self) -> _pyoptpp.NonLinearConstraint | None:
        count = self.constraint_order.size
        if count == 0:
            return None
        upper = np.full(count, np.inf)
        upper[: self.num_equalities] = 0.0
        constraint_nlf1 = _pyoptpp.ConstraintNLF1(
            self.x0.size, count, self._evaluate_callback
        )
        return _pyoptpp.NonLinearConstraint(
            _pyoptpp.NLP(constraint_nlf1),
            _pyoptpp.SerialDenseVector(np.zeros(count)),
            _pyoptpp.SerialDenseVector(upper),
            self.num_equalities,
            count - self.num_equalities,
        )

    def _create_evaluate_callback(self) -> _EvaluateCallback:
        evaluate = weakref.WeakMethod(self._evaluate)

        def callback(x: NDArray[np.float64], mode: int) -> _EvaluateCallbackResult:
            method = evaluate()
            if method is None:
                msg = "The optimizer problem no longer exists"
                raise RuntimeError(msg)
            return method(x, mode)

        return callback

    def _evaluate(self, x: NDArray[np.float64], mode: int) -> _EvaluateCallbackResult:
        evaluation_mode = EvaluationMode(mode)
        key = tuple(x.tolist())
        evaluation = self._stored_evaluation(key, evaluation_mode)
        if evaluation is None:
            evaluation = self._evaluate_requested(x, evaluation_mode)
            self._evaluations.setdefault(key, []).append(evaluation)

        values = evaluation.values if evaluation_mode & EvaluationMode.VALUES else None
        gradients = (
            evaluation.gradients if evaluation_mode & EvaluationMode.GRADIENTS else None
        )
        if values is not None:
            values = np.concatenate((values[:1], values[1:][self.constraint_order]))
        if gradients is not None:
            gradients = np.concatenate(
                (gradients[:1], gradients[1:][self.constraint_order]), axis=0
            )
        return values, gradients

    def _stored_evaluation(
        self, key: tuple[float, ...], mode: EvaluationMode
    ) -> _Evaluation | None:
        for evaluation in self._evaluations.get(key, ()):
            if evaluation.mode & mode == mode:
                return evaluation
        return None

    def _evaluate_requested(
        self, point: NDArray[np.float64], mode: EvaluationMode
    ) -> _Evaluation:
        result = self.evaluate(point, mode)
        if not isinstance(result, tuple) or len(result) != 2:  # ruff: ignore[magic-value-comparison]
            msg = "evaluate must return a pair of values and gradients"
            raise TypeError(msg)
        values, gradients = result
        return _Evaluation(
            mode,
            self._validated_values(values) if mode & EvaluationMode.VALUES else None,
            (
                self._validated_gradients(gradients)
                if mode & EvaluationMode.GRADIENTS
                else None
            ),
        )

    def _validated_values(
        self, values: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        if values is None:
            msg = "evaluate must return values when requested"
            raise ValueError(msg)
        result = np.asarray(values, dtype=np.float64)
        expected_shape = (1 + self.constraint_order.size,)
        if result.shape != expected_shape:
            msg = (
                f"evaluate values must have shape {expected_shape}, got {result.shape}"
            )
            raise ValueError(msg)
        return result.copy()

    def _validated_gradients(
        self, gradients: NDArray[np.float64] | None
    ) -> NDArray[np.float64]:
        if gradients is None:
            msg = "evaluate must return gradients when requested"
            raise ValueError(msg)
        result = np.asarray(gradients, dtype=np.float64)
        expected_shape = (1 + self.constraint_order.size, self.x0.size)
        if result.shape != expected_shape:
            msg = f"evaluate gradients must have shape {expected_shape}, got {result.shape}"
            raise ValueError(msg)
        return result.copy()
