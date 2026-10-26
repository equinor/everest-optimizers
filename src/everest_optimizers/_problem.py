import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from everest_optimizers import pyoptpp


class NLF1Problem:
    def __init__(
        self,
        fun: Callable[..., float],
        x0: NDArray[np.float64],
        args: tuple[Any, ...],
        jac: Callable[..., NDArray[np.float64]],
        callback: Callable[..., None] | None = None,
    ) -> None:
        self.fun = fun
        self.x0 = np.asarray(x0, dtype=float)
        self.args = args
        self.jac = jac
        self.callback = callback

        self.nfev = 0
        self.njev = 0
        self.current_x: NDArray[np.float64] | None = None
        self.current_f: float | None = None
        self.current_g: NDArray[np.float64] | None = None

        self.nlf1_problem = self._create_nlf1_problem()

    def _create_nlf1_problem(self) -> pyoptpp.NLF1:
        # Create callback functions for objective evaluation
        def eval_f(x: NDArray[np.float64]) -> float:
            x_np = np.asarray(x, copy=True)
            self.current_x = x_np

            f_val = self.fun(x_np, *self.args)
            self.current_f = float(f_val)
            self.nfev += 1

            if self.callback is not None:
                try:
                    self.callback(x_np)
                except Exception as cb_err:  # ruff: ignore[blind-except]
                    warnings.warn(
                        f"Callback function raised exception: {cb_err}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            return self.current_f

        def eval_g(x: NDArray[np.float64]) -> NDArray[np.float64]:
            x_np = np.asarray(x, copy=True)
            grad = self.jac(x_np, *self.args)
            grad_np = np.asarray(grad, dtype=float)
            self.current_g = grad_np
            self.njev += 1
            return grad_np

        x0_vector = pyoptpp.SerialDenseVector(self.x0)
        return pyoptpp.NLF1(len(self.x0), eval_f, eval_g, x0_vector)
