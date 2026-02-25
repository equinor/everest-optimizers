import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from everest_optimizers import pyoptpp


class NLF1Problem:
    def __init__(
        self,
        fun: Callable,
        x0: np.ndarray,
        args: tuple,
        jac: Callable[..., npt.NDArray[np.float64]] | None = None,
        callback: Callable | None = None,
    ):
        self.fun = fun
        self.x0 = np.asarray(x0, dtype=float)
        self.args = args
        self.jac = jac
        self.callback = callback

        self.nfev = 0
        self.njev = 0
        self.current_x = None
        self.current_f = None
        self.current_g = None

        self.nlf1_problem = self._create_nlf1_problem()

    def _create_nlf1_problem(self):
        """Create the NLF1 problem for OPTPP using C++ CallbackNLF1."""

        # Create callback functions for objective evaluation
        def eval_f(x):
            x_np = np.array(x.to_numpy(), copy=True)
            self.current_x = x_np

            f_val = self.fun(x_np, *self.args)
            self.current_f = float(f_val)
            self.nfev += 1

            if self.callback is not None:
                try:
                    self.callback(x_np)
                except Exception as cb_err:
                    warnings.warn(
                        f"Callback function raised exception: {cb_err}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            return self.current_f

        def eval_g(x):
            x_np = np.array(x.to_numpy(), copy=True)
            assert self.jac is not None
            grad = self.jac(x_np, *self.args)
            grad_np = np.asarray(grad, dtype=float)
            self.current_g = grad_np
            self.njev += 1
            return grad_np

        x0_vector = pyoptpp.SerialDenseVector(self.x0)
        if self.jac is not None:
            return pyoptpp.NLF1(len(self.x0), eval_f, eval_g, x0_vector)
        else:
            return pyoptpp.FDNLF1(len(self.x0), eval_f, x0_vector)
