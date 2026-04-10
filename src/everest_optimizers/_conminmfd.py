from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from everest_optimizers.pyoptsparse import (  # type: ignore[import-untyped]
    CONMIN,
    Optimization,
)


def minimize_conmin_mfd(  # noqa: PLR0913, PLR0917
    fun: Callable[..., float],
    x0: NDArray[np.float64],
    args: tuple[Any, ...] = (),
    bounds: list[tuple[float, float]] | None = None,
    constraints: list[dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
) -> OptimizeResult:
    n = len(x0)
    options = options or {}
    constraints = constraints or []
    bounds = bounds or [(-np.inf, np.inf)] * n

    def objfunc(xdict: dict[str, Any]) -> tuple[dict[str, float], bool]:
        x = xdict["x"]
        funcs = {"obj": fun(x, *args)}
        for i, constr in enumerate(constraints):
            funcs[f"c{i}"] = constr["fun"](x)
        return funcs, False

    # Should probably use jac instead of sens='FD' below:
    opt_prob = Optimization("PyOptSparse CONMIN", objfunc, sens="FD")
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]

    opt_prob.addVarGroup("x", n, "c", lower=lower_bounds, upper=upper_bounds, value=x0)
    opt_prob.addObj("obj")

    for i, constr in enumerate(constraints):
        cname = f"c{i}"
        match constr["type"]:
            case "ineq":
                opt_prob.addCon(cname, upper=0.0)
            case "eq":
                opt_prob.addCon(cname, equals=0.0)
            case other_type:
                msg = f"Unknown constraint type: {other_type}"
                raise ValueError(msg)

    optimizer = CONMIN(options=options)
    solution = optimizer(opt_prob)

    if solution is None:
        return OptimizeResult(  # type: ignore[call-arg]
            x=x0,
            fun=fun(x0, *args),
            success=False,
            message="CONMIN terminated immediately",
            nfev=1,
        )

    x_arrays = list(solution.xStar.values())
    x_final = np.concatenate(x_arrays)

    return OptimizeResult(  # type: ignore[call-arg]
        x=x_final,
        fun=solution.fStar,
        success=solution.optInform is None,
        message="" if solution.optInform is None else solution.optInform.message,
        nfev=solution.userObjCalls,
    )
