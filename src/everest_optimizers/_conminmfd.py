import numpy as np
from scipy.optimize import OptimizeResult

from everest_optimizers.pyoptsparse import (  # type: ignore[import-untyped]
    CONMIN,
    Optimization,
)


def minimize_conmin_mfd(fun, x0, args=(), bounds=None, constraints=None, options=None):
    n = len(x0)
    options = options or {}
    constraints = constraints or []
    bounds = bounds or [(-np.inf, np.inf)] * n

    def objfunc(xdict):
        x = xdict["x"]
        funcs = {"obj": fun(x, *args)}
        for i, constr in enumerate(constraints):
            funcs[f"c{i}"] = constr["fun"](x)
        return funcs, False

    # Should probably use jac instead of sens='FD' below:
    optProb = Optimization("PyOptSparse CONMIN", objfunc, sens="FD")
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]

    optProb.addVarGroup("x", n, "c", lower=lower_bounds, upper=upper_bounds, value=x0)
    optProb.addObj("obj")

    for i, constr in enumerate(constraints):
        cname = f"c{i}"
        match constr["type"]:
            case "ineq":
                optProb.addCon(cname, upper=0.0)
            case "eq":
                optProb.addCon(cname, equals=0.0)
            case other_type:
                raise ValueError(f"Unknown constraint type: {other_type}")

    optimizer = CONMIN(options=options)
    solution = optimizer(optProb)

    if solution is None:
        return OptimizeResult(
            x=x0,
            fun=fun(x0, *args),
            success=False,
            message="CONMIN terminated immediately",
            nfev=1,
        )

    x_arrays = list(solution.xStar.values())
    x_final = np.concatenate(x_arrays)

    return OptimizeResult(
        x=x_final,
        fun=solution.fStar,
        success=solution.optInform is None,
        message="" if solution.optInform is None else solution.optInform.message,
        nfev=solution.userObjCalls,
    )
