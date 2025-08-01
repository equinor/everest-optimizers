import numpy as np
from scipy.optimize import OptimizeResult
from everest_optimizers.pyoptsparse.pyOpt_optimizer import Optimization
from everest_optimizers.pyCONMIN.pyCONMIN import CONMIN


def _minimize_conmin_mfd(fun, x0, args=(), jac=None, bounds=None, constraints=None, options=None, **kwargs):
    options = options or {}
    constraints = constraints or []
    bounds = bounds or [(-np.inf, np.inf)] * len(x0)

    n = len(x0)

    def objfunc(xdict):
        x = xdict['x']
        funcs = {'obj': fun(x, *args)}
        for i, constr in enumerate(constraints):
            funcs[f'c{i}'] = constr['fun'](x)
        return funcs, False

    optProb = Optimization("PyOptSparse CONMIN", objfunc, sens='FD')
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]

    optProb.addVarGroup('x', n, 'c', lower=lower_bounds, upper=upper_bounds, value=x0)
    optProb.addObj('obj')

    for i, constr in enumerate(constraints):
        cname = f'c{i}'
        if constr['type'] == 'ineq':
            optProb.addCon(cname, upper=0.0)
        elif constr['type'] == 'eq':
            optProb.addCon(cname, equals=0.0)
        else:
            raise ValueError(f"Unknown constraint type: {constr['type']}")

    optimizer = CONMIN(options=options)
    sol = optimizer(optProb)

    return OptimizeResult(
        x=np.array([v for v in sol.xStar.values()]),
        fun=sol.fStar,
        success=sol.optInform.get("value", "") == "",
        message=sol.optInform.get("text", ""),
        nfev=sol.userObjCalls,
    )
