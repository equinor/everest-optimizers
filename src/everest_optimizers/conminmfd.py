import numpy as np
from . import myprogram
from scipy.optimize import OptimizeResult


def _minimize_conmin_mfd(
    fun,
    x0,
    args=(),
    jac=None,
    bounds=None,
    constraints=None,
    tol=None,
    callback=None,
    options=None
) -> OptimizeResult:
    x_arr = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    
    ndv = 4
    n1 = 6
    vlb = np.full(n1, -1e20)
    vub = np.full(n1, 1e20)
    
    ncon = 0
    if constraints is not None:
        ineq_constraints = [c for c in constraints if c.get('type') == 'ineq']
        ncon = len(ineq_constraints)
    else:
        ineq_constraints = []
        ncon = 0
    
    n2 = 11
    g = np.zeros(n2)
    for i, con in enumerate(ineq_constraints):
        g[i] = con['fun'](x_arr[:ndv])
    
    myprogram.run_example(x_arr, vlb, vub, g)