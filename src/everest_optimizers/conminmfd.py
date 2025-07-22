import numpy as np
from scipy.optimize import OptimizeResult
from . import conmin_module

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
    x = np.asarray(x0, dtype=float)
    ndv = len(x)
    ncon = 0  # Unconstrained for now
    nacmx1 = ndv + ncon + 4  # Can be set more generously if needed

    # CONMIN-required dimensions
    n1 = ndv + 2
    n2 = ncon + 2 * ndv
    n3 = nacmx1
    n4 = max(n3, ndv)
    n5 = 2 * n4

    max_iter = options.get('maxiter', 100) if options else 100
    tol = tol or 1e-6

    # Allocate required arrays
    x_arr = np.zeros(n1)
    x_arr[:ndv] = x
    # Initialize last two elements, often to zero (depends on CONMIN)
    x_arr[ndv:] = 0.0

    # Bounds: default infinite, fill from user bounds if provided
    vlb = np.full(n1, -1e20)
    vub = np.full(n1, 1e20)
    if bounds is not None:
        lb, ub = bounds
        if lb is not None:
            vlb[:ndv] = np.asarray(lb, dtype=float)
        if ub is not None:
            vub[:ndv] = np.asarray(ub, dtype=float)

    grad = np.zeros(n2)
    scal = np.ones(n1)
    df = np.zeros(n1)
    a = np.zeros((n1, n3))
    s = np.zeros(n1)
    g1 = np.zeros(n2)
    g2 = np.zeros(n2)
    b = np.zeros((n3, n3))
    c = np.zeros(n4)
    isc = np.zeros(n2, dtype=np.int32)
    ic = np.zeros(n3, dtype=np.int32)
    ms1 = np.zeros(n5, dtype=np.int32)
    obj = np.array([0.0])
    info = np.zeros(1, dtype=np.int32)
    infog = np.zeros(1, dtype=np.int32)
    iter_ = np.zeros(1, dtype=np.int32)

    def wrapped_fun(x_in):
        # Only pass the first ndv variables to fun
        return fun(x_in[:ndv], *args)

    if jac is None:
        def compute_grad(xk):
            eps = 1e-8
            grad_ = np.zeros_like(xk)
            fx = wrapped_fun(xk)
            for i in range(len(xk)):
                x_step = np.copy(xk)
                x_step[i] += eps
                grad_[i] = (wrapped_fun(x_step) - fx) / eps
            return grad_
    else:
        compute_grad = lambda xk: jac(xk[:ndv], *args)

    nfev = 0
    njev = 0
    for _ in range(max_iter):
        obj[0] = wrapped_fun(x_arr)
        nfev += 1

        grad[:ndv] = compute_grad(x_arr[:ndv])
        njev += 1

        conmin_module.conmin(
            x_arr, vlb, vub, grad, scal, df, a, s, g1, g2, b, c, isc, ic, ms1,
            0.0, 0.0,                 #delfun, dabfun
            0.0, 0.0,                 #fdch, fdchm
            0.0, 0.0, 0.0, 0.0,       #ct, ctmin, ctl, ctlmin
            0.0, 0.0,                 #alphax, abobj1
            0.0,                      #theta
            obj,         
            0.0, 0.0, 0.0,            #ndv, ncon, nside
            0.0, 0.0, 0.0,            #iprint, nfdg, nscal
            0.0, 0.0, 0.0, 0.0, 0.0,  #linobj, itmax, itrm, icndir, igoto
            0.0, #nac
            info, infog, iter_,
            n1, n2, n3, n4, n5,
        )

        if callback:
            callback(x_arr[:ndv])

        if np.linalg.norm(grad[:ndv]) < tol:
            break

    return OptimizeResult(
        x=x_arr[:ndv],
        fun=obj[0],
        success=(info[0] == 0),
        message="Converged" if info[0] == 0 else f"Not converged (info={info[0]})",
        nfev=nfev,
        njev=njev,
        jac=grad[:ndv]
    )
