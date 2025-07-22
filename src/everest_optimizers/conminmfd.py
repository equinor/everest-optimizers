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

    max_iter = options.get('maxiter', 40) if options else 40
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

    # Compute initial gradient (optional, but CONMIN may expect it)
    grad[:ndv] = compute_grad(x_arr[:ndv])

    # Compute initial objective
    obj[0] = wrapped_fun(x_arr)

    print("Before conmin call:")
    print("iter_ =", iter_[0])
    # Single call to CONMIN — let it handle all iterations internally
    conmin_module.conmin(
        x_arr, vlb, vub, grad, scal, df, a, s, g1, g2, b, c, isc, ic, ms1,
        0.001, 0.001,               #delfun, dabfun
        0.01, 0.01,                 #fdch, fdchm
        -0.1, 0.004, -0.01, 0.001,  #ct, ctmin, ctl, ctlmin
        0.1, 0.1,                   #alphax, abobj1
        1.0,                        #theta
        obj,
        ndv, ncon, 0,               #ndv, ncon, nside
        2, 5, 1,                    #iprint, nfdg, nscal
        0, max_iter, 3, 3, 1,       #linobj, itmax, itrm, icndir, igoto   # docs says IGOTO should be 0 ?
        0,                          #nac
        info, infog, iter_,
        n1, n2, n3, n4, n5,
    )
    
    print("After conmin call:")
    print("iter_ =", iter_[0])
    print("info =", info[0])
    print("infog =", infog[0])
    print("x =", x_arr[:ndv])
    print("obj =", obj[0])

    if callback:
        callback(x_arr[:ndv])

    return OptimizeResult(
        x=x_arr[:ndv],
        fun=obj[0],
        success=(info[0] == 0),
        message="Converged" if info[0] == 0 else f"Not converged (info={info[0]})",
        nfev=None,
        njev=None,
        jac=None
    )
