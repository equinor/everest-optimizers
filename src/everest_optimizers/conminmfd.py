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

    if constraints is not None:
        ineq_constraints = [c for c in constraints if c.get('type') == 'ineq']
        ncon = len(ineq_constraints)
    else:
        ineq_constraints = []
        ncon = 1

    nacmx1 = ndv + ncon + 4
    nside = 0
    nac = 0

    # CONMIN-required dimensions
    n1 = ndv + 2
    n2 = ncon + 2 * ndv
    n3 = nacmx1
    n4 = max(n3, ndv)
    n5 = 2 * n4

    tol = tol or 1e-6

    x_arr = np.zeros(n1)
    x_arr[:ndv] = x

    # Bounds
    vlb = np.full(n1, -1e20)
    vub = np.full(n1, 1e20)
    if bounds is not None:
        lb, ub = bounds
        vlb[:ndv] = np.asarray(lb, dtype=float)
        vub[:ndv] = np.asarray(ub, dtype=float)

    g = np.zeros(n2)
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
    nfdg = 0
    iprint = 3

    itmax = 100
    fdch = 1.0e-5
    fdchm = 1.0e-5
    ct = -0.1
    ctmin = 0.001
    ctl = -0.01
    ctlmin = 0.001
    delfun = 1.0e-7
    dabfun = 1.0e-7
    icndir = 5

    nscal = 0
    linobj = 0
    itrm = 3
    theta = 1
    alphax = 0.1
    abobj1 = 0.1
    igoto = np.array([1], dtype=np.int32)

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

    df[:ndv] = compute_grad(x_arr[:ndv])
    obj[0] = wrapped_fun(x_arr)

    for i, con in enumerate(ineq_constraints):
        g[i] = con['fun'](x_arr[:ndv])

    print("Before conmin call:")
    print("iter_ =", iter_[0])

    for i in range(itmax):
        conmin_module.conmin(
            x_arr, vlb, vub, g, scal, df, a, s, g1, g2, b, c, isc, ic, ms1,
            delfun, dabfun,
            fdch, fdchm,
            ct, ctmin, ctl, ctlmin,
            alphax, abobj1,
            theta,
            obj,
            ndv, ncon, nside,
            iprint, nfdg, nscal,
            linobj, itmax, itrm, icndir, igoto,
            nac,
            info, infog, iter_,
        )
        
        if igoto[0] == 0:
            break

        if info[0] == 1:
            obj[0] = wrapped_fun(x_arr)
            for idx, con in enumerate(ineq_constraints):
                g[idx] = con['fun'](x_arr[:ndv])

        elif info[0] == 2:
            df[:ndv] = compute_grad(x_arr[:ndv])

        if callback:
            callback(x_arr[:ndv])

    success = (info[0] == 0)
    message = "Converged" if success else f"Not converged (info={info[0]})"

    return OptimizeResult(
        x=x_arr[:ndv],
        fun=obj[0],
        success=success,
        message=message,
        nfev=None,
        njev=None,
        jac=None,
    )
