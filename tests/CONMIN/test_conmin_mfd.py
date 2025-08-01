import numpy as np
import pytest
from everest_optimizers.pyOpt_optimizer import Optimization
from everest_optimizers.pyCONMIN.pyCONMIN import CONMIN

def rosen_suzuki_obj(x):
    return (
        x[0]**2 - 5*x[0] +
        x[1]**2 - 5*x[1] +
        2*x[2]**2 - 21*x[2] +
        x[3]**2 + 7*x[3] + 50
    )

def constraint1(x):
    return x[0]**2 + x[0] + x[1]**2 - x[1] + x[2]**2 + x[2] + x[3]**2 - x[3] - 8

def constraint2(x):
    return x[0]**2 - x[0] + 2*x[1]**2 + x[2]**2 + 2*x[3]**2 - x[3] - 10

def constraint3(x):
    return 2*x[0]**2 + 2*x[0] + x[1]**2 - x[1] + x[2]**2 - x[3] - 5

def test_conmin_rosen_suzuki():
    x0 = np.array([1.0, 1.0, 1.0, 1.0])

    def objfunc(xdict):
        x = xdict['x']
        funcs = {
            "obj": rosen_suzuki_obj(x),
            "g1": constraint1(x),
            "g2": constraint2(x),
            "g3": constraint3(x),
        }
        fail = False
        return funcs, fail

    optProb = Optimization("Rosen-Suzuki Problem", objfunc, sens='FD')
    optProb.addVarGroup('x', 4, 'c', lower=-10.0, upper=10.0, value=x0)
    optProb.addObj('obj')
    optProb.addCon('g1', upper=0.0)
    optProb.addCon('g2', upper=0.0)
    optProb.addCon('g3', upper=0.0)


    optimizer = CONMIN(options={"IPRINT": 2, "ITMAX": 100})
    sol = optimizer(optProb)

    expected_x = np.array([0.0, 1.0, 2.0, -1.0])
    expected_fun = 6.0

    assert sol.optInform["value"] == "", f"Optimizer did not return success: {sol.optInform}"
    xstar_arr = np.array(list(sol.xStar.values())).flatten()
    assert np.allclose(xstar_arr, expected_x, atol=1e-2), f"x not close to expected: {xstar_arr}"
    assert np.isclose(sol.fStar, expected_fun, atol=1e-2), f"Function value not close to expected: {sol.fStar}"