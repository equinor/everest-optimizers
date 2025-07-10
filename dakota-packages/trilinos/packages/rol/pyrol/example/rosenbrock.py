import numpy as np
from pyrol import Objective, Problem, Solver, getCout
from pyrol.pyrol.Teuchos import ParameterList
from pyrol.vectors import NumPyVector


class RosenbrockObjective(Objective):
    def __init__(self):
        self.alpha = 100
        super().__init__()

    def value(self, x, tol):
        v = self.alpha * (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2
        return v

    def gradient(self, g, x, tol):
        g[0] = 2 * self.alpha * (x[0] ** 2 - x[1]) * 2 * x[0]
        g[0] += 2 * (x.array[0] - 1)
        g[1] = -2 * self.alpha * (x[0] ** 2 - x[1])

    def hessVec(self, hv, v, x, tol):
        h11 = 12 * self.alpha * x[0] ** 2 - 4 * self.alpha * x[1] + 2
        h12 = -4 * self.alpha * x[0]
        h22 = 2 * self.alpha
        hv[0] = h11 * v[0] + h12 * v[1]
        hv[1] = h12 * v[0] + h22 * v[1]


def build_parameter_list():
    params = ParameterList()
    params["General"] = ParameterList()
    params["General"]["Output Level"] = 1
    params["Step"] = ParameterList()
    params["Step"]["Trust Region"] = ParameterList()
    params["Step"]["Trust Region"]["Subproblem Solver"] = "Truncated CG"
    return params


def main():
    # Configure parameter list.  ################
    params = build_parameter_list()

    # Set the output stream.  ###################
    stream = getCout()

    # Set up vectors.  ##########################
    x = NumPyVector(np.array([-3.0, -4.0]))
    g = x.dual()

    # Set up the problem.  ######################
    objective = RosenbrockObjective()
    problem = Problem(objective, x, g)
    problem.check(True, stream)

    # Solve.  ###################################
    solver = Solver(problem, params)
    solver.solve(stream)

    # Check the solution.  ######################
    target = np.array([1.0, 1.0])
    np.testing.assert_allclose(x.array, target)
    print("Test passed: Correctly optimized!")


if __name__ == "__main__":
    main()
