import torch
from pyrol import Problem, Solver, getCout
from pyrol.pyrol.Teuchos import ParameterList
from TorchObjectives import SquaredNorm
from TorchVectors import TensorVector


def build_parameter_list():
    params = ParameterList()
    params["General"] = ParameterList()
    params["General"]["Output Level"] = 1
    params["Step"] = ParameterList()
    params["Step"]["Trust Region"] = ParameterList()
    params["Step"]["Trust Region"]["Subproblem Solver"] = "Truncated CG"
    params["Step"]["Trust Region"]["Initial Radius"] = 1e2
    return params


def main():
    torch.set_default_dtype(torch.float64)

    x = torch.tensor([[3.0]], requires_grad=False)
    x = TensorVector(x)

    objective = SquaredNorm()
    g = x.clone()

    stream = getCout()

    problem = Problem(objective, x, g)
    problem.checkDerivatives(True, stream)

    params = build_parameter_list()
    solver = Solver(problem, params)
    solver.solve(stream)

    print(f"x = {x.torch_object}")


if __name__ == "__main__":
    main()
