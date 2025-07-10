import numpy as np
from PyROL import Objective_SimOpt, Vector_SimOpt
from PyROL.vectors import npVector


class MyObjective(Objective_SimOpt):
    def value(self, *args):
        if len(args) == 2:
            x, tol = args
            # return self.value(x.get_1(), x.get_2(), tol)
            return super().value(*args)
        return 0.0


u = npVector(np.ones(3))
z = npVector(np.ones(2))
x = Vector_SimOpt(u, z)
tol = 1e-12

obj = MyObjective()
assert obj.value(u, z, tol) == 0.0
assert obj.value(x, tol) == 0.0
