# THIS TEST IS WORKING ON LINUX BUT FAILING ON MACOS IN GITHUB WORKFLOW


# # Copyright 2013 National Renewable Energy Laboratory (NREL)
# #
# #    Licensed under the Apache License, Version 2.0 (the "License");
# #    you may not use this file except in compliance with the License.
# #    You may obtain a copy of the License at
# #
# #        http://www.apache.org/licenses/LICENSE-2.0
# #
# #    Unless required by applicable law or agreed to in writing, software
# #    distributed under the License is distributed on an "AS IS" BASIS,
# #    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #    See the License for the specific language governing permissions and
# #    limitations under the License.
# #
# # ++==++==++==++==++==++==++==++==++==++==

# from __future__ import print_function
# from numpy import array
# import numpy as np
# from traceback import print_exc
# import sys
# import pytest

# from dakota import DakotaBase, DakotaInput
# from everest_optimizers.minimize import minimize


# class TestDriver(DakotaBase):
#     __test__ = False
    
#     def __init__(self, force_exception=False):
#         dakota_input = DakotaInput(
#             environment=[
#                 "tabular_graphics_data",
#                 "output_precision = 8",
#             ],
#             method=[],
#             model=[
#                 "single",
#             ],
#             variables=[],
#             responses=[
#                 "num_objective_functions = 1",
#                 "analytic_gradients",
#                 "analytic_hessians",
#             ],
#             constraints=[]
#         )
#         super(TestDriver, self).__init__(dakota_input)

#         self.force_exception = force_exception

#         self.input.method = [
#             "conmin_mfd",
#             "  max_iterations = 100",
#             "  convergence_tolerance = 1e-6",
#         ]
#         self.input.variables = [
#             "continuous_design = 2",
#             "  cdv_initial_point  -1.2  1.0",
#             "  cdv_lower_bounds   -2.0 -2.0",
#             "  cdv_upper_bounds    2.0  2.0",
#             "  cdv_descriptor      'x1' 'x2'",
#         ]

#         self.input.constraints = [
#             "inequality_constraints = 2",
#             "  icv_descriptor = 'c1' 'c2'",
#             "  icv_lower_bounds = 0.0 0.0",
#         ]

#         self.input.responses = [
#             "num_objective_functions = 1",
#             "analytic_gradients",
#             "analytic_hessians",
#         ]
#         self.best_point = None
#         self.best_fun = None

#     def dakota_callback(self, **kwargs):
#         print("dakota_callback:")
#         cv = kwargs["cv"]
#         asv = kwargs["asv"]
#         print("    cv", cv)
#         print("    asv", asv)

#         x = cv
#         f0 = x[1] - x[0] * x[0]
#         f1 = 1 - x[0]

#         c1 = x[0] + x[1] + 10
#         c2 = -x[0] - x[1] + 10

#         retval = dict()
#         try:
#             if asv[0] & 1:
#                 f = [100 * f0 * f0 + f1 * f1]
#                 retval["fns"] = array(f)
#                 self.best_point = np.array(x)
#                 self.best_fun = f[0]

#             if asv[0] & 2:
#                 g = [[-400 * f0 * x[0] - 2 * f1, 200 * f0]]
#                 retval["fnGrads"] = array(g)

#             if asv[0] & 4:
#                 fx = x[1] - 3 * x[0] * x[0]
#                 h = [[[-400 * fx + 2, -400 * x[0]], [-400 * x[0], 200]]]
#                 retval["fnHessians"] = array(h)
                
#             retval["cons"] = array([c1, c2])
#             if self.force_exception:
#                 raise RuntimeError("Forced exception")

#         except Exception as exc:
#             print("    caught", exc)
#             raise

#         print("    returning", retval)
#         return retval


# def rosenbrock(x):
#     return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2


# def dummy_constraint1(x):
#     return x[0] + x[1] - 10  # <= 0


# def dummy_constraint2(x):
#     return -x[0] - x[1] - 10  # <= 0


# @pytest.mark.parametrize("x0", [np.array([-1.2, 1.0])])
# def test_rosenbrock_with_everest_minimize(x0):
#     constraints = [
#         {"type": "ineq", "fun": dummy_constraint1},
#         {"type": "ineq", "fun": dummy_constraint2},
#     ]
#     result = minimize(
#         fun=rosenbrock,
#         x0=x0,
#         method="conmin_mfd",
#         bounds=[(-2.0, 2.0), (-2.0, 2.0)],
#         constraints=constraints,
#         options={"ITMAX": 50},
#     )

#     expected_x = np.array([1.0, 1.0])
#     expected_fun = 0.0

#     assert result.success
#     assert np.allclose(result.x, expected_x, atol=1e-3)
#     assert np.isclose(result.fun, expected_fun, atol=1e-6)


# def test_dakota_conmin_optimization():
#     driver = TestDriver()
#     driver.run_dakota()

#     best_point = getattr(driver, "best_point", None)
#     if best_point is None:
#         raise RuntimeError("No best_point attribute found on driver - please add code to save the solution.")

#     assert np.allclose(best_point, [1.0, 1.0], atol=1e-4)

# @pytest.mark.parametrize("x0", [np.array([-1.2, 1.0])])
# def test_compare_dakota_and_everest_minimizers(x0):
#     constraints = [
#         {"type": "ineq", "fun": dummy_constraint1},
#         {"type": "ineq", "fun": dummy_constraint2},
#     ]
#     everest_result = minimize(
#         fun=rosenbrock,
#         x0=x0,
#         method="conmin_mfd",
#         bounds=[(-2.0, 2.0), (-2.0, 2.0)],
#         constraints=constraints,
#         options={"ITMAX": 50},
#     )
#     assert everest_result.success

#     driver = TestDriver()
#     driver.run_dakota()
#     dakota_best_point = getattr(driver, "best_point", None)
#     dakota_best_fun = getattr(driver, "best_fun", None)
#     if dakota_best_point is None or dakota_best_fun is None:
#         raise RuntimeError("Dakota driver did not record best solution.")

#     expected_x = np.array([1.0, 1.0])
#     expected_fun = 0.0

#     assert np.allclose(everest_result.x, expected_x, atol=1e-3)
#     assert np.allclose(dakota_best_point, expected_x, atol=1e-6)
    
#     assert np.isclose(everest_result.fun, expected_fun, atol=1e-6)
#     assert np.isclose(dakota_best_fun, expected_fun, atol=1e-6)
    
#     assert np.isclose(dakota_best_fun, everest_result.fun, atol=1e-6)
    
#     print("Dakota best point:", dakota_best_point)
#     print("Everest result x:", everest_result.x)
  
#     assert np.allclose(dakota_best_point, everest_result.x, atol=1e-3)