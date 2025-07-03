#include "AllClassWrappers.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>

#include <string>

#include <functional>
#include <vector>

#include "MUQ/Optimization/ModPieceCostFunction.h"
#include "MUQ/Modeling/ModPiece.h"

using namespace muq::Modeling;
using namespace muq::Optimization;
namespace py = pybind11;

void PythonBindings::CostFunctionWrapper(py::module &m) {
  py::class_<CostFunction, ModPiece, std::shared_ptr<CostFunction> > cost(m, "CostFunction");
  cost
    .def("Cost", (double (CostFunction::*)()) &CostFunction::Cost)
    .def("Cost", (double (CostFunction::*)(Eigen::VectorXd const&)) &CostFunction::Cost)
    .def("Gradient", (Eigen::VectorXd (CostFunction::*)()) &CostFunction::Gradient)
    .def("Gradient", (Eigen::VectorXd (CostFunction::*)(Eigen::VectorXd const&)) &CostFunction::Gradient)
    .def("ApplyHessian", (Eigen::VectorXd (CostFunction::*)(Eigen::VectorXd const&)) &CostFunction::ApplyHessian)
    .def("ApplyHessian", (Eigen::VectorXd (CostFunction::*)(Eigen::VectorXd const&, Eigen::VectorXd const&)) &CostFunction::ApplyHessian)
    .def("Hessian", (Eigen::MatrixXd (CostFunction::*)()) &CostFunction::Hessian)
    .def("Hessian", (Eigen::MatrixXd (CostFunction::*)(Eigen::VectorXd const&)) &CostFunction::Hessian);

  py::class_<ModPieceCostFunction, CostFunction, std::shared_ptr<ModPieceCostFunction> > modCost(m, "ModPieceCostFunction");
  modCost
    .def(py::init<std::shared_ptr<ModPiece>>())
    .def(py::init<std::shared_ptr<ModPiece>,double>());//( [](std::shared_ptr<ModPiece> cost) { return new ModPieceCostFunction(cost); }));
}
