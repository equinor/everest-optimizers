#include "AllClassWrappers.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>

#include <string>

#include <functional>
#include <vector>

#include "MUQ/Optimization/NLoptOptimizer.h"
#include "MUQ/Optimization/Optimizer.h"
#include "MUQ/Modeling/ModPiece.h"

#include "MUQ/Utilities/PyDictConversion.h"

#include "MUQ/Modeling/Python/PyAny.h"

using namespace muq::Utilities;
using namespace muq::Optimization;
using namespace muq::Modeling;
namespace py = pybind11;

void PythonBindings::OptimizationWrapper(pybind11::module &m) {

  py::class_<Optimizer, std::shared_ptr<Optimizer>>(m,"Optimizer")
    .def_static("Construct", [](std::shared_ptr<ModPiece> const& cost, py::dict d){return Optimizer::Construct(cost, ConvertDictToPtree(d));})
    .def("ListMethods", &Optimizer::ListMethods, py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>())
    .def("AddInequalityConstraint", (void (Optimizer::*)(std::vector<std::shared_ptr<muq::Modeling::ModPiece>> const&)) &Optimizer::AddInequalityConstraint)
    .def("AddInequalityConstraint", (void (Optimizer::*)(std::shared_ptr<muq::Modeling::ModPiece> const&)) &Optimizer::AddInequalityConstraint)
    .def("ClearInequalityConstraint", &Optimizer::ClearInequalityConstraint)
    .def("AddEqualityConstraint", (void (Optimizer::*)(std::vector<std::shared_ptr<muq::Modeling::ModPiece>> const&)) &Optimizer::AddEqualityConstraint)
    .def("AddEqualityConstraint", (void (Optimizer::*)(std::shared_ptr<muq::Modeling::ModPiece> const&)) &Optimizer::AddEqualityConstraint)
    .def("ClearEqualityConstraint", &Optimizer::ClearEqualityConstraint)
    .def("Solve", &Optimizer::Solve, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

  py::class_<NLoptOptimizer, Optimizer, std::shared_ptr<NLoptOptimizer>>(m, "NLoptOptimizer")
    .def(py::init( [](std::shared_ptr<CostFunction> cost, py::dict d) { return new NLoptOptimizer(cost, ConvertDictToPtree(d)); }));
  
}
