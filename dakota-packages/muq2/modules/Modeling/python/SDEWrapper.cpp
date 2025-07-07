#include "AllClassWrappers.h"

#include "MUQ/Utilities/PyDictConversion.h"

#include "MUQ/Modeling/LinearSDE.h"
#include <Eigen/Sparse>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

using namespace muq::Modeling;
using namespace muq::Utilities;
namespace py = pybind11;

void muq::Modeling::PythonBindings::SDEWrapper(py::module &m) {

  py::class_<LinearSDE, std::shared_ptr<LinearSDE> >(m, "LinearSDE")
    .def(py::init( [] (Eigen::MatrixXd const& F, Eigen::MatrixXd const& L, Eigen::MatrixXd const& Q, py::dict const& d) { return new LinearSDE(F,L,Q, ConvertDictToPtree(d)); }))
    .def(py::init( [] (Eigen::SparseMatrix<double> const& F, Eigen::MatrixXd const& L, Eigen::MatrixXd const& Q, py::dict const& d) { return new LinearSDE(F,L,Q, ConvertDictToPtree(d)); }))
    .def(py::init( [] (Eigen::MatrixXd const& F, Eigen::SparseMatrix<double> const& L, Eigen::MatrixXd const& Q, py::dict const& d) { return new LinearSDE(F,L,Q, ConvertDictToPtree(d)); }))
    .def(py::init( [] (Eigen::SparseMatrix<double> const& F, Eigen::SparseMatrix<double> const& L, Eigen::MatrixXd const& Q, py::dict const& d) { return new LinearSDE(F,L,Q, ConvertDictToPtree(d)); }))
    .def(py::init( [] (std::shared_ptr<muq::Modeling::LinearOperator> F, std::shared_ptr<muq::Modeling::LinearOperator> L, Eigen::MatrixXd const& Q, py::dict const& d) { return new LinearSDE(F,L,Q,ConvertDictToPtree(d)); }))
    .def("EvolveState", (void (LinearSDE::*)(py::EigenDRef<const Eigen::VectorXd> const&, double, py::EigenDRef<Eigen::VectorXd>) const) &LinearSDE::EvolveState)
    .def("EvolveState", (Eigen::VectorXd (LinearSDE::*)(py::EigenDRef<const Eigen::VectorXd> const&, double) const) &LinearSDE::EvolveState)
    //.def("EvolveDistribution", (void (LinearSDE::*)(std::pair<py::EigenDRef<Eigen::VectorXd>,py::EigenDRef<Eigen::MatrixXd>> const&, double) const) &LinearSDE::EvolveDistribution)
    .def("EvolveDistribution", (void (LinearSDE::*)(py::EigenDRef<const Eigen::VectorXd> const&, py::EigenDRef<const Eigen::MatrixXd> const&, double, py::EigenDRef<Eigen::VectorXd>, py::EigenDRef<Eigen::MatrixXd>) const) &LinearSDE::EvolveDistribution, py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert())
    .def("EvolveDistribution", (std::pair<Eigen::VectorXd, Eigen::MatrixXd> (LinearSDE::*)(std::pair<py::EigenDRef<const Eigen::VectorXd>,py::EigenDRef<const Eigen::MatrixXd>> const&, double) const) &LinearSDE::EvolveDistribution<py::EigenDRef<const Eigen::VectorXd>,py::EigenDRef<const Eigen::MatrixXd>>)
    .def("EvolveDistribution", (std::pair<Eigen::VectorXd, Eigen::MatrixXd> (LinearSDE::*)(py::EigenDRef<const Eigen::VectorXd> const&, py::EigenDRef<const Eigen::MatrixXd> const&, double) const) &LinearSDE::EvolveDistribution<py::EigenDRef<const Eigen::VectorXd>,py::EigenDRef<const Eigen::MatrixXd>>)
    .def("Discretize", &LinearSDE::Discretize)
    .def_static("Concatenate", &LinearSDE::Concatenate)
    .def("GetF", &LinearSDE::GetF)
    .def("GetL", &LinearSDE::GetL)
    .def("GetQ", &LinearSDE::GetQ)
    .def_readonly("stateDim", &LinearSDE::stateDim);
}
