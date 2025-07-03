#include "AllClassWrappers.h"
#include <Eigen/Sparse>

#include "MUQ/Inference/Filtering/KalmanFilter.h"
#include "MUQ/Inference/Filtering/KalmanSmoother.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

using namespace muq::Inference::PythonBindings;
using namespace muq::Inference;
using namespace muq::Modeling;
namespace py = pybind11;

void muq::Inference::PythonBindings::KalmanWrapper(pybind11::module &m)
{
    py::class_<KalmanFilter, std::shared_ptr<KalmanFilter>>(m, "KalmanFilter")
      .def_static("Analyze", static_cast<std::pair<Eigen::VectorXd,Eigen::MatrixXd> (*)
                       (std::pair<Eigen::VectorXd,Eigen::MatrixXd> const&,
                        std::shared_ptr<muq::Modeling::LinearOperator>,
                        Eigen::Ref<const Eigen::VectorXd> const&,
                        Eigen::Ref<const Eigen::MatrixXd> const&)>(&KalmanFilter::Analyze))
      .def_static("Analyze", static_cast<std::pair<Eigen::VectorXd,Eigen::MatrixXd> (*)
                       (std::pair<Eigen::VectorXd,Eigen::MatrixXd> const&,
                        Eigen::MatrixXd const&,
                        Eigen::Ref<const Eigen::VectorXd> const&,
                        Eigen::Ref<const Eigen::MatrixXd> const&)>(&KalmanFilter::Analyze<Eigen::MatrixXd>))
      .def_static("Analyze", static_cast<std::pair<Eigen::VectorXd,Eigen::MatrixXd> (*)
                       (std::pair<Eigen::VectorXd,Eigen::MatrixXd> const&,
                        Eigen::SparseMatrix<double> const&,
                        Eigen::Ref<const Eigen::VectorXd> const&,
                        Eigen::Ref<const Eigen::MatrixXd> const&)>(&KalmanFilter::Analyze<Eigen::SparseMatrix<double>>));

    py::class_<KalmanSmoother, std::shared_ptr<KalmanSmoother>>(m, "KalmanSmoother")
      .def_static("Analyze", static_cast<std::pair<Eigen::VectorXd,Eigen::MatrixXd> (*)
                       (std::pair<Eigen::VectorXd,Eigen::MatrixXd> const&,
                        std::pair<Eigen::VectorXd,Eigen::MatrixXd> const&,
                        std::pair<Eigen::VectorXd,Eigen::MatrixXd> const&,
                        std::shared_ptr<muq::Modeling::LinearOperator>)>(&KalmanSmoother::Analyze))
      .def_static("Analyze", static_cast<std::pair<Eigen::VectorXd,Eigen::MatrixXd> (*)
                       (std::pair<Eigen::VectorXd,Eigen::MatrixXd> const&,
                        std::pair<Eigen::VectorXd,Eigen::MatrixXd> const&,
                        std::pair<Eigen::VectorXd,Eigen::MatrixXd> const&,
                        Eigen::MatrixXd                            const&)>(&KalmanSmoother::Analyze))
      .def_static("Analyze", static_cast<std::pair<Eigen::VectorXd,Eigen::MatrixXd> (*)
                       (std::pair<Eigen::VectorXd,Eigen::MatrixXd> const&,
                        std::pair<Eigen::VectorXd,Eigen::MatrixXd> const&,
                        std::pair<Eigen::VectorXd,Eigen::MatrixXd> const&,
                        Eigen::SparseMatrix<double>                const&)>(&KalmanSmoother::Analyze));
}
