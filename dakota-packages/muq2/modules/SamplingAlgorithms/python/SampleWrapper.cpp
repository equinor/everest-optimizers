#include "AllClassWrappers.h"

#include "MUQ/SamplingAlgorithms/SampleCollection.h"
#include "MUQ/SamplingAlgorithms/MarkovChain.h"
#include "MUQ/SamplingAlgorithms/SampleEstimator.h"
#include "MUQ/SamplingAlgorithms/MultiIndexEstimator.h"
#include "MUQ/SamplingAlgorithms/SamplingState.h"
#include "MUQ/SamplingAlgorithms/Diagnostics.h"

#include "MUQ/Utilities/AnyHelpers.h"

#include "MUQ/Utilities/PyDictConversion.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <string>

#include <functional>
#include <vector>

using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

namespace py = pybind11;

void PythonBindings::SampleWrapper(py::module &m)
{
  py::class_<SamplingStateIdentity, std::shared_ptr<SamplingStateIdentity>> ssID(m, "SamplingStateIdentity");
  ssID
    .def(py::init<int>())
    .def_readonly("blockInd", &SamplingStateIdentity::blockInd);

  // py::class_<SamplingStatePartialMoment, std::shared_ptr<SamplingStatePartialMoment>> ssParMom(m, "SamplingStatePartialMoment");
  // ssParMom
  //   .def(py::init<int, int, Eigen::VectorXd const&>())
  //   .def_readonly("blockInd", &SamplingStatePartialMoment::blockInd)
  //   .def_readonly("momentPower", &SamplingStatePartialMoment::momentPower);
  //   //.def_readonly("mu", &SamplingStatePartialMoment::mu);

  py::class_<SampleEstimator, std::shared_ptr<SampleEstimator>>(m,"SampleEstimator")
    .def("CentralMoment", (Eigen::VectorXd (SampleEstimator::*)(unsigned int, int) const) &SampleEstimator::CentralMoment, py::arg("order"), py::arg("blockDim") = -1)
    .def("CentralMoment", (Eigen::VectorXd (SampleEstimator::*)(unsigned int, Eigen::VectorXd const&, int) const) &SampleEstimator::CentralMoment, py::arg("order"), py::arg("mean"), py::arg("blockDim") = -1)
    .def("Mean", &SampleEstimator::Mean, py::arg("blockDim") = -1)
    .def("Variance", (Eigen::VectorXd (SampleEstimator::*)(int) const) &SampleEstimator::Variance, py::arg("blockDim") = -1)
    .def("Variance", (Eigen::VectorXd (SampleEstimator::*)(Eigen::VectorXd const&, int) const) &SampleEstimator::Variance, py::arg("mean"), py::arg("blockDim") = -1)
    .def("Covariance", (Eigen::MatrixXd (SampleEstimator::*)(int) const) &SampleEstimator::Covariance, py::arg("blockDim") = -1)
    .def("Covariance", (Eigen::MatrixXd (SampleEstimator::*)(Eigen::VectorXd const&, int) const) &SampleEstimator::Covariance, py::arg("mean"), py::arg("blockDim") = -1)
    .def("StandardizedMoment", (Eigen::VectorXd (SampleEstimator::*)(unsigned int, int) const) &SampleEstimator::CentralMoment, py::arg("order"), py::arg("blockDim") = -1)
    .def("StandardizedMoment", (Eigen::VectorXd (SampleEstimator::*)(unsigned int, Eigen::VectorXd const&, int) const) &SampleEstimator::StandardizedMoment, py::arg("order"), py::arg("mean"), py::arg("blockDim") = -1)
    .def("StandardizedMoment", (Eigen::VectorXd (SampleEstimator::*)(unsigned int, Eigen::VectorXd const&, Eigen::VectorXd const&, int) const) &SampleEstimator::StandardizedMoment, py::arg("order"), py::arg("mean"),py::arg("stdDev"), py::arg("blockDim") = -1)
    .def("Skewness", (Eigen::VectorXd (SampleEstimator::*)(int) const) &SampleEstimator::Skewness, py::arg("blockDim") = -1)
    .def("Skewness", (Eigen::VectorXd (SampleEstimator::*)(Eigen::VectorXd const&, int) const) &SampleEstimator::Skewness, py::arg("mean"), py::arg("blockDim") = -1)
    .def("Skewness", (Eigen::VectorXd (SampleEstimator::*)(Eigen::VectorXd const&, Eigen::VectorXd const&, int) const) &SampleEstimator::Skewness, py::arg("mean"),py::arg("stdDev"), py::arg("blockDim") = -1)
    .def("Kurtosis", (Eigen::VectorXd (SampleEstimator::*)(int) const) &SampleEstimator::Kurtosis, py::arg("blockDim") = -1)
    .def("Kurtosis", (Eigen::VectorXd (SampleEstimator::*)(Eigen::VectorXd const&, int) const) &SampleEstimator::Kurtosis, py::arg("mean"), py::arg("blockDim") = -1)
    .def("Kurtosis", (Eigen::VectorXd (SampleEstimator::*)(Eigen::VectorXd const&, Eigen::VectorXd const&, int) const) &SampleEstimator::Kurtosis, py::arg("mean"),py::arg("stdDev"), py::arg("blockDim") = -1)
    .def("ExpectedValue", &SampleEstimator::ExpectedValue, py::arg("f"), py::arg("metasIn")=std::vector<std::string>())
    .def("ESS", (Eigen::VectorXd (SampleEstimator::*)(int, std::string const&) const) &SampleEstimator::ESS, py::arg("blockDim")=-1, py::arg("method")="Batch")
    .def("StandardError", (Eigen::VectorXd (SampleEstimator::*)(int, std::string const&) const) &SampleEstimator::StandardError, py::arg("blockDim")=-1, py::arg("method")="Batch");
    

  py::class_<MultiIndexEstimator, SampleEstimator, std::shared_ptr<MultiIndexEstimator>>(m, "MultiIndexEstimator")
    .def(py::init<std::vector<std::shared_ptr<MIMCMCBox>>>());

  py::class_<SampleCollection, SampleEstimator, std::shared_ptr<SampleCollection>> sampColl(m, "SampleCollection");
  sampColl
    .def(py::init<>())
    .def("__getitem__", (const std::shared_ptr<SamplingState> (SampleCollection::*)(unsigned) const) &SampleCollection::at)
//    .def("at", &SampleCollection::at)
    .def("size", &SampleCollection::size)
    .def("RunningCovariance", (std::vector<Eigen::MatrixXd> (SampleCollection::*)(Eigen::VectorXd const&, int) const) &SampleCollection::RunningCovariance, py::arg("mean"), py::arg("blockDim") = -1)
    .def("RunningCovariance", (std::vector<Eigen::MatrixXd> (SampleCollection::*)(int) const) &SampleCollection::RunningCovariance, py::arg("blockDim") = -1)
    .def("Add", &SampleCollection::Add)
    .def("Weights", &SampleCollection::Weights)
    .def("AsMatrix", &SampleCollection::AsMatrix, py::arg("blockDim")=-1)
    .def_static("FromMatrix", &SampleCollection::FromMatrix)
    .def("GetMeta", (Eigen::MatrixXd (SampleCollection::*)(std::string const&) const) &SampleCollection::GetMeta)
    .def("ListMeta", &SampleCollection::ListMeta, py::arg("requireAll")=true)
    .def("WriteToFile", (void (SampleCollection::*)(std::string const&, std::string const&) const) &SampleCollection::WriteToFile, py::arg("filename"), py::arg("dataset") = "/")
    .def("head", &SampleCollection::head)
    .def("tail", &SampleCollection::tail)
    .def("segment", &SampleCollection::segment, py::arg("startInd"),py::arg("length"),py::arg("skipBy")=1)
    .def("BatchESS", &SampleCollection::BatchESS, py::arg("blockDim")=-1, py::arg("batchSize")=-1, py::arg("overlap")=-1)
    .def("MultiBatchESS", &SampleCollection::MultiBatchESS, py::arg("blockDim")=-1, py::arg("batchSize")=-1, py::arg("overlap")=-1)
    .def("BatchError", &SampleCollection::BatchError, py::arg("blockDim")=-1, py::arg("batchSize")=-1, py::arg("overlap")=-1)
    .def("MultiBatchError", &SampleCollection::MultiBatchError, py::arg("blockDim")=-1, py::arg("batchSize")=-1, py::arg("overlap")=-1);

  py::class_<MarkovChain, SampleCollection, SampleEstimator, std::shared_ptr<MarkovChain>>(m,"MarkovChain")
    .def(py::init<>())
    .def("WolfESS", &MarkovChain::WolffESS, py::arg("blockInd")=-1)
    .def("WolfError", &MarkovChain::WolffError, py::arg("blockInd")=-1)
    .def_static("SingleComponentWolffESS", &MarkovChain::SingleComponentWolffESS);

  m.def_submodule("Diagnostics")
    .def("Rhat", [](std::vector<std::shared_ptr<SampleCollection>> const& collections){return Diagnostics::Rhat(collections);})
    .def("Rhat", [](std::vector<std::shared_ptr<SampleCollection>> const& collections, py::dict opts){return Diagnostics::Rhat(collections, ConvertDictToPtree(opts));})
    .def("Rhat", [](std::vector<std::shared_ptr<MarkovChain>> const& collections){return Diagnostics::Rhat(collections);})
    .def("Rhat", [](std::vector<std::shared_ptr<MarkovChain>> const& collections, py::dict opts){return Diagnostics::Rhat(collections, ConvertDictToPtree(opts));})
    .def("Rhat", [](std::vector<std::shared_ptr<MultiIndexEstimator>> const& collections){return Diagnostics::Rhat(collections);})
    .def("Rhat", [](std::vector<std::shared_ptr<MultiIndexEstimator>> const& collections, py::dict opts){return Diagnostics::Rhat(collections, ConvertDictToPtree(opts));})
    .def("BasicRhat", &Diagnostics::BasicRhat)
    .def("BasicMPSRF", &Diagnostics::BasicMPSRF)
    .def("SplitChains", [](std::vector<std::shared_ptr<SampleCollection>> const& origChains, unsigned int numSegments){return Diagnostics::SplitChains(origChains,numSegments);})
    .def("TransformChains", &Diagnostics::TransformChains)
    .def("ComputeRanks", &Diagnostics::ComputeRanks);


  py::class_<SamplingState, std::shared_ptr<SamplingState>> sampState(m, "SamplingState");
  sampState
    .def(py::init<Eigen::VectorXd const&>())
    .def(py::init<Eigen::VectorXd const&, double>())
    .def(py::init<std::vector<Eigen::VectorXd> const&>())
    .def(py::init<std::vector<Eigen::VectorXd> const&, double>())
    .def_readonly("weight", &SamplingState::weight)
    .def_readonly("state", &SamplingState::state)
    .def("HasMeta", &SamplingState::HasMeta)
    .def("GetMeta", [](std::shared_ptr<SamplingState> self, std::string const& metaKey)
                                  -> boost::any& {
                                     return self->meta.at(metaKey);
                                  })
    .def("GetMetaSamplingState", [](std::shared_ptr<SamplingState> self, std::string const& metaKey)
                                  -> std::shared_ptr<SamplingState> {
                                     return muq::Utilities::AnyCast(self->meta.at(metaKey));
                                  })
    .def("TotalDim", &SamplingState::TotalDim)
    .def("ToVector", &SamplingState::ToVector,py::arg("blockInd")=-1);
}
