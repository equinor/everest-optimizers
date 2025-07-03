#include "AllClassWrappers.h"

#include "MUQ/config.h"

#include "MUQ/SamplingAlgorithms/MIMCMC.h"
#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/SamplingAlgorithm.h"
#include "MUQ/SamplingAlgorithms/MCMCFactory.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/ParallelTempering.h"

#include "MUQ/SamplingAlgorithms/ConcatenatingInterpolation.h"
#include "MUQ/Utilities/PyDictConversion.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>

#include <string>

#include <functional>
#include <vector>

using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;
namespace py = pybind11;

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/SamplingAlgorithms/CrankNicolsonProposal.h"
#include "MUQ/SamplingAlgorithms/SubsamplingMIProposal.h"
#include "MUQ/SamplingAlgorithms/DefaultComponentFactory.h"

using namespace muq::Modeling;


void PythonBindings::MCMCWrapper(py::module &m) {

  py::class_<SingleChainMCMC, std::shared_ptr<SingleChainMCMC>> singleMCMC(m, "SingleChainMCMC");
  singleMCMC
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> problem) {return new SingleChainMCMC(ConvertDictToPtree(d), problem);}))
    .def(py::init( [](py::dict d, std::vector<std::shared_ptr<TransitionKernel>> kernels) {return new SingleChainMCMC(ConvertDictToPtree(d), kernels);}))
    .def("SetState", (void (SingleChainMCMC::*)(std::shared_ptr<SamplingState> const&)) &SingleChainMCMC::SetState)
    .def("SetState", (void (SingleChainMCMC::*)(std::vector<Eigen::VectorXd> const&)) &SingleChainMCMC::SetState)
    .def("Kernels", &SingleChainMCMC::Kernels)
    .def("Run", (std::shared_ptr<MarkovChain> (SingleChainMCMC::*)(std::vector<Eigen::VectorXd> const&)) &SingleChainMCMC::Run, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("AddNumSamps", &SingleChainMCMC::AddNumSamps)
    .def("NumSamps", &SingleChainMCMC::NumSamps)
    .def("TotalTime", &SingleChainMCMC::TotalTime)
    .def("GetSamples", &SingleChainMCMC::GetSamples)
    .def("GetQOIs", &SingleChainMCMC::GetQOIs);
  
  py::class_<ParallelTempering, std::shared_ptr<ParallelTempering>>(m,"ParallelTempering")
    .def(py::init( [](py::dict d, std::shared_ptr<InferenceProblem> problem) {return new ParallelTempering(ConvertDictToPtree(d), problem);}))
    .def(py::init( [](py::dict d, Eigen::VectorXd const& invTemps, std::vector<std::shared_ptr<TransitionKernel>> kerns) {return new ParallelTempering(ConvertDictToPtree(d), invTemps, kerns);}))
    .def(py::init( [](py::dict d, Eigen::VectorXd const& invTemps, std::vector<std::vector<std::shared_ptr<TransitionKernel>>> kerns) {return new ParallelTempering(ConvertDictToPtree(d), invTemps, kerns);}))
    .def("SetState", (void (ParallelTempering::*)(std::vector<std::shared_ptr<SamplingState>> const&)) &ParallelTempering::SetState)
    .def("SetState", (void (ParallelTempering::*)(std::vector<Eigen::VectorXd> const&)) &ParallelTempering::SetState)
    .def("SetState", (void (ParallelTempering::*)(std::vector<std::vector<Eigen::VectorXd>> const&)) &ParallelTempering::SetState)
    .def("GetInverseTemp", &ParallelTempering::GetInverseTemp)
    .def("Kernels", &ParallelTempering::Kernels)
    .def("Run", (std::shared_ptr<MarkovChain> (ParallelTempering::*)(Eigen::VectorXd const&)) &ParallelTempering::Run)
    .def("Run", (std::shared_ptr<MarkovChain> (ParallelTempering::*)(std::vector<Eigen::VectorXd> const&)) &ParallelTempering::Run)
    .def("Run", (std::shared_ptr<MarkovChain> (ParallelTempering::*)(std::vector<std::vector<Eigen::VectorXd>> const&)) &ParallelTempering::Run)
    .def("AddNumSamps", &ParallelTempering::AddNumSamps)
    .def("NumSamps", &ParallelTempering::NumSamps)
    .def("GetSamples", &ParallelTempering::GetSamples)
    .def("GetQOISs", &ParallelTempering::GetQOIs)
    .def_readonly("numTemps", &ParallelTempering::numTemps);
    

  py::class_<MIMCMCBox, std::shared_ptr<MIMCMCBox>> multiindexMCMCBox(m, "MIMCMCBox");
  multiindexMCMCBox
    .def("FinestChain", &MIMCMCBox::FinestChain)
    .def("GetChain", &MIMCMCBox::GetChain)
    .def("GetBoxIndices", &MIMCMCBox::GetBoxIndices)
    .def("GetHighestIndex", &MIMCMCBox::GetHighestIndex);


  py::class_<MIMCMC, std::shared_ptr<MIMCMC>> multiindexMCMC(m, "MIMCMC");
  multiindexMCMC
    .def(py::init( [](py::dict d, Eigen::VectorXd startingPoint, std::vector<std::shared_ptr<AbstractSamplingProblem>> const& problems) {return new MIMCMC(ConvertDictToPtree(d), startingPoint, problems); }))
    .def(py::init( [](py::dict d, Eigen::VectorXd startingPoint, std::vector<std::shared_ptr<ModPiece>> const& models) {return new MIMCMC(ConvertDictToPtree(d), startingPoint, models); }))
    .def(py::init( [](py::dict d, Eigen::VectorXd startingPoint, std::vector<std::shared_ptr<AbstractSamplingProblem>> const& problems, std::shared_ptr<MultiIndexSet> const& indices) {return new MIMCMC(ConvertDictToPtree(d), startingPoint, problems, indices); }))
    .def(py::init( [](py::dict d, Eigen::VectorXd startingPoint, std::vector<std::shared_ptr<ModPiece>> const& models, std::shared_ptr<MultiIndexSet> const& indices) {return new MIMCMC(ConvertDictToPtree(d), startingPoint, models, indices); }))
    .def("Run", &MIMCMC::Run, py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
    .def("GetSamples", &MIMCMC::GetSamples)
    .def("GetQOIs", &MIMCMC::GetQOIs)
    .def("GetIndices", &MIMCMC::GetIndices)
    .def("GetMIMCMCBox", &MIMCMC::GetMIMCMCBox);

  py::class_<MCMCFactory, std::shared_ptr<MCMCFactory>> fact(m, "MCMCFactory");
  fact
    .def_static("CreateSingleChain", [](py::dict d, std::shared_ptr<AbstractSamplingProblem> problem) {return MCMCFactory::CreateSingleChain(ConvertDictToPtree(d), problem);},
                py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>() );


}
