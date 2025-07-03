#include "AllClassWrappers.h"

#include "MUQ/SamplingAlgorithms/AMProposal.h"
#include "MUQ/SamplingAlgorithms/MCMCProposal.h"
#include "MUQ/SamplingAlgorithms/MALAProposal.h"
#include "MUQ/SamplingAlgorithms/sMMALAProposal.h"
#include "MUQ/SamplingAlgorithms/InfMALAProposal.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/CrankNicolsonProposal.h"
#include "MUQ/SamplingAlgorithms/InverseGammaProposal.h"
#include "MUQ/SamplingAlgorithms/IndependenceProposal.h"

#include "MUQ/Utilities/PyDictConversion.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <string>

#include <functional>
#include <vector>

//using namespace muq::SamplingAlgorithms::PythonBindings;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;
namespace py = pybind11;

void PythonBindings::ProposalWrapper(py::module &m) {
  py::class_<MCMCProposal, std::shared_ptr<MCMCProposal>> mcmcPro(m, "MCMCProposal");
  mcmcPro
    .def("Sample", &MCMCProposal::Sample)
    .def("LogDensity", &MCMCProposal::LogDensity)
    .def("GetBlockInd", &MCMCProposal::GetBlockInd)
    .def("SetBlockInd", &MCMCProposal::SetBlockInd);
    //.def("Construct", &MCMCProposal::Construct)
    //.def("GetMCMCProposalMap", &MCMCProposal::GetMCMCProposalMap);

  py::class_<MHProposal, MCMCProposal, std::shared_ptr<MHProposal>> mhPro(m, "MHProposal");
  mhPro
    .def(py::init([](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob) {return new MHProposal(ConvertDictToPtree(d), prob);} ))
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob, std::shared_ptr<muq::Modeling::GaussianBase> gauss) { return new MHProposal(ConvertDictToPtree(d), prob, gauss);}));


  py::class_<CrankNicolsonProposal, MCMCProposal, std::shared_ptr<CrankNicolsonProposal>> cnPro(m, "CrankNicolsonProposal");
  cnPro
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob) {return new CrankNicolsonProposal(ConvertDictToPtree(d), prob);} ))
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob, std::shared_ptr<muq::Modeling::GaussianBase> gauss) { return new CrankNicolsonProposal(ConvertDictToPtree(d), prob, gauss);}));

  py::class_<InverseGammaProposal, MCMCProposal, std::shared_ptr<InverseGammaProposal>>(m, "InverseGammaProposal")
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob) {return new InverseGammaProposal(ConvertDictToPtree(d), prob);} ));

  py::class_<AMProposal, MCMCProposal, std::shared_ptr<AMProposal>> amPro(m, "AMProposal");
  amPro
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob) {return new AMProposal(ConvertDictToPtree(d), prob);} ))
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob, Eigen::MatrixXd const& cov) {return new AMProposal(ConvertDictToPtree(d), prob, cov);} ))
    .def("Adapt", &AMProposal::Adapt);

  py::class_<MALAProposal, MCMCProposal, std::shared_ptr<MALAProposal>>(m,"MALAProposal")
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob) {return new MALAProposal(ConvertDictToPtree(d), prob);} ))
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob, std::shared_ptr<muq::Modeling::GaussianBase> const& prop) {return new MALAProposal(ConvertDictToPtree(d), prob, prop);} ));

  py::class_<SMMALAProposal, MCMCProposal, std::shared_ptr<SMMALAProposal>>(m,"SMMALAProposal")
         .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob, std::shared_ptr<muq::Modeling::ModPiece> const& mod, std::shared_ptr<muq::Modeling::Gaussian> const& prior,std::shared_ptr<muq::Modeling::Gaussian> const& likelihood) {return new SMMALAProposal(ConvertDictToPtree(d), prob, mod,prior,likelihood);} ));

  py::class_<InfMALAProposal, MCMCProposal, std::shared_ptr<InfMALAProposal>>(m, "InfMALAProposal")
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob) {return new InfMALAProposal(ConvertDictToPtree(d), prob);} ))
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob, std::shared_ptr<muq::Modeling::GaussianBase> prop) {return new InfMALAProposal(ConvertDictToPtree(d), prob, prop);} ));

  py::class_<IndependenceProposal, MCMCProposal, std::shared_ptr<IndependenceProposal>>(m,"IndependenceProposal")
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob) {return new IndependenceProposal(ConvertDictToPtree(d), prob);} ))
    .def(py::init( [](py::dict d, std::shared_ptr<AbstractSamplingProblem> prob, std::shared_ptr<muq::Modeling::Distribution> dist) {return new IndependenceProposal(ConvertDictToPtree(d), prob, dist);} ));
    
}
