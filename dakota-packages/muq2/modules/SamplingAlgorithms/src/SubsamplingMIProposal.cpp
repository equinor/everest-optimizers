#include "MUQ/SamplingAlgorithms/SubsamplingMIProposal.h"
namespace muq {
  namespace SamplingAlgorithms {

    SubsamplingMIProposal::SubsamplingMIProposal (pt::ptree const& pt, std::shared_ptr<AbstractSamplingProblem> prob, std::shared_ptr<muq::Utilities::MultiIndex> coarseIndex, std::shared_ptr<SingleChainMCMC> coarseChain)
     : MCMCProposal(pt,prob),
       coarseChain(coarseChain),
       subsampling(pt.get<int>("MLMCMC.Subsampling" + multiindexToConfigString(coarseIndex)))
    {}

    std::shared_ptr<SamplingState> SubsamplingMIProposal::Sample(std::shared_ptr<SamplingState> const& currentState) {

      sampleID += subsampling+1;
      while (coarseChain->GetSamples()->size() <= sampleID) {
        coarseChain->AddNumSamps(1);
        coarseChain->Run();
      }

      return coarseChain->GetSamples()->at(sampleID);
    }

    double SubsamplingMIProposal::LogDensity(std::shared_ptr<SamplingState> const& currState,
                                             std::shared_ptr<SamplingState> const& propState) {
      return 0;
    }

    std::string SubsamplingMIProposal::multiindexToConfigString (std::shared_ptr<muq::Utilities::MultiIndex> index) {
      std::stringstream strs;
      for (int i = 0; i < index->GetLength(); i++) {
        strs << "_" << index->GetValue(i);
      }
      return strs.str();
    }

  }
}
