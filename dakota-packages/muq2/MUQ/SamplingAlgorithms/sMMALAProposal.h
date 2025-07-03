#ifndef SMMALAPROPOSAL_H_
#define SMMALAPROPOSAL_H_

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/ModPiece.h"

#include "MUQ/SamplingAlgorithms/MCMCProposal.h"

namespace muq {
  namespace SamplingAlgorithms {

    
    class SMMALAProposal : public MCMCProposal {
    public:

      SMMALAProposal(boost::property_tree::ptree                   pt,
                     std::shared_ptr<AbstractSamplingProblem> const& probIn,
                     std::shared_ptr<muq::Modeling::ModPiece> const& forwardModIn,
                     std::shared_ptr<muq::Modeling::Gaussian> const& priorIn,
                     std::shared_ptr<muq::Modeling::Gaussian> const& likelihoodIn);

      virtual ~SMMALAProposal() = default;

    protected:

      const double meanScaling = 0.5;

      double stepSize;

      /// The proposal distribution
      std::shared_ptr<muq::Modeling::Gaussian> prior;
      std::shared_ptr<muq::Modeling::Gaussian> likelihood;
      std::shared_ptr<muq::Modeling::ModPiece> model;


      virtual std::shared_ptr<SamplingState> Sample(std::shared_ptr<SamplingState> const& currentState) override;

      virtual double LogDensity(std::shared_ptr<SamplingState> const& currState,
                                std::shared_ptr<SamplingState> const& propState) override;


    };

  } // namespace SamplingAlgoirthms
} // namespace muq

#endif
