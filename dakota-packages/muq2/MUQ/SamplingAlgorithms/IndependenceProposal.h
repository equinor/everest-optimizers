#ifndef INDEPENDENCEPROPOSAL_H_
#define INDEPENDENCEPROPOSAL_H_

#include "MUQ/Modeling/Distributions/Distribution.h"

#include "MUQ/SamplingAlgorithms/MCMCProposal.h"

namespace muq {
  namespace SamplingAlgorithms {

    /** @ingroup MCMCProposals
        @class IndependenceProposal
        @brief Implementation of an independence proposal with arbitrary distribution.  
        @details 
            If not explicitly given a proposal distribution, this class will assume a 
            zero mean Gaussian distribution with isotropic variance.  The variance of
            the Gaussian proposal can be set with the options ptree.

        <B>Configuration Parameters:</B>

        Parameter Key | Type | Default Value | Description |
        ------------- | ------------- | ------------- | ------------- |
        "ProposalVariance" | Double | - | The variance of an isotropic zero mean Gaussian proposal distribution. |
        "BlockIndex"  | Int | 0 | The block of the sampling problem that this proposal should act on. |
    */
    class IndependenceProposal : public MCMCProposal {
    public:

      IndependenceProposal(boost::property_tree::ptree              const& pt,
                           std::shared_ptr<AbstractSamplingProblem> const& prob);

      IndependenceProposal(boost::property_tree::ptree              const& pt,
                           std::shared_ptr<AbstractSamplingProblem> const& prob,
                           std::shared_ptr<muq::Modeling::Distribution>    proposalIn);

      virtual ~IndependenceProposal() = default;

    protected:

      /// The proposal distribution
      std::shared_ptr<muq::Modeling::Distribution> proposal;

      virtual std::shared_ptr<SamplingState> Sample(std::shared_ptr<SamplingState> const& currentState) override;

      virtual double LogDensity(std::shared_ptr<SamplingState> const& currState,
                                std::shared_ptr<SamplingState> const& propState) override;

      /** Constructs a distribution using the information in the options ptree. */
      static std::shared_ptr<muq::Modeling::Distribution> ExtractDistribution(boost::property_tree::ptree const& opts,
                                                                              std::shared_ptr<AbstractSamplingProblem> const& prob);


    };

  } // namespace SamplingAlgoirthms
} // namespace muq

#endif
