#ifndef INFMALAPROPOSAL_H_
#define INFMALAPROPOSAL_H_

#include "MUQ/Modeling/Distributions/GaussianBase.h"

#include "MUQ/SamplingAlgorithms/MCMCProposal.h"

namespace muq {
  namespace SamplingAlgorithms {

    /**
    @ingroup MCMCProposals
    @class InfMALAProposal
    @brief An implement of the dimension-independent MALA (or Inf-MALA) proposal.

    @details
    This class implements the \f$\infty-\f$MALA proposal (and its
    geometry-aware version: \f$\infty-\f$mMALA proposal) described in Beskos et
    al., 2017.   The proposal takes the form
    \f[
    u^\prime = \rho u + \sqrt{1 - \rho^2} \frac{\sqrt{h}}{2} \left\{ u - K
    C^{-1} (u - u_0) - K \nabla \Phi \right\} + \sqrt{1-\rho} z
    \f]
    where \f$u\f$ is the current state of the chain, \f$\rho = \frac{4 - h}{4 +
    h}\f$ with \f$h\f$ being the step size used to discretize the Langevin SDE,
    \f$u_0\f$ is the prior mean, \f$C\f$ is the prior covariance, \f$\Phi\f$ is
    the negagive log likelihood (note that \f$\nabla \pi_\text{post} = - C^{-1}
    (u - u_0) - \nabla \Phi\f$), \f$z\sim N(0,K)\f$ is a normal random
    variable with a strategically chosen covariance \f$K\f$, and \f$u^\prime\f$
    is the propsed point.  
    
    Note that the above proposal is the geometry-aware version, and it is the
    \f$\infty-\f$MALA proposal when \f$K = C\f$.

    <B>Configuration Parameters:</B>
    Parameter Key | Type | Default Value | Description |
    ------------- | ------------- | ------------- | ------------- |
    "StepSize" | double | 1.0 | The step size used to discrete the Langevin SDE. |
    */
    class InfMALAProposal : public MCMCProposal {
    public:

      /**
      Construct the Inf-mMALA proposal with identity covariance for \f$K\f$.
      */
      InfMALAProposal(boost::property_tree::ptree           const& pt,
                      std::shared_ptr<AbstractSamplingProblem>     prob);

      /**
      Construct the Inf-mMALA proposal with strategically chosen Gaussian
      distribution zDistIn for \f$z\f$.
      */
      InfMALAProposal(boost::property_tree::ptree           const& pt,
                      std::shared_ptr<AbstractSamplingProblem>     prob,
                      std::shared_ptr<muq::Modeling::GaussianBase> zDistIn);

      virtual ~InfMALAProposal() = default;

    protected:

      double stepSize;
      
      double rho;

      // The proposal distribution
      std::shared_ptr<muq::Modeling::GaussianBase> zDist;

      virtual std::shared_ptr<SamplingState> Sample(std::shared_ptr<SamplingState> const& currentState) override;

      virtual double LogDensity(std::shared_ptr<SamplingState> const& currState,
                                std::shared_ptr<SamplingState> const& propState) override;


      /** Returns the product of the proposal covariance times gradient of the
          log target density at the current state.  Checks the metadata of the
          currentstate first to see if this has already been computed.
      */
      Eigen::VectorXd GetSigmaGrad(std::shared_ptr<SamplingState> const& currentState) const;


    }; // class InfMALAProposal
  } // namespace SamplingAlgoirthms
} // namespace muq

#endif
