#include "MUQ/SamplingAlgorithms/InfMALAProposal.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"

#include "MUQ/Utilities/AnyHelpers.h"

using namespace muq::SamplingAlgorithms;
using namespace muq::Modeling;
using namespace muq::Utilities;

REGISTER_MCMC_PROPOSAL(InfMALAProposal)

InfMALAProposal::InfMALAProposal(boost::property_tree::ptree       const& pt,
                                 std::shared_ptr<AbstractSamplingProblem> prob) : 
                                 MCMCProposal(pt,prob),
                                 stepSize(pt.get("StepSize",1.0))
{
  rho = (4.0 - stepSize) / (4.0 + stepSize);
  
  unsigned int dim = prob->blockSizes(blockInd);
  
  const Eigen::VectorXd cov = Eigen::VectorXd::Ones(dim);
  zDist = std::make_shared<Gaussian>(Eigen::VectorXd::Zero(dim), cov);
}



InfMALAProposal::InfMALAProposal(boost::property_tree::ptree       const& pt,
                                 std::shared_ptr<AbstractSamplingProblem> prob,
                                 std::shared_ptr<GaussianBase>            zDistIn) : 
                                 MCMCProposal(pt,prob),
                                 zDist(zDistIn),
                                 stepSize(pt.get("StepSize",1.0))
{
  rho = (4.0 - stepSize) / (4.0 + stepSize);
}

std::shared_ptr<SamplingState> InfMALAProposal::Sample(std::shared_ptr<SamplingState> const& currentState)
{
  assert(currentState->state.size()>blockInd);

  // the mean of the proposal is the current point
  std::vector<Eigen::VectorXd> props = currentState->state;
  assert(props.size()>blockInd);
  Eigen::VectorXd const& uc = currentState->state.at(blockInd);

  // vector holding the covariance of the proposal times the gradient
  Eigen::VectorXd sigmaGrad = GetSigmaGrad(currentState);
  
  props.at(blockInd) = rho*uc + sqrt(1 - rho*rho)*(0.5*sqrt(stepSize)*(uc + sigmaGrad) + zDist->Sample());

  // store the new state in the output
  return std::make_shared<SamplingState>(props, 1.0);
}

double InfMALAProposal::LogDensity(std::shared_ptr<SamplingState> const& currState,
                                   std::shared_ptr<SamplingState> const& propState)
{
  double const beta = sqrt(1 - rho*rho);
  Eigen::VectorXd sigmaGrad = GetSigmaGrad(currState);
  Eigen::VectorXd mean = rho*currState->state.at(blockInd) + beta*0.5*sqrt(stepSize)*(currState->state.at(blockInd) + sigmaGrad);
  Eigen::VectorXd diff = (propState->state.at(blockInd) - mean)/beta;

  return zDist->LogDensity(diff);

}

Eigen::VectorXd InfMALAProposal::GetSigmaGrad(std::shared_ptr<SamplingState> const& state) const
{
  std::stringstream blockId;
  blockId << "_" << blockInd;

  if(!state->HasMeta("MALA_SigmaGrad" + blockId.str())){
    Eigen::VectorXd grad; // vector holding the gradient of the log density

    if(!state->HasMeta("GradLogDensity" + blockId.str()))
      state->meta["GradLogDensity" + blockId.str()] = prob->GradLogDensity(state, blockInd);

    grad = AnyCast( state->meta["GradLogDensity" + blockId.str()] );

    state->meta["MALA_SigmaGrad" + blockId.str()] = zDist->ApplyCovariance(grad).col(0).eval();
  }

  Eigen::VectorXd sigmaGrad = AnyCast(state->meta["MALA_SigmaGrad" + blockId.str()]);
  return sigmaGrad;
}
