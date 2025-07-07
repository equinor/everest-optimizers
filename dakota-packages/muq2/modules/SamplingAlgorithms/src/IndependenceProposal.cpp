#include "MUQ/SamplingAlgorithms/IndependenceProposal.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"

#include "MUQ/Utilities/AnyHelpers.h"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

REGISTER_MCMC_PROPOSAL(IndependenceProposal)

IndependenceProposal::IndependenceProposal(pt::ptree                                const& pt,
                                           std::shared_ptr<AbstractSamplingProblem> const& prob) : IndependenceProposal(pt,prob, ExtractDistribution(pt, prob))
{}

IndependenceProposal::IndependenceProposal(pt::ptree            const& pt,
                       std::shared_ptr<AbstractSamplingProblem> const& prob,
                       std::shared_ptr<Distribution>                   dist) : MCMCProposal(pt,prob),
                                                                               proposal(dist)  
{
}

std::shared_ptr<Distribution> IndependenceProposal::ExtractDistribution(pt::ptree const& opts,
                                                  std::shared_ptr<AbstractSamplingProblem> const& prob)
{

  Eigen::VectorXd mu = Eigen::VectorXd::Zero(prob->blockSizes(opts.get("BlockIndex",0)));
  Eigen::VectorXd var = opts.get("ProposalVariance", 1.0)*Eigen::VectorXd::Ones(mu.size());
  
  return std::make_shared<Gaussian>(mu,var);
}

std::shared_ptr<SamplingState> IndependenceProposal::Sample(std::shared_ptr<SamplingState> const& currentState) {
  assert(currentState->state.size()>blockInd);

  // the mean of the proposal is the current point
  std::vector<Eigen::VectorXd> props = currentState->state;
  assert(props.size()>blockInd);

  Eigen::VectorXd prop = proposal->Sample();
  props.at(blockInd) = prop;

  // store the new state in the output
  return std::make_shared<SamplingState>(props, 1.0);
}

double IndependenceProposal::LogDensity(std::shared_ptr<SamplingState> const& currState,
                              std::shared_ptr<SamplingState> const& propState) 
{   
   proposal->LogDensity(propState->state.at(blockInd));
   return proposal->LogDensity(propState->state.at(blockInd));
}
