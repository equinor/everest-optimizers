#include "MUQ/SamplingAlgorithms/InverseGammaProposal.h"
#include "MUQ/SamplingAlgorithms/SamplingProblem.h"

#include "MUQ/Modeling/Distributions/InverseGamma.h"
#include "MUQ/Modeling/Distributions/Density.h"
#include "MUQ/Modeling/Distributions/Gaussian.h"

#include "MUQ/Modeling/ModGraphPiece.h"
#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/LinearAlgebra/IdentityOperator.h"

#include "MUQ/Utilities/Exceptions.h"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

REGISTER_MCMC_PROPOSAL(InverseGammaProposal)

InverseGammaProposal::InverseGammaProposal(pt::ptree                                pt,
                                           std::shared_ptr<AbstractSamplingProblem> prob) : MCMCProposal(pt,prob),
                                                                                            alpha(ExtractAlpha(prob,pt.get<std::string>("InverseGammaNode"))),
                                                                                            beta(ExtractBeta(prob,pt.get<std::string>("InverseGammaNode"))),
                                                                                            gaussInfo(ExtractGaussInfo(prob,pt.get<std::string>("GaussianNode"))),
                                                                                            varModel(ExtractVarianceModel(prob,pt.get<std::string>("GaussianNode"),pt.get<std::string>("InverseGammaNode"))),
                                                                                            gaussMean(ExtractMean(prob,pt.get<std::string>("GaussianNode")))
{
}


Eigen::VectorXd InverseGammaProposal::GetGaussianInput(std::shared_ptr<SamplingState> const& currentState) const
{
  // Extract the model inputs from the sampling state
  ref_vector<Eigen::VectorXd> inputs;
  for(auto& ind : std::get<1>(gaussInfo)){
    inputs.push_back(std::cref(currentState->state.at(ind)));
  }

  // Evaluate the model
  return std::get<0>(gaussInfo)->Evaluate(inputs).at(std::get<2>(gaussInfo));
}


std::shared_ptr<SamplingState> InverseGammaProposal::Sample(std::shared_ptr<SamplingState> const& currentState) {

  // the mean of the proposal is the current point
  std::vector<Eigen::VectorXd> props = currentState->state;
  Eigen::VectorXd gaussState = GetGaussianInput(currentState);

  Eigen::VectorXd varOnes = Eigen::VectorXd::Ones(varModel->outputSizes(0));
  Eigen::VectorXd varZeros = Eigen::VectorXd::Zero(varModel->inputSizes(0));

  Eigen::VectorXd newAlpha = alpha + 0.5*varModel->Gradient(0,0,varZeros,varOnes);
  Eigen::VectorXd newBeta = beta + 0.5*varModel->Gradient(0,0,varZeros, (gaussState - gaussMean).array().pow(2).matrix().eval());

  props.at(blockInd) = InverseGamma(newAlpha, newBeta).Sample();

  // store the new state in the output
  return std::make_shared<SamplingState>(props, 1.0);
}

double InverseGammaProposal::LogDensity(std::shared_ptr<SamplingState> const& currState,
                                        std::shared_ptr<SamplingState> const& propState) {
  Eigen::VectorXd gaussState = GetGaussianInput(currState);
  Eigen::VectorXd const& sigmaState = propState->state.at(blockInd);

  Eigen::VectorXd varOnes = Eigen::VectorXd::Ones(varModel->outputSizes(0));
  Eigen::VectorXd varZeros = Eigen::VectorXd::Zero(varModel->inputSizes(0));

  Eigen::VectorXd newAlpha = alpha + 0.5*varModel->Gradient(0,0,varZeros,varOnes);

  Eigen::VectorXd newBeta = beta + 0.5*varModel->Gradient(0,0,varZeros, (gaussState - gaussMean).array().pow(2).matrix().eval());

  return InverseGamma(newAlpha, newBeta).LogDensity(sigmaState);
}

Eigen::VectorXd InverseGammaProposal::ExtractMean(std::shared_ptr<AbstractSamplingProblem> prob,
                                                  std::string                        const& gaussNode)
{
  // Cast the abstract base class into a sampling problem
  std::shared_ptr<SamplingProblem> prob2 = std::dynamic_pointer_cast<SamplingProblem>(prob);
  if(prob2==nullptr){
    throw std::runtime_error("Could not downcast AbstractSamplingProblem to SamplingProblem.");
  }

  // From the sampling problem, extract the ModPiece and try to cast it to a ModGraphPiece
  std::shared_ptr<ModPiece> targetDens = prob2->GetDistribution();
  std::shared_ptr<ModGraphPiece> targetDens2 = std::dynamic_pointer_cast<ModGraphPiece>(targetDens);
  if(targetDens2==nullptr){
    throw std::runtime_error("Could not downcast target density to ModGraphPiece.");
  }

  // Get the graph
  auto graph = targetDens2->GetGraph();

  // Get the Gaussian node
  auto gaussPiece = graph->GetPiece(gaussNode);
  if(gaussPiece==nullptr){
    throw std::runtime_error("Could not find " + gaussNode + " node.");
  }

  // try to cast the gaussPiece to a density
  auto dens = std::dynamic_pointer_cast<Density>(gaussPiece);
  if(dens==nullptr){
    throw std::runtime_error("Could not convert specified Gausssian ModPiece to Density.");
  }

  auto gaussDist = std::dynamic_pointer_cast<Gaussian>(dens->GetDistribution());
  if(gaussDist==nullptr){
    throw std::runtime_error("Could not cast Gausssian ModPiece to Gaussian.");
  }

  return gaussDist->GetMean();
}

std::shared_ptr<InverseGamma> InverseGammaProposal::ExtractInverseGamma(std::shared_ptr<AbstractSamplingProblem> prob, std::string const& igNode)
{
  // Cast the abstract base class into a sampling problem
  std::shared_ptr<SamplingProblem> prob2 = std::dynamic_pointer_cast<SamplingProblem>(prob);
  if(prob2==nullptr){
    throw std::runtime_error("Could not downcast AbstractSamplingProblem to SamplingProblem.");
  }

  // From the sampling problem, extract the ModPiece and try to cast it to a ModGraphPiece
  std::shared_ptr<ModPiece> targetDens = prob2->GetDistribution();
  std::shared_ptr<ModGraphPiece> targetDens2 = std::dynamic_pointer_cast<ModGraphPiece>(targetDens);
  if(targetDens2==nullptr){
    throw std::runtime_error("Could not downcast target density to ModGraphPiece.");
  }

  // Get the graph
  auto graph = targetDens2->GetGraph();

  // Get the Gaussian node
  auto piece = graph->GetPiece(igNode);
  if(piece==nullptr){
    throw std::runtime_error("Could not find " + igNode + " node.");
  }

  // try to cast the gaussPiece to a density
  auto dens = std::dynamic_pointer_cast<Density>(piece);
  if(dens==nullptr){
    throw std::runtime_error("Could not convert specified InverseGamma ModPiece to Density.");
  }

  auto dist = std::dynamic_pointer_cast<InverseGamma>(dens->GetDistribution());
  if(dist==nullptr){
    throw std::runtime_error("Could not convert specified InverseGamma ModPiece to InverseGamma distribution.");
  }

  return dist;
}

Eigen::VectorXd InverseGammaProposal::ExtractAlpha(std::shared_ptr<AbstractSamplingProblem> prob, std::string const& igNode)
{
  return ExtractInverseGamma(prob,igNode)->alpha;
}

Eigen::VectorXd InverseGammaProposal::ExtractBeta(std::shared_ptr<AbstractSamplingProblem> prob, std::string const& igNode)
{
  return ExtractInverseGamma(prob,igNode)->beta;
}

std::tuple<std::shared_ptr<muq::Modeling::ModPiece>, std::vector<int>, int> InverseGammaProposal::ExtractGaussInfo(std::shared_ptr<AbstractSamplingProblem> prob, std::string const& gaussNode)
{

  // Cast the abstract base class into a sampling problem
  std::shared_ptr<SamplingProblem> prob2 = std::dynamic_pointer_cast<SamplingProblem>(prob);
  if(prob2==nullptr){
    throw std::runtime_error("Could not downcast AbstractSamplingProblem to SamplingProblem.");
  }

  // From the sampling problem, extract the ModPiece and try to cast it to a ModGraphPiece
  std::shared_ptr<ModPiece> targetDens = prob2->GetDistribution();
  std::shared_ptr<ModGraphPiece> targetDens2 = std::dynamic_pointer_cast<ModGraphPiece>(targetDens);
  if(targetDens2==nullptr){
    throw std::runtime_error("Could not downcast target density to ModGraphPiece.");
  }

  // Get the graph
  auto graph = targetDens2->GetGraph();

  // Get a pointer to the input to the Gaussian piece
  std::string inputName = graph->GetParent(gaussNode,0);
  
  std::shared_ptr<ModPiece> gaussInputPiece;
  std::vector<int> inputInds;
  if(inputName==gaussNode+"_0"){
    auto piece = std::dynamic_pointer_cast<ModPiece>(graph->GetPiece(gaussNode));
    assert(piece!=nullptr);

    gaussInputPiece = std::make_shared<IdentityOperator>(piece->inputSizes(0));
    inputInds = std::vector<int>{0};
  }else{
    auto graphPiece = targetDens2->GetSubModel(inputName);
    inputInds = targetDens2->MatchInputs(graphPiece);
    gaussInputPiece = std::dynamic_pointer_cast<ModPiece>(graphPiece);
  }


  // Make sure we were able to match all the inputs
  for(auto& ind : inputInds){
    if(ind<0){
      throw std::runtime_error("Something went wrong in constructing Gaussian input model and not all inputs could be matched to the original graph.");
    }
  }

  // Figure out which output of the parent node is passed to the Gaussian
  int gaussInd;
  for(auto edge : graph->GetEdges(inputName, gaussNode)){
    if(edge.second==0){
      gaussInd=edge.first;
      break;
    }
  }

  return std::make_tuple(gaussInputPiece, inputInds, gaussInd);
}

std::shared_ptr<muq::Modeling::ModPiece> InverseGammaProposal::ExtractVarianceModel(std::shared_ptr<AbstractSamplingProblem> prob, std::string const& gaussNode, std::string const& igNode)
{
  std::shared_ptr<SamplingProblem> prob2 = std::dynamic_pointer_cast<SamplingProblem>(prob);
  if(prob2==nullptr){
    throw std::runtime_error("Could not downcast AbstractSamplingProblem to SamplingProblem.");
  }

  // From the sampling problem, extract the ModPiece and try to cast it to a ModGraphPiece
  std::shared_ptr<ModPiece> targetDens = prob2->GetDistribution();
  std::shared_ptr<ModGraphPiece> targetDens2 = std::dynamic_pointer_cast<ModGraphPiece>(targetDens);
  if(targetDens2==nullptr){
    throw std::runtime_error("Could not downcast target density to ModGraphPiece.");
  }

  // Get the graph
  auto graph = targetDens2->GetGraph();

  // Get a pointer to the variance input to the Gaussian piece
  std::string inputName = graph->GetParent(gaussNode,1);
  auto varPiece = targetDens2->GetSubModel(inputName);

  // Make sure there is only one input
  if(varPiece->inputSizes.size()!=1){
    throw std::runtime_error("The Gaussian variance can only depend on one input parameter.");
  }

  if(varPiece->outputSizes.size()!=1){
    throw std::runtime_error("The Gaussian variance can only have one output.");
  }

  // Make sure the input to the variance piece matches the input to the hyper prior
  auto hyperPrior = targetDens2->GetSubModel(igNode);
  std::vector<int> sharedInds = varPiece->MatchInputs(hyperPrior);

  if((sharedInds.size()!=1)||(sharedInds.at(0)<0)){
    throw std::runtime_error("Something is strange with the WorkGraph.  Could not match the input of the hyperprior with a path to the variance of the Gaussian node.");
  }

  return varPiece;
}
