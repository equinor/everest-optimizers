#include <gtest/gtest.h>

#include <boost/property_tree/ptree.hpp>

#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/WorkGraphPiece.h"
#include "MUQ/Modeling/ModGraphPiece.h"
#include "MUQ/Modeling/LinearAlgebra/IdentityOperator.h"
#include "MUQ/Modeling/ReplicateOperator.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/InverseGamma.h"
#include "MUQ/Modeling/Distributions/DensityProduct.h"

#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/InverseGammaProposal.h"

#include "MUQ/Utilities/AnyHelpers.h"
#include "MUQ/Utilities/RandomGenerator.h"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

TEST(MCMC, InverseGammaProposal_DirectInput) {

  WorkGraph graph;

  graph.AddNode(std::make_shared<IdentityOperator>(1), "Variance");

  Eigen::VectorXd mean(2);
  mean << 1,2;

  auto gaussDist = std::make_shared<Gaussian>(mean, Gaussian::DiagCovariance);
  graph.AddNode(gaussDist->AsDensity(), "Gaussian Density");

  double alpha = 2.5;
  double beta = 1.0;
  auto varDist = std::make_shared<InverseGamma>(alpha,beta);
  graph.AddNode(varDist->AsDensity(), "Variance Density");

  graph.AddNode(std::make_shared<DensityProduct>(2), "Joint Density");
  graph.AddNode(std::make_shared<ReplicateOperator>(1,2), "Replicated Variance");

  graph.AddEdge("Variance", 0, "Replicated Variance", 0);
  graph.AddEdge("Replicated Variance", 0, "Gaussian Density", 1);
  graph.AddEdge("Variance", 0, "Variance Density", 0);

  graph.AddEdge("Gaussian Density", 0, "Joint Density", 0);
  graph.AddEdge("Variance Density", 0, "Joint Density", 1);

  auto jointDens = graph.CreateModPiece("Joint Density");
  auto problem = std::make_shared<SamplingProblem>(jointDens);



  boost::property_tree::ptree opts;
  opts.put("InverseGammaNode", "Variance Density");
  opts.put("GaussianNode", "Gaussian Density");
  opts.put("NumSamples", 10000);
  opts.put("BurnIn", 1000);
  opts.put("ProposalVariance", 0.5);
  opts.put("PrintLevel",0);

  // Define the transition kernel for the parameter block
  auto paramProp = std::make_shared<MHProposal>(opts, problem);
  auto paramKernel = std::make_shared<MHKernel>(opts, problem, paramProp);
  paramKernel->SetBlockInd(0);

  // Define the transition kernel for the variance block
  auto varProp = std::make_shared<InverseGammaProposal>(opts, problem);
  auto varKernel = std::make_shared<MHKernel>(opts, problem, varProp);
  varKernel->SetBlockInd(1);

  std::vector<std::shared_ptr<TransitionKernel>> kernels;
  kernels.push_back(paramKernel);
  kernels.push_back(varKernel);

  auto mcmc = std::make_shared<SingleChainMCMC>(opts, kernels);

  Eigen::VectorXd initialVar = (beta/(alpha-1))*Eigen::VectorXd::Ones(1);
  std::vector<Eigen::VectorXd> startPt{mean, initialVar};
  auto samps = mcmc->Run(startPt);

  Eigen::VectorXd postMean = samps->Mean();

  Eigen::VectorXd ess = samps->ESS();
  Eigen::VectorXd postVar = samps->Variance();

  Eigen::VectorXd mcStd = (postVar.array()/ess.array()).sqrt();// Monte Carlo standard error
  EXPECT_NEAR(mean(0), postMean(0), 4.0*mcStd(0));
  EXPECT_NEAR(mean(1), postMean(1), 4.0*mcStd(1));
}



TEST(MCMC, InverseGammaProposal_ModelInput) {

  WorkGraph graph;

  graph.AddNode(std::make_shared<IdentityOperator>(1), "Variance");
  graph.AddNode(std::make_shared<IdentityOperator>(2), "Parameters");

  Eigen::VectorXd mean(2);
  mean << 1,2;

  auto gaussDist = std::make_shared<Gaussian>(mean, Gaussian::DiagCovariance);
  graph.AddNode(gaussDist->AsDensity(), "Gaussian Density");

  double alpha = 2.5;
  double beta = 1.0;
  auto varDist = std::make_shared<InverseGamma>(alpha,beta);
  graph.AddNode(varDist->AsDensity(), "Variance Density");

  graph.AddNode(std::make_shared<DensityProduct>(2), "Joint Density");
  graph.AddNode(std::make_shared<ReplicateOperator>(1,2), "Replicated Variance");

  graph.AddEdge("Parameters", 0, "Gaussian Density", 0);
  graph.AddEdge("Variance", 0, "Replicated Variance", 0);
  graph.AddEdge("Replicated Variance", 0, "Gaussian Density", 1);
  graph.AddEdge("Variance", 0, "Variance Density", 0);

  graph.AddEdge("Gaussian Density", 0, "Joint Density", 0);
  graph.AddEdge("Variance Density", 0, "Joint Density", 1);

  auto jointDens = graph.CreateModPiece("Joint Density");
  auto problem = std::make_shared<SamplingProblem>(jointDens);



  boost::property_tree::ptree opts;
  opts.put("InverseGammaNode", "Variance Density");
  opts.put("GaussianNode", "Gaussian Density");
  opts.put("NumSamples", 30000);
  opts.put("BurnIn", 1000);
  opts.put("ProposalVariance", 0.5);
  opts.put("PrintLevel",0);

  // Define the transition kernel for the parameter block
  auto paramProp = std::make_shared<MHProposal>(opts, problem);
  auto paramKernel = std::make_shared<MHKernel>(opts, problem, paramProp);
  paramKernel->SetBlockInd(0);

  // Define the transition kernel for the variance block
  auto varProp = std::make_shared<InverseGammaProposal>(opts, problem);
  auto varKernel = std::make_shared<MHKernel>(opts, problem, varProp);
  varKernel->SetBlockInd(1);

  std::vector<std::shared_ptr<TransitionKernel>> kernels;
  kernels.push_back(paramKernel);
  kernels.push_back(varKernel);

  auto mcmc = std::make_shared<SingleChainMCMC>(opts, kernels);

  Eigen::VectorXd initialVar = (beta/(alpha-1))*Eigen::VectorXd::Ones(1);
  std::vector<Eigen::VectorXd> startPt{mean, initialVar};
  auto samps = mcmc->Run(startPt);

  Eigen::VectorXd postMean = samps->Mean();

  Eigen::VectorXd ess = samps->ESS();
  Eigen::VectorXd postVar = samps->Variance();

  Eigen::VectorXd mcStd = samps->StandardError();// Monte Carlo standard error
  EXPECT_NEAR(mean(0), postMean(0), 5.0*mcStd(0));
  EXPECT_NEAR(mean(1), postMean(1), 5.0*mcStd(1));
  EXPECT_NEAR(beta/(alpha-1), postMean(2), 5.0*mcStd(2));
}
