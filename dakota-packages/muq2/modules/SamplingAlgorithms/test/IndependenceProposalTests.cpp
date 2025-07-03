#include <gtest/gtest.h>

#include <boost/property_tree/ptree.hpp>

#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/IndependenceProposal.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/Gamma.h"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

TEST(MCMC, IndependenceProposal_FromOpts){
  
  const unsigned int N = 3e4;

  // parameters for the sampler
  pt::ptree pt;
  pt.put("NumSamples", N); // number of Monte Carlo samples
  pt.put("PrintLevel",0);
  pt.put("KernelList", "Kernel1"); // the transition kernel
  pt.put("Kernel1.Method","MHKernel");
  pt.put("Kernel1.Proposal", "MyProposal"); // the proposal
  pt.put("Kernel1.MyProposal.Method", "IndependenceProposal");
  pt.put("Kernel1.MyProposal.ProposalVariance", 0.75); // the variance of the isotropic MH proposal

  // create a Gaussian distribution---the sampling problem is built around characterizing this distribution
  const Eigen::VectorXd mu = Eigen::VectorXd::Zero(2);
  auto dist = std::make_shared<Gaussian>(mu)->AsDensity(); // standard normal Gaussian

  // create a sampling problem
  auto problem = std::make_shared<SamplingProblem>(dist);

  // starting point
  const Eigen::VectorXd start = mu;

  // create an instance of MCMC
  auto mcmc = std::make_shared<SingleChainMCMC>(pt,problem);

  // Make sure the kernel and proposal are correct
  std::shared_ptr<TransitionKernel> kernelBase = mcmc->Kernels().at(0);
  ASSERT_TRUE(kernelBase);
  std::shared_ptr<MHKernel> kernelMH = std::dynamic_pointer_cast<MHKernel>(kernelBase);
  ASSERT_TRUE(kernelMH);

  std::shared_ptr<MCMCProposal> proposalBase = kernelMH->Proposal();
  std::shared_ptr<IndependenceProposal> proposal = std::dynamic_pointer_cast<IndependenceProposal>(proposalBase);
  ASSERT_TRUE(proposal);

  std::shared_ptr<SampleCollection> samps = mcmc->Run(start);

  EXPECT_EQ(pt.get<int>("NumSamples"), samps->size());

  //boost::any anyMean = samps.Mean();
  Eigen::VectorXd mean = samps->Mean();
  Eigen::VectorXd ess = samps->ESS();
  Eigen::VectorXd var = samps->Variance();
  Eigen::VectorXd mcse = (var.array()/ess.array()).sqrt();
  EXPECT_NEAR(mu(0), mean(0), 3.0*mcse(0));
  EXPECT_NEAR(mu(1), mean(1), 3.0*mcse(1));

  Eigen::MatrixXd cov = samps->Covariance();
  EXPECT_NEAR(1.0, cov(0,0), 10.0*mcse(0));
  EXPECT_NEAR(0.0, cov(0,1), 10.0*mcse(0));
  EXPECT_NEAR(0.0, cov(1,0), 10.0*mcse(1));
  EXPECT_NEAR(1.0, cov(1,1), 10.0*mcse(1));
}


TEST(MCMC, IndependenceProposal_Manual){
  
  const unsigned int N = 3e4;

  // parameters for the sampler
  pt::ptree pt;
  pt.put("NumSamples", N); // number of Monte Carlo samples
  pt.put("PrintLevel",0);
  
  // create a Gaussian distribution---the sampling problem is built around characterizing this distribution
  const Eigen::VectorXd alpha = Eigen::VectorXd::Ones(2);
  const Eigen::VectorXd beta = Eigen::VectorXd::Ones(2);

  auto tgtDist = std::make_shared<Gamma>(alpha,beta)->AsDensity(); // standard normal Gaussian

  // create a sampling problem
  auto problem = std::make_shared<SamplingProblem>(tgtDist);

  // starting point
  const Eigen::VectorXd start = Eigen::VectorXd::Ones(2);

  // Manually construct the proposal, kernel, and chain
  auto prop = std::make_shared<IndependenceProposal>(pt,problem, std::make_shared<Gamma>(alpha,beta));
  auto kernel = std::make_shared<MHKernel>(pt, problem, prop);
  auto mcmc = std::make_shared<SingleChainMCMC>(pt, kernel);

  std::shared_ptr<SampleCollection> samps = mcmc->Run(start);

  EXPECT_EQ(pt.get<int>("NumSamples"), samps->size());

  //boost::any anyMean = samps.Mean();
  Eigen::VectorXd mean = samps->Mean();
  Eigen::VectorXd ess = samps->ESS();
  Eigen::VectorXd var = samps->Variance();
  Eigen::VectorXd mcse = (var.array()/ess.array()).sqrt();
  EXPECT_NEAR(alpha(0)/beta(0), mean(0), 3.0*mcse(0));
  EXPECT_NEAR(alpha(1)/beta(1), mean(1), 3.0*mcse(1));

  Eigen::MatrixXd cov = samps->Covariance();
  EXPECT_NEAR(alpha(0)/(beta(0)*beta(0)), cov(0,0), 5.0*mcse(0));
  EXPECT_NEAR(0.0, cov(0,1), 5.0*mcse(0));
  EXPECT_NEAR(0.0, cov(1,0), 5.0*mcse(1));
  EXPECT_NEAR(alpha(1)/(beta(1)*beta(1)), cov(1,1), 5.0*mcse(1));
}
