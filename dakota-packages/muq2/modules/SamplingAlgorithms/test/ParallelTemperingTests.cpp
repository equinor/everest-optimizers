#include <gtest/gtest.h>

#include <boost/property_tree/ptree.hpp>

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/Density.h"

#include "MUQ/SamplingAlgorithms/ParallelTempering.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/IndependenceProposal.h"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;




TEST(MCMC, ParallelTempering_FromOpts) {

    // parameters for the sampler
    pt::ptree pt;
    pt.put("NumSamples", 1e4); // number of Monte Carlo samples
    pt.put("PrintLevel",0);
    pt.put("Inverse Temperatures","0.0,0.25,0.5,0.75,1.0");
    pt.put("Swap Increment", 100);
    pt.put("Swap Type", "SEO");
    pt.put("Adapt Start", 100); // Start adapting after this many steps
    pt.put("Kernel Lists", "Kernel1;Kernel1;Kernel1;Kernel1;Kernel1"); // the transition kernel for the first dimension
    pt.put("Kernel1.Method","MHKernel");
    pt.put("Kernel1.Proposal", "MyProposal"); // the proposal
    pt.put("Kernel1.MyProposal.Method", "MHProposal");
    pt.put("Kernel1.MyProposal.ProposalVariance", 0.5); // the variance of the isotropic MH proposal

    // create a Gaussian distribution---the sampling problem is built around characterizing this distribution
    Eigen::VectorXd priorMu = Eigen::VectorXd::Ones(2);
    Eigen::MatrixXd priorCov(2,2);
    priorCov << 2.0, 1.0,
                1.0, 1.5;
    auto prior = std::make_shared<Gaussian>(priorMu, priorCov)->AsDensity();

    Eigen::VectorXd obsVal(2);
    obsVal << 0.5, 0.6;
    double obsVar = 1;
    auto likely = std::make_shared<Gaussian>(obsVal, obsVar*Eigen::VectorXd::Ones(2))->AsDensity();

    // Compute the analytic solution 
    auto priorDist = std::make_shared<Gaussian>(priorMu, priorCov);
    auto post = priorDist->Condition(Eigen::MatrixXd::Identity(2,2), obsVal, obsVar*Eigen::MatrixXd::Identity(2,2));
    Eigen::VectorXd postMu = post->GetMean();
    Eigen::MatrixXd postCov = post->GetCovariance();


    // create a sampling problem
    auto problem = std::make_shared<InferenceProblem>(likely, prior);

    ParallelTempering sampler(pt, problem);

    EXPECT_DOUBLE_EQ(0.0, sampler.GetInverseTemp(0));
    EXPECT_DOUBLE_EQ(1.0, sampler.GetInverseTemp(4));

    auto samps = sampler.Run(post->Sample());

    Eigen::VectorXd q = samps->Mean();
    Eigen::VectorXd mcse = samps->StandardError();
    EXPECT_NEAR(postMu(0), q(0), 4.0*mcse(0));
    EXPECT_NEAR(postMu(1), q(1), 4.0*mcse(1));
}

TEST(MCMC, ParallelTempering_Manual) {

    // parameters for the sampler
    pt::ptree pt;
    pt.put("NumSamples", 1e4); // number of Monte Carlo samples
    pt.put("BurnIn", 0);
    pt.put("PrintLevel",0);
    pt.put("Swap Increment", 100);
    pt.put("Swap Type", "DEO");
    pt.put("Adapt Start", 100); // Start adapting after this many steps

    // create a Gaussian distribution---the sampling problem is built around characterizing this distribution
    Eigen::VectorXd priorMu = Eigen::VectorXd::Ones(2);
    Eigen::MatrixXd priorCov(2,2);
    priorCov << 2.0, 1.0,
                1.0, 1.5;
    auto prior = std::make_shared<Gaussian>(priorMu, priorCov)->AsDensity();

    Eigen::VectorXd obsVal(2);
    obsVal << 0.5, 0.6;
    double obsVar = 1;
    auto likely = std::make_shared<Gaussian>(obsVal, obsVar*Eigen::VectorXd::Ones(2))->AsDensity();

    // Compute the analytic solution 
    auto priorDist = std::make_shared<Gaussian>(priorMu, priorCov);
    auto post = priorDist->Condition(Eigen::MatrixXd::Identity(2,2), obsVal, obsVar*Eigen::MatrixXd::Identity(2,2));
    Eigen::VectorXd postMu = post->GetMean();
    Eigen::MatrixXd postCov = post->GetCovariance();


    // create a sampling problem
    auto problem = std::make_shared<InferenceProblem>(likely, prior);

    // Construct the proposals and kernels for each temperature
    unsigned int numTemps = 5;
    Eigen::VectorXd invTemps(numTemps);
    invTemps << 0.0, 0.25, 0.5, 0.75, 1.0;
    
    std::vector<std::shared_ptr<TransitionKernel>> kernels(numTemps);

    // Independence proposal for zero temperature
    std::shared_ptr<MCMCProposal> prop = std::make_shared<IndependenceProposal>(pt, problem, std::make_shared<Gaussian>(priorMu, priorCov));
    kernels.at(0) = std::make_shared<MHKernel>(pt, problem, prop);

    for(unsigned int i=1; i<numTemps; ++i){
        pt.put("ProposalVariance", 2.0);
        auto probCopy = problem->Clone();
        prop = std::make_shared<MHProposal>(pt, probCopy);
        kernels.at(i) = std::make_shared<MHKernel>(pt, probCopy, prop);
    }

    ParallelTempering sampler(pt, invTemps, kernels);

    auto samps = sampler.Run(post->Sample());

    EXPECT_EQ(pt.get<int>("NumSamples")-pt.get<int>("BurnIn"), samps->size());

    Eigen::VectorXd q = samps->Mean();
    Eigen::VectorXd mcse = samps->StandardError();
    std::cout << "MCSE: " << mcse.transpose() << std::endl;
    EXPECT_NEAR(postMu(0), q(0), 4.0*mcse(0));
    EXPECT_NEAR(postMu(1), q(1), 4.0*mcse(1));

    q = samps->Variance();
    EXPECT_NEAR(postCov(0,0), q(0), 1.0);
    EXPECT_NEAR(postCov(1,1), q(1), 1.0);

}

