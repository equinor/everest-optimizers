#include "MUQ/SamplingAlgorithms/SLMCMC.h"
#include "MUQ/SamplingAlgorithms/GreedyMLMCMC.h"
#include "MUQ/SamplingAlgorithms/MIMCMC.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/Density.h"

#include "MUQ/SamplingAlgorithms/MHKernel.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/CrankNicolsonProposal.h"
#include "MUQ/SamplingAlgorithms/SamplingProblem.h"
#include "MUQ/SamplingAlgorithms/SubsamplingMIProposal.h"
#include "MUQ/SamplingAlgorithms/MultiIndexEstimator.h"

#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"

#include <boost/property_tree/ptree.hpp>

#include <gtest/gtest.h>

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;


class MySamplingProblem : public AbstractSamplingProblem {
public:
  MySamplingProblem(std::shared_ptr<muq::Modeling::ModPiece> targetIn)
   : AbstractSamplingProblem(Eigen::VectorXi::Constant(1,2), Eigen::VectorXi::Constant(1,2)),
     target(targetIn){}

  virtual ~MySamplingProblem() = default;


  virtual double LogDensity(std::shared_ptr<SamplingState> const& state) override {
    lastState = state;
    return target->Evaluate(state->state).at(0)(0);
  };

  virtual std::shared_ptr<SamplingState> QOI() override {
    assert (lastState != nullptr);
    return std::make_shared<SamplingState>(lastState->state, 1.0);
  }

private:
  std::shared_ptr<SamplingState> lastState = nullptr;

  std::shared_ptr<muq::Modeling::ModPiece> target;

};


class MyMLInterpolation : public MIInterpolation {
public:
  std::shared_ptr<SamplingState> Interpolate (std::shared_ptr<SamplingState> const& coarseProposal, std::shared_ptr<SamplingState> const& fineProposal) {
    return std::make_shared<SamplingState>(coarseProposal->state);
  }
};

class MyMLComponentFactory : public MIComponentFactory {
public:
  MyMLComponentFactory(pt::ptree pt)
   : pt(pt) {}

  virtual std::shared_ptr<MCMCProposal> Proposal (std::shared_ptr<MultiIndex> const& index, std::shared_ptr<AbstractSamplingProblem> const& samplingProblem) override {
    pt::ptree pt;
    pt.put("BlockIndex",0);

    Eigen::VectorXd mu(2);
    mu << 1.0, 2.0;
    Eigen::MatrixXd cov(2,2);
    cov << 0.7, 0.6,
    0.6, 1.0;
    cov *= 20.0;

    auto prior = std::make_shared<Gaussian>(mu, cov);

    return std::make_shared<CrankNicolsonProposal>(pt, samplingProblem, prior);
  }

  virtual std::shared_ptr<MultiIndex> FinestIndex() override {
    auto index = std::make_shared<MultiIndex>(1);
    index->SetValue(0, 3);
    return index;
  }

  virtual std::shared_ptr<MCMCProposal> CoarseProposal (std::shared_ptr<MultiIndex> const& fineIndex,
                                                        std::shared_ptr<MultiIndex> const& coarseIndex,
                                                        std::shared_ptr<AbstractSamplingProblem> const& coarseProblem,
                                                        std::shared_ptr<SingleChainMCMC> const& coarseChain) override {
    pt::ptree ptProposal = pt;
    ptProposal.put("BlockIndex",0);
    return std::make_shared<SubsamplingMIProposal>(ptProposal, coarseProblem, coarseIndex, coarseChain);
  }

  virtual std::shared_ptr<AbstractSamplingProblem> SamplingProblem (std::shared_ptr<MultiIndex> const& index) override {
    Eigen::VectorXd mu(2);
    mu << 1.0, 2.0;
    Eigen::MatrixXd cov(2,2);
    cov << 0.7, 0.6,
           0.6, 1.0;

    if (index->GetValue(0) == 0) {
      mu *= 0.8;
      cov *= 2.0;
    } else if (index->GetValue(0) == 1) {
      mu *= 0.9;
      cov *= 1.5;
    } else if (index->GetValue(0) == 2) {
      mu *= 0.99;
      cov *= 1.1;
    } else if (index->GetValue(0) == 3) {
      mu *= 1.0;
      cov *= 1.0;
    } else {
      std::cerr << "Sampling problem not defined!" << std::endl;
      assert (false);
    }

    auto coarseTargetDensity = std::make_shared<Gaussian>(mu, cov)->AsDensity();
    return std::make_shared<MySamplingProblem>(coarseTargetDensity);
  }

  virtual std::shared_ptr<MIInterpolation> Interpolation (std::shared_ptr<MultiIndex> const& index) override {
    return std::make_shared<MyMLInterpolation>();
  }

  virtual Eigen::VectorXd StartingPoint (std::shared_ptr<MultiIndex> const& index) override {
    Eigen::VectorXd mu(2);
    mu << 1.0, 2.0;
    return mu;
  }
  pt::ptree pt;
};

TEST(MLMCMCTest, GreedyMLMCMC)
{

  pt::ptree pt;

  pt.put("NumSamples", 1e4); // number of samples for single level
  pt.put("NumInitialSamples", 1e3); // number of initial samples for greedy MLMCMC
  pt.put("GreedyTargetVariance", 0.05); // estimator variance to be achieved by greedy algorithm
  pt.put("MLMCMC.Subsampling_0", 5); // estimator variance to be achieved by greedy algorithm
  pt.put("MLMCMC.Subsampling_1", 3); // estimator variance to be achieved by greedy algorithm
  pt.put("MLMCMC.Subsampling_2", 1); // estimator variance to be achieved by greedy algorithm
  pt.put("MLMCMC.Subsampling_3", 0); // estimator variance to be achieved by greedy algorithm

  auto componentFactory = std::make_shared<MyMLComponentFactory>(pt);

  GreedyMLMCMC greedymlmcmc (pt, componentFactory);
  greedymlmcmc.Run();
  greedymlmcmc.Draw(false);

  Eigen::VectorXd trueMu(2);
  trueMu << 1.0, 2.0;
  Eigen::MatrixXd trueCov(2,2);
  trueCov << 0.7, 0.6,
             0.6, 1.0;

  auto params = greedymlmcmc.GetSamples();
  Eigen::VectorXd mean = params->Mean();
  Eigen::VectorXd mcse = params->StandardError();
  EXPECT_NEAR(trueMu(0), mean(0), 3.*mcse(0));
  EXPECT_NEAR(trueMu(1), mean(1), 3.0*mcse(1));

  Eigen::VectorXd variance = params->Variance();
  EXPECT_NEAR(trueCov(0,0), variance(0), 5.0*mcse(0));
  EXPECT_NEAR(trueCov(1,1), variance(1), 5.0*mcse(1));

  Eigen::VectorXd skewness = params->Skewness();
  EXPECT_NEAR(0.0, skewness(0), 0.5);
  EXPECT_NEAR(0.0, skewness(0), 0.5);

  Eigen::MatrixXd covariance = params->Covariance();
  EXPECT_NEAR(trueCov(0,0), covariance(0,0), 0.2);
  EXPECT_NEAR(trueCov(1,1), covariance(1,1), 0.2);
  EXPECT_NEAR(trueCov(0,1), covariance(0,1), 0.2);
  EXPECT_NEAR(trueCov(1,0), covariance(1,0), 0.2);


  auto qois = greedymlmcmc.GetQOIs();
  mean = qois->Mean();
  EXPECT_NEAR(trueMu(0), mean(0), 0.3);
  EXPECT_NEAR(trueMu(1), mean(1), 0.3);

  variance = qois->Variance();
  EXPECT_NEAR(trueCov(0,0), variance(0), 0.3);
  EXPECT_NEAR(trueCov(1,1), variance(1), 0.3);

  skewness = qois->Skewness();
  EXPECT_NEAR(0.0, skewness(0), 0.5);
  EXPECT_NEAR(0.0, skewness(0), 0.5);
}
