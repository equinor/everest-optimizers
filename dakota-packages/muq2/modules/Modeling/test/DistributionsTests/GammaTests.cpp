#include <gtest/gtest.h>

#include "MUQ/Modeling/Distributions/Gamma.h"

using namespace muq::Modeling;


TEST(GammaDistributionTests, EvaluateLogDensity) {

  const double alpha = 1.5;
  const double beta = 0.5;

  auto dist = std::make_shared<Gamma>(1.5, 0.5);

  double logScale = alpha * std::log(beta) - std::lgamma(alpha);

  // evalute the log-density
  Eigen::VectorXd x(1);
  x << 0.5;

  double logdens = dist->LogDensity(x);
  EXPECT_DOUBLE_EQ(logScale + (alpha-1.0)*std::log(x(0)) - beta*x(0), logdens);

  x << 1.5;
  logdens = dist->LogDensity(x);
  EXPECT_DOUBLE_EQ(logScale + (alpha-1.0)*std::log(x(0)) - beta*x(0), logdens);

  x << -1.0;
  logdens = dist->LogDensity(x);
  EXPECT_DOUBLE_EQ(-1.0*std::numeric_limits<double>::infinity(), logdens);
}

TEST(GammaDistributionTests, EvaluateLogDensity_Multivariate) {

  Eigen::VectorXd alpha(2);
  alpha << 1.5, 2.0;
  Eigen::VectorXd beta(2);
  beta << 0.5, 0.5;

  auto dist = std::make_shared<Gamma>(alpha, beta);

  double logScale = alpha(0) * std::log(beta(0)) - std::lgamma(alpha(0));
  logScale += alpha(1)*std::log(beta(1)) - std::lgamma(alpha(1));

  // evalute the log-denstion
  Eigen::VectorXd x(2);
  x << 0.5, 0.6;

  double logdens = dist->LogDensity(x);
  EXPECT_DOUBLE_EQ(logScale + (alpha(0)-1.0)*std::log(x(0)) - beta(0)*x(0)+ (alpha(1)-1.0)*std::log(x(1)) - beta(1)*x(1), logdens);
}

TEST(GammaDistributionTests, Sample) {

  const double alpha = 2.5;
  const double beta = 1.0;

  auto dist = std::make_shared<Gamma>(alpha, beta);

  unsigned int numSamps  = 1e6;
  double mu = 0.0;
  for(int i=0; i<numSamps; ++i){
    double samp  = dist->Sample()(0);
    EXPECT_TRUE(samp>0);
    mu += (1.0/double(numSamps))*samp;
  }

  EXPECT_NEAR(alpha/beta, mu, 5e-3);
}


TEST(GammaDistributionTests, Sample_Multivariate) {

  Eigen::VectorXd alpha(2);
  alpha << 1.5, 2.0;
  Eigen::VectorXd beta(2);
  beta << 0.5, 0.5;

  auto dist = std::make_shared<Gamma>(alpha, beta);

  unsigned int numSamps  = 1e6;
  Eigen::VectorXd mu = Eigen::VectorXd::Zero(2);
  for(int i=0; i<numSamps; ++i){
    Eigen::VectorXd samp  = dist->Sample();
    EXPECT_TRUE(samp(0)>0);
    EXPECT_TRUE(samp(1)>0);
    mu += (1.0/double(numSamps))*samp;
  }

  EXPECT_NEAR(alpha(0)/beta(0), mu(0), 5e-2);
  EXPECT_NEAR(alpha(1)/beta(1), mu(1), 5e-2);
}

TEST(GammaDistributionTests, FromMoments) {

  Eigen::VectorXd alpha(2);
  alpha << 1.5, 2.0;
  Eigen::VectorXd beta(2);
  beta << 0.5, 0.5;

  Eigen::VectorXd mu = alpha.array() / beta.array();
  Eigen::VectorXd var = alpha.array() / beta.array().square();

  auto dist = Gamma::FromMoments(mu,var);

  EXPECT_DOUBLE_EQ(alpha(0), dist->alpha(0));
  EXPECT_DOUBLE_EQ(alpha(1), dist->alpha(1));
  EXPECT_DOUBLE_EQ(beta(0), dist->beta(0));
  EXPECT_DOUBLE_EQ(beta(1), dist->beta(1));
}
