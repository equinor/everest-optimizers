#include <gtest/gtest.h>

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/UniformBox.h"
#include "MUQ/Modeling/Distributions/Density.h"

#include "MUQ/SamplingAlgorithms/InferenceProblem.h"
#include "MUQ/SamplingAlgorithms/SamplingState.h"

using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;

TEST(InferenceProblem, Gaussian) {

  // create a Gaussian distribution---the sampling problem is built around characterizing this distribution
  Eigen::VectorXd mu(2);
  mu << 1,1;
  auto prior = std::make_shared<Gaussian>(mu)->AsDensity(); // it is standard normal (1D) by default
  auto likely = std::make_shared<Gaussian>(mu)->AsDensity(); // it is standard normal (1D) by default

  // create a sampling problem
  double temp = 0.5;
  auto problem = std::make_shared<InferenceProblem>(likely, prior, temp);

  Eigen::VectorXd xc = Eigen::VectorXd::Zero(2);
  auto state = std::make_shared<SamplingState>(xc, 1.0);

  EXPECT_DOUBLE_EQ(temp*likely->LogDensity(xc) + prior->LogDensity(xc), problem->LogDensity(state));

  Eigen::VectorXd trueGrad, probGrad;
  trueGrad = temp*likely->GradLogDensity(0,xc) + prior->GradLogDensity(0,xc);
  probGrad = problem->GradLogDensity(state,0);

  EXPECT_DOUBLE_EQ(trueGrad(0), probGrad(0));
  EXPECT_DOUBLE_EQ(trueGrad(1), probGrad(1));
  

}
