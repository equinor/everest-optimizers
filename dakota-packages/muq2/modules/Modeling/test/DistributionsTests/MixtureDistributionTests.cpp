#include <gtest/gtest.h>

#include <Eigen/Core>

#include "MUQ/Utilities/RandomGenerator.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/MixtureDistribution.h"

using namespace muq::Utilities;
using namespace muq::Modeling;

TEST(MixtureDistributionTests, GaussianMixture) {

    Eigen::VectorXd mu1(2);
    mu1 << 1.0, 1.0;
    Eigen::MatrixXd cov1(2,2);
    cov1 << 1.0, -0.5,
            -0.5, 1.0;

    Eigen::VectorXd mu2(2);
    mu2 << -1.0, -1.0;
    Eigen::MatrixXd cov2(2,2);
    cov2 << 1.0, 0.5,
            0.5, 1.0;

    std::vector<std::shared_ptr<Distribution>> comps(2);
    comps.at(0) = std::make_shared<Gaussian>(mu1,cov1);
    comps.at(1) = std::make_shared<Gaussian>(mu2,cov2);

    Eigen::VectorXd probs(2);
    probs << 0.25, 0.75;

    std::shared_ptr<MixtureDistribution> dist = std::make_shared<MixtureDistribution>(comps, probs);

    Eigen::VectorXd x(2);
    x << 0.0, 0.0;

    double trueDens = std::log( probs(0) * std::exp(comps.at(0)->LogDensity(x)) + probs(1)*std::exp(comps.at(1)->LogDensity(x)));
    double dens = dist->LogDensity(x);
    EXPECT_NEAR(trueDens, dens, 1e-15);
    
    dist->Sample(x);
    dist->GradLogDensity(0,x);
}
