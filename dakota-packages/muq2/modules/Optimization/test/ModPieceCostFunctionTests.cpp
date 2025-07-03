#include <gtest/gtest.h>

#include "RosenbrockFunction.h"

#include "MUQ/Optimization/ModPieceCostFunction.h"

using namespace muq::Modeling;
using namespace muq::Optimization;

TEST(ModPieceCostFunctionTests, RosenbrockCost) {
  double a = 100.0;

  // the Rosenbrock cost function
  std::shared_ptr<ModPiece> rosen = std::make_shared<RosenbrockModPiece>(a);
  std::shared_ptr<CostFunction> cost = std::make_shared<ModPieceCostFunction>(rosen);

  // choose a random point
  const Eigen::VectorXd x = Eigen::Vector2d::Random();

  // the true value
  const double cst = (1.0-x(0))*(1.0-x(0))+100.0*(x(1)-x(0)*x(0))*(x(1)-x(0)*x(0));

  // check the cost evaluations
  EXPECT_DOUBLE_EQ(cst, cost->Evaluate(x).at(0)(0));
  EXPECT_DOUBLE_EQ(cst, cost->Cost(x));

  // the true gradient
  const Eigen::Vector2d grad_true(-400.0*(x(1)-x(0)*x(0))*x(0)-2.0*(1.0-x(0)), 200.0*(x(1)-x(0)*x(0)));

  // compute the gradient
  const Eigen::VectorXd& grad_test0 = cost->Gradient(x);

  EXPECT_DOUBLE_EQ((grad_true-grad_test0).norm(), 0.0);
}
