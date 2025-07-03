#include <gtest/gtest.h>

#include "RosenbrockFunction.h"

using namespace muq::Modeling;
using namespace muq::Optimization;

TEST(CostFunctionTests, RosenbrockCost) {

  double a = 5.0;

  // the Rosenbrock cost function
  std::shared_ptr<CostFunction> rosen = std::make_shared<RosenbrockFunction>(a);

  // choose a random point
  const Eigen::VectorXd x = Eigen::Vector2d::Random();

  // the true value
  const double cst = (1.0-x(0))*(1.0-x(0))+a*(x(1)-x(0)*x(0))*(x(1)-x(0)*x(0));

  // check the cost evaluations
  EXPECT_DOUBLE_EQ(cst, rosen->Cost(x));

  // the true gradient
  const Eigen::Vector2d grad_true(-4.0*a*(x(1)-x(0)*x(0))*x(0)-2.0*(1.0-x(0)), 2.0*a*(x(1)-x(0)*x(0)));

  // compute the gradient
  Eigen::VectorXd grad_test0 = rosen->Gradient(x);

  EXPECT_DOUBLE_EQ((grad_true-grad_test0).norm(), 0.0);

  // the true hessian
  Eigen::Matrix2d hess_temp;
  hess_temp << 12.0*a*x(0)*x(0)-4.0*a*x(1)+2.0, -4.0*a*x(0),
               -4.0*a*x(0), 2.0*a;
  Eigen::Matrix2d hess_true(hess_temp);

  // compute the Hessian
  Eigen::MatrixXd hess_test0 = rosen->Hessian(x);

  EXPECT_NEAR((hess_true-hess_test0).norm(), 0.0, 1.0e-5);

  // Test the Hessian action
  Eigen::Vector2d vec(-12.3, 34.6);

  Eigen::VectorXd hessAction_true = hess_true*vec;
  Eigen::VectorXd hessAction_test = rosen->ApplyHessian(x,vec);


  EXPECT_NEAR((hessAction_true-hessAction_test).norm(), 0.0, 3.0e-4);
}
