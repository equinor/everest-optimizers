#include "MUQ/Optimization/ModPieceCostFunction.h"
#include "MUQ/Modeling/ModPiece.h"

using namespace muq::Modeling;
using namespace muq::Optimization;

ModPieceCostFunction::ModPieceCostFunction(std::shared_ptr<ModPiece> cost, double scaleIn) : CostFunction(cost->inputSizes(0)), cost(cost), scale(scaleIn) {
  // can only have one output of size one
  assert(cost->outputSizes.size()==1);
  assert(cost->outputSizes(0)==1);
  assert(cost->inputSizes.size()==1);
}

double ModPieceCostFunction::Cost() {
  assert(cost);
  return scale*cost->Evaluate(x).at(0) (0);
}

Eigen::VectorXd ModPieceCostFunction::Gradient() {
  assert(cost);
  Eigen::VectorXd sensitivity = Eigen::VectorXd::Ones(1);
  return scale*cost->Gradient(0, 0, x, sensitivity);
}

Eigen::VectorXd ModPieceCostFunction::ApplyHessian(Eigen::VectorXd const& vec) {
  assert(cost);
  Eigen::VectorXd sensitivity = Eigen::VectorXd::Ones(1);
  return scale*cost->ApplyHessian(0, 0, 0, ref_vector<Eigen::VectorXd>(1,x), sensitivity, vec);
}
