#include "MUQ/Optimization/CostFunction.h"

using namespace muq::Modeling;
using namespace muq::Optimization;


void CostFunction::EvaluateImpl(ref_vector<Eigen::VectorXd> const& input) {
  outputs.resize(1);
  outputs.at(0) = Eigen::VectorXd::Constant(1, Cost(input.at(0)));
}

void CostFunction::GradientImpl(unsigned int const outputDimWrt,
                                unsigned int const inputDimWrt,
                                ref_vector<Eigen::VectorXd> const& input,
                                Eigen::VectorXd const& sensitivity) {
  gradient = sensitivity(0)*Gradient(input.at(0));
}

void CostFunction::SetPoint(Eigen::VectorXd const& evalPt) {
  assert(evalPt.size()==inputSizes(0));
  x = evalPt;
};

Eigen::VectorXd CostFunction::Gradient() {
  Eigen::VectorXd sens = Eigen::VectorXd::Ones(1);
  return ModPiece::GradientByFD(0, 0, muq::Modeling::ref_vector<Eigen::VectorXd>(1,x), sens);
}


Eigen::MatrixXd CostFunction::Hessian() {
  return HessianByFD();
}


Eigen::MatrixXd CostFunction::HessianByFD() {

  Eigen::VectorXd f0 = Gradient();
  Eigen::VectorXd x0 = x;
  Eigen::VectorXd f;

  double eps;

  Eigen::VectorXd newInput(x0);

  Eigen::MatrixXd hes(inputSizes(0), inputSizes(0));

  for (int i=0; i<inputSizes(0); ++i) {

    eps = std::max(1.0e-8, 1.0e-10*std::abs(x(i)));

    newInput(i) = x(i) + eps;
    f = Gradient(newInput);

    hes.col(i) = (f-f0)/eps;

    newInput(i) = x0(i);

  }

  // Reset the point to the original
  SetPoint(x0);

  return hes;
}


Eigen::VectorXd CostFunction::ApplyHessian(Eigen::VectorXd const& vec) {
  Eigen::VectorXd sens = Eigen::VectorXd::Ones(1);
  return ModPiece::ApplyHessianByFD(0,0,0,muq::Modeling::ref_vector<Eigen::VectorXd>({std::cref(x)}), sens, vec);
}

void CostFunction::JacobianImpl(unsigned int outputDimWrt, 
                                unsigned int inputDimWrt, 
                                muq::Modeling::ref_vector<Eigen::VectorXd> const& input)
{ 
  jacobian = Gradient(input.at(0)).transpose();
}
                                
void CostFunction::ApplyHessianImpl(unsigned int outWrt,
                                    unsigned int inWrt1,
                                    unsigned int inWrt2,
                                    muq::Modeling::ref_vector<Eigen::VectorXd> const& input,
                                    Eigen::VectorXd const& sensitivity,
                                    Eigen::VectorXd const& vec)
{
  hessAction = ApplyHessian(input.at(0),vec);
}

