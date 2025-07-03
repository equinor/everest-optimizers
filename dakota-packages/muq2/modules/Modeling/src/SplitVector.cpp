#include "MUQ/Modeling/SplitVector.h"

using namespace muq::Modeling;

SplitVector::SplitVector(std::vector<int> const& ind,
                         std::vector<int> const& size,
                         unsigned int const insize) : SplitVector(Eigen::Map<const Eigen::VectorXi>(&ind[0],ind.size()),
                                                                  Eigen::Map<const Eigen::VectorXi>(&size[0],size.size()),
                                                                  insize){};
                                                                  
SplitVector::SplitVector(Eigen::VectorXi const& ind,
                         Eigen::VectorXi const& size,
                         unsigned int const insize) : ModPiece(Eigen::VectorXi::Constant(1, insize), size),
                                                      ind(ind),
                                                      size(size) {
  assert(ind.size()==size.size());
  assert(size.sum()<=insize);
  assert(ind.maxCoeff()<insize);
}

void SplitVector::EvaluateImpl(ref_vector<Eigen::VectorXd> const& inputs) {
  const Eigen::VectorXd& in = inputs[0];

  outputs.resize(ind.size());
  for( unsigned int i=0; i<ind.size(); ++i ) {
    outputs[i] = in.segment(ind(i), size(i)).eval();
  }
}

void SplitVector::JacobianImpl(unsigned int const outwrt, unsigned int const inwrt, ref_vector<Eigen::VectorXd> const& inputs) {
  assert(inwrt==0);
  jacobian = Eigen::MatrixXd::Zero(size(outwrt), inputSizes(0));
  jacobian.block(0, ind(outwrt), size(outwrt), size(outwrt)) += Eigen::MatrixXd::Identity(size(outwrt), size(outwrt)).eval();
}

void SplitVector::GradientImpl(unsigned int const outwrt, unsigned int const inwrt, ref_vector<Eigen::VectorXd> const& inputs, Eigen::VectorXd const& sens) {
  assert(inwrt==0);
  assert(sens.size()==size(outwrt));
  gradient = Eigen::VectorXd::Zero(inputSizes(0));
  gradient.segment(ind(outwrt), size(outwrt)) += sens;
}

void SplitVector::ApplyJacobianImpl(unsigned int const outwrt, unsigned int const inwrt, ref_vector<Eigen::VectorXd> const& inputs, Eigen::VectorXd const& targ) {
  assert(inwrt==0);
  assert(targ.size()==inputSizes(0));
  jacobianAction = targ.segment(ind(outwrt), size(outwrt));
}

void SplitVector::ApplyHessianImpl(unsigned int const outwrt, unsigned int const inwrt1,unsigned int const inwrt2, ref_vector<Eigen::VectorXd> const& inputs, Eigen::VectorXd const& sens, Eigen::VectorXd const& targ) {

  assert(inwrt1==0);
  assert(inwrt2<2);

  hessAction = Eigen::VectorXd::Zero(inputSizes(0));

  // If hessian wrt sensitivity...
  if(inwrt2==1){
    hessAction.segment(ind(outwrt), size(outwrt)) = Eigen::VectorXd::Ones(size(outwrt));
  }
}
