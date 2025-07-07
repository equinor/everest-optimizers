#include "MUQ/SamplingAlgorithms/sMMALAProposal.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"

#include "MUQ/Utilities/AnyHelpers.h"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;




SMMALAProposal::SMMALAProposal(pt::ptree                                       pt,
                               std::shared_ptr<AbstractSamplingProblem> const& probIn,
                               std::shared_ptr<muq::Modeling::ModPiece> const& forwardModIn,
                                std::shared_ptr<muq::Modeling::Gaussian> const& priorIn,
                                std::shared_ptr<muq::Modeling::Gaussian> const& likelihoodIn) :
            MCMCProposal(pt,probIn),
            prior(priorIn), likelihood(likelihoodIn), model(forwardModIn), meanScaling(pt.get("MeanScaling",0.5))
{
  stepSize = pt.get("StepSize", 1.0);
  assert(stepSize>0);
  
  assert(meanScaling>0);
}

std::shared_ptr<SamplingState> SMMALAProposal::Sample(std::shared_ptr<SamplingState> const& currentState) {
  assert(currentState->state.size()>blockInd);

  // the mean of the proposal is the current point
  std::vector<Eigen::VectorXd> props = currentState->state;
  assert(props.size()>blockInd);
  Eigen::VectorXd const& xc = currentState->state.at(blockInd);

  // Get the gradient from the previous step
  //Eigen::VectorXd sigmaGrad; // vector holding the covariance of the proposal times the gradient

  std::stringstream blockId;
  blockId << "_" << blockInd;

  Eigen::MatrixXd hess;
  Eigen::VectorXd grad;
  if(!currentState->HasMeta("SMMALA_Hess" + blockId.str())){
    
    Eigen::MatrixXd jac = model->Jacobian(0,blockInd, currentState->state);
    hess = prior->GetPrecision() + jac.transpose() * likelihood->ApplyPrecision(jac);

    currentState->meta["SMMALA_Hess" + blockId.str()] = hess;
  }else{
    hess = AnyCast( currentState->meta["SMMALA_Hess" + blockId.str()] );
  }

  if(!currentState->HasMeta("SMMALA_Grad" + blockId.str())){
    grad = prob->GradLogDensity(currentState, blockInd);
    currentState->meta["SMMALA_Grad" + blockId.str()] = grad;
  }else{
      grad = AnyCast( currentState->meta["SMMALA_Grad" + blockId.str()] );
  }

  hess /= stepSize*stepSize;
  Gaussian prop(Eigen::VectorXd::Zero(xc.size()).eval(), hess, Gaussian::Mode::Precision);

  // Draw a sample
  props.at(blockInd) = xc + meanScaling*prop.ApplyCovariance(grad) + prop.Sample();
  
  // store the new state in the output
  return std::make_shared<SamplingState>(props, 1.0);
}

double SMMALAProposal::LogDensity(std::shared_ptr<SamplingState> const& currState,
                                  std::shared_ptr<SamplingState> const& propState) {

  std::stringstream blockId;
  blockId << "_" << blockInd;

  Eigen::MatrixXd hess;
  Eigen::VectorXd grad;
  if(!currState->HasMeta("SMMALA_Hess" + blockId.str())){
    
    Eigen::MatrixXd jac = model->Jacobian(0,blockInd, currState->state);
    hess = prior->GetPrecision() + jac.transpose() * likelihood->ApplyPrecision(jac);

    currState->meta["SMMALA_Hess" + blockId.str()] = hess;
  }else{
    hess = AnyCast( currState->meta["SMMALA_Hess" + blockId.str()] );
  }

  if(!currState->HasMeta("SMMALA_Grad" + blockId.str())){
    grad = prob->GradLogDensity(currState, blockInd);
    currState->meta["SMMALA_Grad" + blockId.str()] = grad;
  }else{
      grad = AnyCast( currState->meta["SMMALA_Grad" + blockId.str()] );
  }

  hess /= stepSize*stepSize;
  Gaussian prop(Eigen::VectorXd::Zero( propState->state.at(blockInd).size()).eval(), hess, Gaussian::Mode::Precision);

  // Draw a sample
  Eigen::VectorXd delta = propState->state.at(blockInd) - currState->state.at(blockInd) - meanScaling*prop.ApplyCovariance(grad);
  
  return prop.LogDensity(delta);
}
