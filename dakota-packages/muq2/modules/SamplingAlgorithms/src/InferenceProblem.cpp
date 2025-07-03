#include "MUQ/SamplingAlgorithms/InferenceProblem.h"
#include "MUQ/SamplingAlgorithms/SamplingState.h"

using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;

InferenceProblem::InferenceProblem(std::shared_ptr<muq::Modeling::ModPiece> const& likelyIn,
                                   std::shared_ptr<muq::Modeling::ModPiece> const& priorIn, 
                                   double                                          inverseTempIn) : AbstractSamplingProblem(likelyIn->inputSizes),
                                                                                                    likely(likelyIn),
                                                                                                    prior(priorIn),
                                                                                                    inverseTemp(inverseTempIn){}

InferenceProblem::InferenceProblem(std::shared_ptr<muq::Modeling::ModPiece> const& likelyIn,
                                   std::shared_ptr<muq::Modeling::ModPiece> const& priorIn,
                                   std::shared_ptr<muq::Modeling::ModPiece> const& qoiIn,
                                   double                                          inverseTempIn) : AbstractSamplingProblem(likelyIn->inputSizes),
                                                                                                    likely(likelyIn),
                                                                                                    prior(priorIn),
                                                                                                    qoi(qoiIn),
                                                                                                    inverseTemp(inverseTempIn) {}


double InferenceProblem::LogDensity(std::shared_ptr<SamplingState> const& state) {
  assert(likely);
  assert(prior);

  lastState = state;
  double likelyVal = likely->Evaluate(state->state).at(0)(0);
  double priorVal = prior->Evaluate(state->state).at(0)(0);
  state->meta["LogLikelihood"] = likelyVal;
  state->meta["LogPrior"] = priorVal;
  state->meta["InverseTemp"] = inverseTemp;
  
  return inverseTemp * likelyVal + priorVal;
}

std::shared_ptr<SamplingState> InferenceProblem::QOI() {
  assert(lastState);
  if (qoi == nullptr)
    return nullptr;

  return std::make_shared<SamplingState>(qoi->Evaluate(lastState->state));
}

Eigen::VectorXd InferenceProblem::GradLogDensity(std::shared_ptr<SamplingState> const& state,
                                                 unsigned                       const  blockWrt)
{   
  assert(likely);
  assert(prior);

  Eigen::VectorXd sens = Eigen::VectorXd::Ones(1);
  return inverseTemp * likely->Gradient(0, blockWrt, state->state, sens) + prior->Gradient(0, blockWrt, state->state, sens);
}

std::shared_ptr<AbstractSamplingProblem> InferenceProblem::Clone() const{
    return std::make_shared<InferenceProblem>(likely, prior, qoi, inverseTemp);
}