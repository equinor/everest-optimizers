#include "MUQ/Modeling/Distributions/Gamma.h"
#include "MUQ/Utilities/RandomGenerator.h"
#include <Eigen/Core>

using namespace muq::Modeling;
using namespace muq::Utilities;

Gamma::Gamma(Eigen::VectorXd const& alphaIn,
             Eigen::VectorXd const& betaIn) : Distribution(alphaIn.size()),
                                    alpha(alphaIn),
                                    beta(betaIn),
                                    logConst(ComputeConstant(alphaIn, betaIn))
                                    {};

Gamma::Gamma(double       alphaIn,
             double       betaIn) : Gamma(alphaIn*Eigen::VectorXd::Ones(1),
                                          betaIn*Eigen::VectorXd::Ones(1)){};

double Gamma::ComputeConstant(Eigen::VectorXd const& alphaIn,
                              Eigen::VectorXd const& betaIn)
{
    double logConst = 0;
    for(int i=0; i<alphaIn.size(); ++i)
      logConst += alphaIn(i)*std::log(betaIn(i)) - std::lgamma(alphaIn(i));

    return logConst;
}

double Gamma::LogDensityImpl(ref_vector<Eigen::VectorXd> const& inputs)
{
  Eigen::VectorXd const& x = inputs.at(0).get();

  if(x.minCoeff()<std::numeric_limits<double>::epsilon())
    return -1.0*std::numeric_limits<double>::infinity();

  return logConst + ((alpha.array()-1.0)*x.array().log() - beta.array()*x.array()).sum();
}


Eigen::VectorXd Gamma::SampleImpl(ref_vector<Eigen::VectorXd> const& inputs)
{
  Eigen::VectorXd output(alpha.size());
  for(int i=0; i<alpha.size(); ++i)
    output(i) = RandomGenerator::GetGamma(alpha(i),1.0/beta(i));

  return output;
}

Eigen::VectorXd Gamma::GradLogDensity(unsigned int wrt, 
                                      ref_vector<Eigen::VectorXd> const& inputs)
{   
  Eigen::VectorXd const& x = inputs.at(0).get();

  Eigen::VectorXd grad(x.size());
  for(int i=0; i<x.size(); ++i){
      if(x(i)<std::numeric_limits<double>::epsilon()){
        grad(i) = 0.0;
      }else{
        grad(i) = (alpha(i)-1.0)/x(i) - beta(i);
      } 
  }

  return grad;
}

Eigen::VectorXd Gamma::ApplyLogDensityHessian(unsigned int                const  inWrt1,
                                              unsigned int                const  inWrt2,
                                              ref_vector<Eigen::VectorXd> const& inputs,
                                              Eigen::VectorXd             const& vec)
{
  Eigen::VectorXd const& x = inputs.at(0).get();

  Eigen::VectorXd hessAct(x.size());

  for(int i=0; i<x.size(); ++i){
      if(x(i)<std::numeric_limits<double>::epsilon()){
        hessAct(i) = 0.0;
      }else{
        hessAct(i) = -1.0 * vec(i) * (alpha(i)-1.0)/(x(i)*x(i));
      } 
  }

  return hessAct;
}


std::shared_ptr<Gamma> Gamma::FromMoments(Eigen::VectorXd const& mean,
                                          Eigen::VectorXd const& var)
{
   Eigen::VectorXd beta = mean.array() / var.array();
   Eigen::VectorXd alpha = mean.array()*beta.array();
   return std::make_shared<Gamma>(alpha,beta);
}
