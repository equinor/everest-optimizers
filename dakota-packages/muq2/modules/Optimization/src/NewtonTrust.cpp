#include "MUQ/Optimization/NewtonTrust.h"

#include <stdio.h>

using namespace muq::Optimization;

REGISTER_OPTIMIZER(NewtonTrust, NewtonTrust)

NewtonTrust::NewtonTrust(std::shared_ptr<muq::Modeling::ModPiece> const& cost,
                         boost::property_tree::ptree const& pt) : Optimizer(cost, pt),
                             maxRadius(pt.get("MaxRadius", std::numeric_limits<double>::infinity())),
                             initialRadius(pt.get("InitialRadius", std::min(1.0,maxRadius))),
                             acceptRatio(pt.get("AcceptRatio", 0.05)),
                             shrinkRatio(pt.get("ShrinkRatio", 0.25)),
                             growRatio(pt.get("GrowRatio", 0.75)),
                             shrinkRate(pt.get("ShrinkRate",0.25)),
                             growRate(pt.get("GrowRate", 2.0)),
                             trustTol(pt.get("TrustTol", std::min(1e-4,xtol_abs))),
                             printLevel(pt.get("PrintLevel", 0)){
}



std::pair<Eigen::VectorXd, double> NewtonTrust::Solve(std::vector<Eigen::VectorXd> const& inputs) {

  // Trust region approach with a double dogleg step
  trustRadius = initialRadius;

  Eigen::VectorXd const& x0 = inputs.at(0);
  Eigen::VectorXd x = x0;
  Eigen::VectorXd step;

  if(printLevel>0){
    std::cout << "Using NewtonTrust optimizer..." << std::endl;
    std::cout << "  Iteration, TrustRadius,       fval,      ||g||" << std::endl;
  }

  double fval;

  for(int it=0; it<maxEvals; ++it) {

    opt->SetPoint(x);
    fval = opt->Cost();
    Eigen::VectorXd const& grad = opt->Gradient();

    if(printLevel>0){
      char buf[1024]; 
      std::sprintf(buf, "  %9d, %11.3f,  %4.3e,  %5.3e\n", it, trustRadius, fval, grad.norm()); // create string and then cout so pybind11 can capture output
      std::cout << std::string(buf);
    }//std::printf("  %9d, %11.3f,  %4.3e,  %5.3e\n", it, trustRadius, fval, grad.norm());

    if(grad.norm() < xtol_abs)
        return std::make_pair(x,fval);

    step = SolveSub(fval, x, grad);

    double modDelta = grad.dot(step)+0.5*step.dot(opt->ApplyHessian(step));
    Eigen::VectorXd newX = x+step;

    double newf = opt->Cost(newX);
    double rho = (newf-fval)/modDelta;

    // Update the position.  If the model is really bad, we'll just stay put
    if(rho>acceptRatio){

      if(step.norm() < xtol_abs)
        return std::make_pair(newX,newf);

      if((fval-newf)<ftol_abs)
        return std::make_pair(newX,newf);

      x = newX;
      fval = newf;
    }

    // Update the trust region size
    if(rho<shrinkRatio){
      trustRadius = shrinkRate*trustRadius; // shrink trust region
    }else if((rho>growRatio)&&(std::abs(step.norm()-trustRadius)<1e-10)) {
      trustRadius = std::min(growRate*trustRadius, maxRadius);
    }

  }

  return std::make_pair(x,fval);
}


Eigen::VectorXd NewtonTrust::SolveSub(double                 fval,
                                      Eigen::VectorXd const& x0,
                                      Eigen::VectorXd const& grad) {
  const unsigned int dim = x0.size();

  // Current estimate of the subproblem minimum
  Eigen::VectorXd z = Eigen::VectorXd::Zero(dim);

  // Related to the step direction
  Eigen::VectorXd r = grad;
  Eigen::VectorXd d = -r;

  // If the gradient is small enough where we're starting, then we're done
  if(r.norm()<trustTol)
    return z;

  Eigen::VectorXd Bd; // the Hessian (B) applied to a vector d

  double alpha, beta, gradd, dBd, rr;

  for(int i=0; i<dim; ++i){
    Bd = opt->ApplyHessian(d);
    gradd = grad.dot(d);
    dBd = d.dot(Bd);
    rr = r.squaredNorm();

    // If the Hessian isn't positive definite in this direction, we can go all
    // the way to the trust region boundary
    if(dBd<=0){
      // do something

      double dz = d.dot(z);
      double dd = d.squaredNorm();
      double zz = z.squaredNorm();
      double r2 = trustRadius*trustRadius;

      double tau1 = (-dz + sqrt(dz*dz - dd*(zz-r2)))/dd;
      double tau2 = (-dz - sqrt(dz*dz - dd*(zz-r2)))/dd;

      double zBd = z.dot(Bd);
      double mval1 = tau1*gradd + tau1*zBd + tau1*tau1*dBd;
      double mval2 = tau2*gradd + tau2*zBd + tau2*tau2*dBd;

      return (mval1<mval2) ? (z+tau1*d) : (z+tau2*d);
    }

    alpha = rr / dBd;
    Eigen::VectorXd newZ = z + alpha * d;

    if(newZ.norm()>trustRadius){

      double dz = d.dot(z);
      double dd = d.squaredNorm();
      double zz = z.squaredNorm();
      double r2 = trustRadius*trustRadius;

      double tau = (-dz + sqrt(dz*dz - dd*(zz-r2)))/dd;
      return z + tau*d;
    }

    z = newZ;

    r += alpha*Bd;

    if(r.norm()<trustTol)
      return z;

    beta = r.squaredNorm() / rr;
    d = (-r + beta*d).eval();
  }

  return z;
}
