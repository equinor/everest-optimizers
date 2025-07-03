#include "MUQ/Optimization/CostFunction.h"

/// A muq::Modeling::ModPiece version
class RosenbrockModPiece : public muq::Modeling::ModPiece {
public:
  inline RosenbrockModPiece(double aIn) : muq::Modeling::ModPiece(Eigen::VectorXi::Constant(1, 2), Eigen::VectorXi::Constant(1, 1)), a(aIn) {}

  virtual inline ~RosenbrockModPiece() {}

 private:
  const double a;

  inline virtual void EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) override {
    const Eigen::VectorXd& xc = input[0];
   
    outputs.resize(1);
    outputs[0] = (Eigen::VectorXd)Eigen::VectorXd::Constant(1, (1.0-xc(0))*(1.0-xc(0))+a*(xc(1)-xc(0)*xc(0))*(xc(1)-xc(0)*xc(0)));
  }

  inline virtual void GradientImpl(unsigned int const outputDimWrt, unsigned int const inputDimWrt, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity) override {
    assert(outputDimWrt==0);
    assert(inputDimWrt==0);

    const Eigen::VectorXd& xc = input[0];
    
    gradient = Eigen::Vector2d::Constant(2, std::numeric_limits<double>::quiet_NaN());
    gradient(0) = -4.0*a*(xc(1)-xc(0)*xc(0))*xc(0)-2.0*(1.0-xc(0));
    gradient(1) = 2.0*a*(xc(1)-xc(0)*xc(0));

    gradient *= sensitivity(0);
  }

  inline virtual void ApplyHessianImpl(unsigned int outWrt, unsigned int inWrt1, unsigned int inWrt2, muq::Modeling::ref_vector<Eigen::VectorXd> const& input, Eigen::VectorXd const& sensitivity, Eigen::VectorXd const& vec) override {

    assert(inWrt2==0);

    const Eigen::VectorXd& xc = input[0];

    Eigen::MatrixXd hess(2,2);
    hess(0,0) = -4.0*a*((xc(1)-xc(0)*xc(0)) - 2.0*xc(0)*xc(0)) + 2.0;
    hess(0,1) = -4.0*a*xc(0);
    hess(1,0) = hess(0,1);
    hess(1,1) = 2.0*a;

    hessAction = sensitivity(0)*(hess*vec);
  }
};

/// A muq::Optimization::CostFunction version
class RosenbrockFunction : public muq::Optimization::CostFunction {
public:
  RosenbrockFunction(double aIn) : CostFunction(2), a(aIn) {}

  virtual ~RosenbrockFunction() = default;

  virtual Eigen::MatrixXd Hessian() override {

    Eigen::MatrixXd hess(2,2);
    hess(0,0) = 2.0 - 4.0*a*(x(1)-3.0*x(0)*x(0));
    hess(1,0) = -4.0*a*x(0);
    hess(0,1) = hess(1,0);
    hess(1,1) = 2.0*a;

    return hess;
  }

  virtual double Cost() override {
    return (1.0-x(0))*(1.0-x(0))+a*(x(1)-x(0)*x(0))*(x(1)-x(0)*x(0));
  }

  virtual Eigen::VectorXd Gradient() override {

    gradient = Eigen::Vector2d::Constant(2, std::numeric_limits<double>::quiet_NaN());
    gradient(0) = -4.0*a*(x(1)-x(0)*x(0))*x(0)-2.0*(1.0-x(0));
    gradient(1) = 2.0*a*(x(1)-x(0)*x(0));

    return gradient;
  }

  virtual Eigen::VectorXd ApplyHessian(Eigen::VectorXd const& vec) override {

    Eigen::MatrixXd hess(2,2);
    hess(0,0) = -4.0*a*((x(1)-x(0)*x(0)) - 2.0*x(0)*x(0)) + 2.0;
    hess(0,1) = -4.0*a*x(0);
    hess(1,0) = hess(0,1);
    hess(1,1) = 2.0*a;

    return hess*vec;
  }

 private:

  const double a;

};
