#ifndef GAMMA_H_
#define GAMMA_H_

#include "MUQ/Modeling/Distributions/Distribution.h"

namespace muq {
  namespace Modeling {

    /** Defines the Gamma distribution, which has probability density function
    \f[
    \pi(x) = \frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}\exp\left(-\beta x\right),
    \f]
    where \f$\alpha\f$ and \f$\beta\f$ are parameters in the distribution. For
    multivariate \f$x\f$, it is assumed that all components of \f$x\f$ are independent.
    However, each component can have different parameters \f$\alpha\f$ and \f$\beta\f$, which 
    are specified as vectors in the constructor.
    */
    class Gamma : public Distribution {
    public:

      Gamma(double       alphaIn,
            double       betaIn);

      Gamma(Eigen::VectorXd const& alphaIn,
            Eigen::VectorXd const& betaIn);

      virtual ~Gamma() = default;

      /**
      Creates a Gamma distribution given a mean and variance, which are converted to alpha and beta.
      */
      static std::shared_ptr<Gamma> FromMoments(Eigen::VectorXd const& mean,
                                                Eigen::VectorXd const& var);

      const Eigen::VectorXd alpha;
      const Eigen::VectorXd beta;

    private:

      static double ComputeConstant(Eigen::VectorXd const& alphaIn,
                                    Eigen::VectorXd const& betaIn);

      const double logConst; // sum( log( beta^alpha / Gamma(alpha) ) )

      virtual double LogDensityImpl(ref_vector<Eigen::VectorXd> const& inputs) override;

      virtual Eigen::VectorXd GradLogDensity(unsigned int wrt, 
                                             ref_vector<Eigen::VectorXd> const& inputs) override;

      virtual Eigen::VectorXd ApplyLogDensityHessian(unsigned int                const  inWrt1,
                                                     unsigned int                const  inWrt2,
                                                     ref_vector<Eigen::VectorXd> const& input,
                                                     Eigen::VectorXd             const& vec) override;

      /// Sample the distribution
      virtual Eigen::VectorXd SampleImpl(ref_vector<Eigen::VectorXd> const& inputs) override;

    };
  } // namespace Modeling
} // namespace muq

#endif
