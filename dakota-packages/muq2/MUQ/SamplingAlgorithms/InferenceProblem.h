#ifndef INFERENCEPROBLEM_H_
#define INFERENCEPROBLEM_H_

// include Density and not ModPiece so that if a SamplingProblem is constructed with a Density the compiler knows it is a child of ModPiece
#include "MUQ/Modeling/Distributions/Density.h"
#include "MUQ/SamplingAlgorithms/AbstractSamplingProblem.h"

namespace muq {
  namespace SamplingAlgorithms {

    /**
    @class InferenceProblem
    @brief Class for sampling problems based on the product of a prior and likelihood, possibly with a tempering 
           applied to the likelihood.
    @details This class implements posterior distributions \f$\pi(x_1,\ldots,x_N|y)\f$ over \f$N\f$ vectors 
             \f$x_1,\ldots,x_N\f$.  It is defined in terms of a likelihood, \f$\pi(y | x_1,\ldots,x_N)\f$,
             prior density \f$\pi(x)\f$ and optionally, a parameter \f$\beta\f$ called the "inverse temperature". 
             Together, these quantities define \f$\pi(x|y)\f$ up to a normalizing constant:
    \f\[
        \log\pi(x | y) \propto \beta\log\pi(y|x) +\log\pi(x).
    \f\]
            The inverse temperature \f$\beta\f$ is included to support implementations of parallel tempering 
            algorithms.
    */
    class InferenceProblem : public AbstractSamplingProblem{
    public:

      /**
	     @param[in] likelyIn A ModPiece that evaluates the log-likelihood function, which will be multiplied by the inverse temperature
         @param[in] priorIn A ModPiece that evaluates the log-prior density.  The number and sizes of the prior inputs must match the likelihood.
         @param[in] inverseTempIn The inverse temperature \f$\beta\f$ that scales the likelihood function. 
       */
      InferenceProblem(std::shared_ptr<muq::Modeling::ModPiece> const& likelyIn, 
                       std::shared_ptr<muq::Modeling::ModPiece> const& priorIn,
                       double                                          inverseTempIn=1.0);

      /**
	     @param[in] likelyIn A ModPiece that evaluates the log-likelihood function, which will be multiplied by the inverse temperature
         @param[in] priorIn A ModPiece that evaluates the log-prior density.  The number and sizes of the prior inputs must match the likelihood.
         @param[in] qoi Quantity of interest associated with model
         @param[in] inverseTempIn The inverse temperature \f$\beta\f$ that scales the likelihood function.
       */
      InferenceProblem(std::shared_ptr<muq::Modeling::ModPiece> const& likelyIn, 
                       std::shared_ptr<muq::Modeling::ModPiece> const& priorIn,
                       std::shared_ptr<muq::Modeling::ModPiece> const& qoiIn,
                       double                                          inverseTempIn=1.0);

      virtual ~InferenceProblem() = default;


      virtual double LogDensity(std::shared_ptr<SamplingState> const& state) override;

      virtual Eigen::VectorXd GradLogDensity(std::shared_ptr<SamplingState> const& state,
                                             unsigned                       const  blockWrt) override;

      virtual std::shared_ptr<SamplingState> QOI() override;

      std::shared_ptr<muq::Modeling::ModPiece> const& Likelihood() const{return likely;};
      std::shared_ptr<muq::Modeling::ModPiece> const& Prior() const{return prior;};

      /** Set the inverse temperature of this problem. */
      void SetInverseTemp(double newTemp){inverseTemp=newTemp;};

      /** Get the inverse temperature of this problem. */
      double GetInverseTemp() const{return inverseTemp;}

      virtual std::shared_ptr<AbstractSamplingProblem> Clone() const override;
      
    protected:

      /// The log-likelihood function
      std::shared_ptr<muq::Modeling::ModPiece> likely;

      /// The prior log-density
      std::shared_ptr<muq::Modeling::ModPiece> prior;
    
      std::shared_ptr<muq::Modeling::ModPiece> qoi;

      double inverseTemp;

    private:

      std::shared_ptr<SamplingState> lastState;
    };

  } // namespace SamplingAlgorithms
} // namespace muq

#endif
