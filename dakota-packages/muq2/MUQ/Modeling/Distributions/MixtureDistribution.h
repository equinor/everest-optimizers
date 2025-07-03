#ifndef MIXTUREDISTRIBUTION_H_
#define MIXTUREDISTRIBUTION_H_

#include "MUQ/Modeling/Distributions/Distribution.h"

namespace muq {
  namespace Modeling {

    /**
    @brief Defines a mixture of distributions.
    @class MixtureDistributions
    @ingroup Distributions
    */
    class MixtureDistribution : public Distribution {
    public:

      /** Construct a mixture distribution from a list of components and a vector probabilities for each component.
      @param[in] components List of component distributions
      @param[in] probs Vector of probabilities.  Must have the same size as components. Probabilities will be normalized to sum to 1.
      */
      MixtureDistribution(std::vector<std::shared_ptr<Distribution>> const& components,
                          Eigen::VectorXd                             const& probs);

      virtual ~MixtureDistribution() = default;

      /** Compute the gradient of the log density with respect to either the
          distribution input or the hyperparameters. 
          @param[in] wrt Specifies the index of the variable we wish to take the gradient wrt.  If wrt==0, then the gradient should be taken wrt the input variable.
          @return The gradient of the log density.
      */
      virtual Eigen::VectorXd GradLogDensityImpl(unsigned int wrt,
                                                 ref_vector<Eigen::VectorXd> const& inputs) override;

      
      /** Returns the vector of component distributions. */
      std::vector<std::shared_ptr<Distribution>> Components(){return components;};

      /** Returns the probability vector. */
      Eigen::VectorXd Probabilities(){return probs;};

    protected:

      /**
      Compute the log density.
      @param[in] A vector of extra hyperparameter vectors.
      @return A double containing the log density.
      */
      virtual double LogDensityImpl(ref_vector<Eigen::VectorXd> const& inputs) override;

      /// Sample the distribution
      virtual Eigen::VectorXd SampleImpl(ref_vector<Eigen::VectorXd> const& inputs) override;

      std::vector<std::shared_ptr<Distribution>> components;
      Eigen::VectorXd probs;

    };
  } // namespace Modeling
} // namespace muq

#endif
