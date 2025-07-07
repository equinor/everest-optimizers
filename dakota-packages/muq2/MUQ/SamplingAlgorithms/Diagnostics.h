#ifndef DIAGNOSTICS_H_
#define DIAGNOSTICS_H_

#include <vector>

#include "MUQ/SamplingAlgorithms/SampleCollection.h"
#include "MUQ/SamplingAlgorithms/SampleEstimator.h"

#include <boost/property_tree/ptree.hpp>

namespace muq{
  namespace SamplingAlgorithms{

    namespace Diagnostics{
      /**
          @ingroup mcmcdiag
          @{
      */


      /** If this function is given a vector of SampleCollection instances, it can return either the standard scale reduction
          \f$\hat{R}\f$ diagnostic from \cite Gelman2013, one of the modifications presented in \cite Vehtari2021, the multivariate 
          potential scale reduction factor (MPSRF) of \cite Brooks1998, or multivariate adapations of MPSRF that are similar 
          to the split and ranked methods of \cite Vehtari2021.
          
          NOTE: If the SampleEstimator inputs cannot be cast to SampleCollections, then the unsplit \f$\hat{R}\f$ diagnostic from \cite Gelman2013 is employed.  
          
          The basic \f$\hat{R}\f$ estimator of \cite Gelman2013 and the MPSRF estimator of \cite Brooks1998 only require sample estimates of the
          mean and variance.  They do not require access to individual samples or knowledge of the length of each 
          chain.  These estimators can therefore be employed with methods like multi-index MCMC (see the MIMCMC class), which 
          utilize more sophisticated approaches for computing sample-based expectations where notions of chain length 
          and the definition of a sample are less straightforward.  
          
          Consider \f$m\f$ independent chains of length \f$n\f$.   Let \f$x_{ij}\f$ denote the \f$i^{th}\f$ sample 
          of the \f$j^{th}\f$ chain.  The original scale reduction factor \f$\hat{R}\f$ from \cite Gelman2013 is 
          defined in terms of the variance between chains, denoted by \f$B\f$, and the average variance within each
          chain, denoted by \f$W\f$.   Mathematically, $B$ and $W$ are given by 
          \f[
           B = \frac{n}{m-1} \sum_{j=1}^m\left( \bar{x}_{\cdot j} - \bar{x}_{\cdot \cdot}\right)^2
          \f]
          and
          \f[
            W = \frac{1}{m}\sum_{j=1}^m s_j^2,
          \f]
          where \f$s_j^2\f$ is the sample variance of chain \f$j\f$ given by 
          \f[
            s_j^2 = \frac{1}{n-1}\sum_{i=1}^n \left(x_{ij} - \bar{x}_{\cdot j}\right)^2,
          \f]
          \f$\bar{x}_{\cdot j}\f$ is the sample mean of chain \f$j\f$, given by 
          \f[
            \bar{x}_{\cdot,j} = \frac{1}{n}\sum_{i=1}^n x_{ij},
          \f]
          and \f$\bar{x}_{\cdot \cdot}\f$ is the sample mean computed with all chains, given by
          \f[
            \bar{x}_{\cdot \cdot} = \frac{1}{m}\sum_{j=1}^m \bar{x}_{\cdot j}.
          \f]
          Using \f$W\f$ and \f$B\f$ it is possible to construct an estimate for the variance of \f$x\f$ 
          using
          \f[
            \hat{\text{var}}(x) = \frac{n-1}{n} W + \frac{1}{n}B.
          \f]
          When each chain is started from overdisperse initial points \f$x_{0j}\f$ (i.e., sampled from a distribution
          with larger variance than the  target distribution \f$p(x)\f$), the variance estimate 
          \f$\hat{\text{var}}(x)\f$ tends to be larger than the true target variance \f$\text{Var}[x]\f$.   Similarly, the 
          average within chain variance \f$W\f$ tends to be smaller than the true variance \f$\text{Var}[x]\f$.  Comparing 
          the ratio of these two estimates to \f$1\f$ thus provides a way of detecting ``convergence.''  In particular, 
          the \f$\hat{R}\f$ diagnostic, given by 
          \f[
            \hat{R} = \sqrt{\frac{\hat{\text{var}}(x)}{W}},
          \f]
          can be used to assess whether the chains have converged.   

          For general estimator classes (children of SampleEstimator, like MultiIndexSampleEstimator), this function
          will compute \f$\hat{R}\f$ by expanding \f$W\f$ and \f$B\f$ in the definition of \f$\hat{\text{var}}(x)\f$ into 
          \f[
            \hat{\text{var}}(x) = \frac{1}{m}\sum_{j=1}^m s_{bj}^2 + \frac{1}{m-1} \sum_{j=1}^m\left( \bar{x}_{\cdot j} - \bar{x}_{\cdot \cdot}\right)^2,
          \f]
          where \f$s_{bj}\f$ is a biased estimate of the variance using samples from chain \f$j\f$, which for standard Markov chains
          is given by
          \f[
            s_{bj} = \frac{1}{n}\sum_{i=1}^n \left(x_{ij} - \bar{x}_{\cdot j}\right)^2.
          \f]
          This function computes \f$s_{bj}\f$ by calling the SampleEstimator::ExpectedValue function.  For non-standard 
          estimators, like those defined by the MultiIndexEstimator class, the value of \f$s_{bj}\f$ might not be given
          by a simple summation like that shown here.   Regardless, this function will use the SampleEstimator class to compute
          \f$s_{bj}\f$ and \f$\bar{x}_{\cdot j}\f$ for each component of the `estimators` vector before computing 
          \f$\hat{\text{var}}(x)\f$ and \f$\hat{R}\f$ using the expressions above.

          Note: To interpret \f$\hat{R}\f$, it is critical that each of the independent chains have be started from 
          "overdisperse" points.   Sampling \f$x_{0,j}\f$ from the prior distribution is a good way of accomplishing this 
          when sampling Bayesian posterior distributions.

          If the estimators are can be cast to the `SampleCollection` class, the following options will be passed 
          on to the SplitRankRhat function.

      Parameter Key | Type | Default Value | Description |
      ------------- | ------------- | ------------- | ------------- |
      "Split"  | boolean | True  | Whether the chains should be split in half as proposed by \cite Vehtari2021. |
      "Transform"   | boolean | False  | If the parameters should be rank-transformed before computing Rhat, as in \cite Vehtari2021. |
      "Multivariate" | boolean | False | If the MPSRF value should be returned instead of the componentwise \$\hat{R}\f$ statistic. If `true`, the output vector will have a single component.  The MPSRF serves as a worse case estimate of \f$\hat{R}\f$ over all linear combinations of the parameters. |

      @param[in] collections A vector of SampleEstimator variables returned by independent runs of an MCMC algorithm.
      @param[in] options (optional) A property tree possibly containing settings for the "Split" or "Transform" parameters listed above.  Note that the Split and transform options are only available for children of the SampleCollection class.
      @returns If Multivariate==False, a vector of \f$\hat{R}\f$ values for each component of the parameters.  If Multivariate==true, then a length 1 vector containing the MPSRF.
      */
      template<typename EstimatorType>
      Eigen::VectorXd Rhat(std::vector<std::shared_ptr<EstimatorType>> const& estimators,
                           boost::property_tree::ptree                           options = boost::property_tree::ptree());

      /** Computes the standard \f$\hat{R}\f$ diagnostic from \cite Gelman2013 on the given chains. 
      Returns a vector containing \f$\hat{R}\f$ for each component of the chain.
      */
      Eigen::VectorXd BasicRhat(std::vector<std::shared_ptr<SampleEstimator>> const& collections);

      /** Computes the standard multivariate \f$\hat{R}^p\f$ MPSRF diagnostic from Section 4.1 of \cite Brooks1998 on the given chains.

      NOTE: There is a (N-1)/N scaling term in the MPSRF definition, where $N$ is chain length. If called with general esimators that do not have a "size()" function, like the multilevel MCMC estimators), this term is set to 1.0, which may result in biased estimates for small $N$.
      */
      double BasicMPSRF(std::vector<std::shared_ptr<SampleEstimator>> const& collections);

      /** Splits the chains into equally sized segments. */
      std::vector<std::shared_ptr<SampleCollection>> SplitChains(std::vector<std::shared_ptr<const SampleCollection>> const& collections, unsigned int numSegs=2);
      std::vector<std::shared_ptr<SampleCollection>> SplitChains(std::vector<std::shared_ptr<SampleCollection>> const& origChains, unsigned int numSegments=2);

      /** Performas an inplace transformation of the chains based on ranking the samples and applying a Gaussian transform. */
      std::vector<std::shared_ptr<SampleCollection>> TransformChains(std::vector<std::shared_ptr<SampleCollection>> const& collections);


      /** For a set of scalar values \f$\{x_1,\ldots, x_S\}\f$, the rank of \f$x_i\f$ is the index of \f$x_i\f$ after sorting this set into a list that satisfies \f$x_i\leq x_{i+1}\f$.  We use \f$r_i\f$  to denote the rank of \f$x_i\f$ and adopt the convention that for repeated values (i.e., \f$x_{i}=x_{i+1}\f$), \f$r_i\f$ is given by the average rank of consecutive repeated values.

      This function computes and returns the values of \f$r_i\f$ when the initial set is given by the combined states of multiple SampleCollections.   The states are vector-valued, but this function operates only on a single component of the state vector.

      @param[in] collections Several SampleCollection instances used to define the set of values we want to rank
      @param[in] dim The component of the vector-valued states that we want to rank.
      @return A std::vector of Eigen::VectorXd containing the ranks of all samples in all sample collections.  The std::vector has the same size as the std::vector of collections.  Following \cite Vehtari2021, "Average rank for ties are used to conserve the number of unique values of discrete quantities."
      */
      std::vector<Eigen::VectorXd> ComputeRanks(std::vector<std::shared_ptr<SampleCollection>> const& collections,
                                                unsigned int                                          dim);

      /**
      @}
      */
      
    } // namespace Diagnostics
  } // namespace SamplingAlgorithms
} // namespace muq

#endif // #ifndef DIAGNOSTICS_H_
