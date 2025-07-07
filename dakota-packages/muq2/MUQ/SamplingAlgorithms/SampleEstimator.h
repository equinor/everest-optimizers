#ifndef SAMPLEESTIMATOR_H
#define SAMPLEESTIMATOR_H

#include <Eigen/Core>

#include "MUQ/Modeling/ModPiece.h"

namespace muq{
namespace SamplingAlgorithms{

  /**
   @class SampleEstimator
   @ingroup mcmc
   @brief Abstract base class for computing sample-based approximations of expectations.
   @details Consider a vector-valued random variable $x$ taking values in \f$\mathbb{R}^N\f$.  Now consider a decomposition of this 
            random variable into \f$M\f$ "blocks" \f$x_1, x_2, \ldots, x_M\f$ with lengths \f$N_1,\ldots, N_M\f$, where \f$\sum N_i = N\f$.    
            This class provides an abstract interface for methods that approximate expectations of the form 
            \f[
            \mathbb{E}[f(x)] = \int f(x) p(x) dx
            \f]  
            or
            \f[
            \mathbb{E}[f(x_i)] = \int f(x_i) p(x_i) dx,
            \f]
            where \f$p(x)\f$ and \f$p(x_i)\f$ are the distributions of \f$x\f$ and \f$x_i\f$, respectively.  This class of expectations
            include the mean \f$\mu=\mathbb{E}[x]\f$ and variance \f$\sigma^2 = \mathbb{E}[(x-\mu)^2]\f$. 

            In Monte Carlo, samples \f$x^{(k)}\f$ of the random variable are used to approximate the expectation via the law of 
            large numbers:
            \f[
            \mathbb{E}[f(x)] \approx \frac{1}{K} \sum_{k=1}^K f\left(x^{(k)}\right). 
            \f]
            The muq::SamplingAlgorithms::SampleCollection class is a child of this class that provides such Monte Carlo 
            estimators.   Of course, it is possible to construct other estimators of the expection, using 
            multilevel Monte Carlo methods for example.   This class aims to provide a common interface for all such approaches.
  */
  class SampleEstimator {

  public:

      virtual ~SampleEstimator() = default;
    
      /** Returns the size \f$N_i\f$ of each block.   If `blockInd==-1`, the size \f$N\f$ of the joint random variable is 
          returned.
      */ 
      virtual unsigned int BlockSize(int blockInd) const = 0;

      /** Returns the nubmer of block \f$M\f$. */
      virtual unsigned int NumBlocks() const = 0;

      /**  The central moment of order $p$ is given by 
           \f[
           \mathbb{E}\left[ \left(X-\mathbb{E}[X]\right)^p \right]
           \f]
           This function returns a sample-based estimate of the central moment.   In the default implementation of this function,
           the SampleEstimator::ExpectedValue function is called twice.  Once to compute \f$\mathbb{E}[X]\f$ and then again to compute 
           \f$\mathbb{E}\left[ \left(X-\mathbb{E}[X]\right)^p \right]\f$.   Child classes might provide alternative implementations.
        
           @param[in] order The order \f$p\f$ of the central moment.   \f$p=2\f$ yields the variance.
           @param[in] blockNum (Optional) The block of the random variable \f$x\f$ to use in the expectation.  By default, `blockNum=-1`
                      and the expectation is computed with respect to the entire random variable \f$x\f$.
           @return A vector with the same size as \f$x\f$ or \f$x_i\f$ containing an estimate of the central moment.
      */
      virtual Eigen::VectorXd CentralMoment(unsigned int order, 
                                            int          blockNum=-1) const;

      /** Compute the central moment, as in the other SampleEstimator::CentralMoment function, but use a precomputed (or known) 
          mean.  Note that using a vector of zeros for the mean allows non-central moments to be computed.

          @param[in] order The order \f$p\f$ of the central moment.   \f$p=2\f$ yields the variance.
          @param[in] mean A vector containing the mean of \f$x\f$ (if `blockNum==-1`) or \f$x_i\f$ (if `blockNum==i`).
          @param[in] blockNum (Optional) The block of the random variable \f$x\f$ to use in the expectation.  By default, blockNum=-1
                      and the expectation is computed with respect to the entire random variable $x$.
          @return A vector with the same size as \f$x\f$ or \f$x_i\f$ containing an estimate of the central moment.
      */
      virtual Eigen::VectorXd CentralMoment(unsigned int           order, 
                                            Eigen::VectorXd const& mean, 
                                            int                    blockNum=-1) const;

      /**
       The standardize moment of order $p$ is similar to the central moment, but also includes a scaling of the random variable 
       \f$x\f$ by the standard deviation.   Mathematially, the standardized moment is given by 
       \f[
        \mathbb{E}\left[ \left(\frac{x-\mu}{\sigma}\right)^p\right],
       \f]
       where \f$\mu=\mathbb{E}[x]\f$ is the mean and $\sigma$ is the standard deviation of $x$.   

       In the default implementation of this function, three calls to the ExpectedValue function will be made to 
       compute the mean, compute the variance, and then to compute the outer expectation.

       Note that the standardized moment of order \f$p=3\f$ is commonly called the skewness and the standardized moment of 
       order $p=4$ is called the kurtosis.  Shortcuts for these common moments can be found in the SampleEstimator::Skewness
       and SampleEstimator::Kurtosis functions.

       @param[in] order The order \f$p\f$ of the central moment.   \f$p=2\f$ yields the variance.
       @param[in] blockNum (Optional) The block of the random variable \f$x\f$ to use in the expectation.  By default, `blockNum=-1`
                           and the expectation is computed with respect to the entire random variable \f$x\f$.
       @return A vector with the same size as \f$x\f$ or \f$x_i\f$ containing an estimate of the standardized moment.
      */
      virtual Eigen::VectorXd StandardizedMoment(unsigned int order,
                                                 int          blockInd=-1) const;

      /**
       The standardize moment of order $p$ is similar to the central moment, but also includes a scaling of the random variable 
       \f$x\f$ by the standard deviation.   Mathematially, the standardized moment is given by 
       \f[
        \mathbb{E}\left[ \left(\frac{x-\mu}{\sigma}\right)^p\right],
       \f]
       where \f$\mu=\mathbb{E}[x]\f$ is the mean and $\sigma$ is the standard deviation of $x$. 

       @param[in] order The order \f$p\f$ of the central moment.   \f$p=2\f$ yields the variance.
       @param[in] mean A vector containing a precomputed (or known) estimate of \f$\mu=\mathbb{E}[x]\f$.
       @param[in] blockNum (Optional) The block of the random variable \f$x\f$ to use in the expectation.  By default, `blockNum=-1`
                           and the expectation is computed with respect to the entire random variable \f$x\f$.
       @return A vector with the same size as \f$x\f$ or \f$x_i\f$ containing an estimate of the standardized moment.
      */
      virtual Eigen::VectorXd StandardizedMoment(unsigned int           order,
                                                 Eigen::VectorXd const& mean,
                                                 int                    blockInd=-1) const;

      /** Computes the standardized moment with precomputed (or known) mean and standard deviation. */
      virtual Eigen::VectorXd StandardizedMoment(unsigned int           order,
                                                 Eigen::VectorXd const& mean,
                                                 Eigen::VectorXd const& stdDev,
                                                 int                    blockInd=-1) const;

      

      /**
       Computes the sample mean of \f$x\f$ (if `blockInd==-1`) or \f$x_i\f$ (if `blockInd==i`).  
       If blockInd is non-negative, only the mean of one block of the samples is computed.
      */
      virtual Eigen::VectorXd Mean(int blockInd=-1) const;
    
      /**
       Computes an estiamte of the variance of \f$x\f$ (if `blockInd==-1`) or \f$x_i\f$ (if `blockInd==i`).  The variance is defined 
       as 
       \f[
       \text{Var}[x] = \mathbb{E}\left[\left(x-\mathbb{E}[x]\right)^2\right].
       \f]
       If blockInd is non-negative, only the variance of one block, i.e. \f$\text{Var}[x_i]\f$ of the samples is computed.

       Note that the default implementation of this function makes two calls to SampleEstimator::ExpectedValue.
       One call is used to compute the mean \f$\mathbb{E}[x]\f$ and another call is used to evaluate the outer expectation 
       \f$\mathbb{E}\left[\left(x-\mathbb{E}[x]\right)^2\right]\f$.   In some cases this can lead to biased estimates of the variance.

       The SamplingAlgorithms::SampleCollection child of this class provides an unbiased implementation for use with 
       standard Monte Carlo and Markov chain Monte Carlo samplers.
      */
      virtual Eigen::VectorXd Variance(int blockInd=-1) const;

      /**
       Computes the sample variance using a precomputed (or known) mean vector.  If blockInd is non-negative, only the mean of one block of the samples is computed.
      */
      virtual Eigen::VectorXd Variance(Eigen::VectorXd const& mean, 
                                       int                    blockInd=-1) const;

      /**
       The marginal skewness of a random variable is given by 
       \f[
           \tilde{\mu}_3 = \mathbb{E}\left[ \left(\frac{x-\mu}{\sigma}\right)^3 \right],
       \f]
       where \f$\mu\f$ is the mean and \f$\sigma\f$ is the standard deviation.  This function returns
       an estimate of this quantity by using sample approximations of \f$\mu\f$, \f$\sigma\f$, and the outer expectation.

       In the default implementation, three calls to the ExpectedValue function are used to compute this 
       quantity.  Children of this class, like the SampleCollection, may provide more efficient implementations.

       Just like the variance is the diagonal of the covariance matrix, this function returns a vector representing
       the "diagonal" of the full third order skewness tensor.
      */
      virtual Eigen::VectorXd Skewness(int blockInd=-1) const;

      /** Evaluate the skewness using a precomputed (or known) mean vector. */
      virtual Eigen::VectorXd Skewness(Eigen::VectorXd const& mean,
                                       int                    blockInd=-1) const;

      virtual Eigen::VectorXd Skewness(Eigen::VectorXd const& mean,
                                       Eigen::VectorXd const& stdDev,
                                       int                    blockInd=-1) const;
      /**
       The marginal kurtosis of a random variable is given by 
       \f[
           \tilde{\mu}_4 = \mathbb{E}\left[ \left(\frac{x-\mu}{\sigma}\right)^4 \right],
       \f]
       where \f$\mu\f$ is the mean and \f$\sigma\f$ is the standard deviation.  This function returns
       an estimate of this quantity by using sample approximations of \f$\mu\f$, \f$\sigma\f$, and the outer expectation.

       In the default implementation, three calls to the ExpectedValue function are used to compute this 
       quantity.  Children of this class, like the SampleCollection, may provide more efficient implementations.

       Just like the variance is the diagonal of the covariance matrix, this function returns a vector representing
       the "diagonal" of the full fourth order kurtosis tensor.
      */
      virtual Eigen::VectorXd Kurtosis(int blockInd=-1) const;

      /** Evaluate the kurtosis using a precomputed (or known) mean vector. */
      virtual Eigen::VectorXd Kurtosis(Eigen::VectorXd const& mean,
                                       int                    blockInd=-1) const;

      virtual Eigen::VectorXd Kurtosis(Eigen::VectorXd const& mean,
                                       Eigen::VectorXd const& stdDev,
                                       int                    blockInd=-1) const;
      /**
       Computes the sample covariance of \f$x\f$ with itself (if `blockInd==-1`) or \f$x_i\f$ with itself (if `blockInd==i`), i.e., 
       \f[
       \text{Cov}[x] = \mathbb{E}\left[ \left(x-\mathbb{E}[x]\right)\left(x-\mathbb{E}[x]\right)^T\right]
       \f]
       or 
       \f[
       \text{Cov}[x_i] = \mathbb{E}\left[ \left(x_i-\mathbb{E}[x_i]\right)\left(x_i-\mathbb{E}[x_i]\right)^T\right]
       \f]
       Note that it is only possible to compute the cross covariance of \f$x_i\f$ with \f$x_j\f$ by setting `blockInd=-1` and 
       computing the entire covariance matrix.   

       @param[in] blockInd (Optional) The block of the random variable $x$ to use in the expectation.  By default, blockInd=-1
                      and the expectation is computed with respect to the entire random variable $x$.
       @return A matrix containing an estimate of \f$\text{Cov}[x]\f$ or \f$\text{Cov}[x_i]\f$.
      */
      virtual Eigen::MatrixXd Covariance(int blockInd=-1) const;

      /**
       Computes the sample covariance using a precomputed (or known) mean.  
       
       If blockInd is non-negative, only the mean of one block of the samples is computed.
      */
      virtual Eigen::MatrixXd Covariance(Eigen::VectorXd const& mean, 
                                         int                    blockInd=-1) const = 0;

      /**
      Using samples of \f$x\f$ stored in this sample collection, this function
      approximates the expected value of \f$f(x)\f$ for some function \f$f\f$ defined
      as a muq::Modeling::ModPiece.  The output is a vector containing the expected
      value of \f$f\f$.
      */
      virtual Eigen::VectorXd ExpectedValue(std::shared_ptr<muq::Modeling::ModPiece> const& f,
                                            std::vector<std::string> const& metains = std::vector<std::string>()) const = 0;


    /**
    Returns an estimate of the the Monte Carlo standard error (MCSE) \f$\hat{\sigma}\f$ for a Monte Carlo estimate of the mean derived using this SampleEstimator.
    Recall at the MCSE is the standard deviation of the estimator variance employed in the Central Limit Theorem.

    @param[in] blockInd Specifies the block of the sampling state we're interested in.  Defaults to -1, which will result in all blocks of the sampling state being concatenated in the MCSE estimate.
    @param[in] method A string describing what method should be used to estimate the MCSE.  Defaults to "Batch"
    @return A vector containing either the MCSE for each component (if method!="MultiBatch") or a single component vector containing the square root of the generalized estimator variance (if method=="MultiBatch").
    */
    virtual Eigen::VectorXd StandardError(int                blockInd, 
                                          std::string const& method) const = 0;

    virtual Eigen::VectorXd StandardError(std::string const& method="Batch") const{return StandardError(-1,method);};
    virtual Eigen::VectorXd StandardError(int blockDim) const{return StandardError(blockDim,"Batch");};
    
    /**
    Returns an estimate of the effective sample size (ESS), which is the number of independent samples of the target 
    distribution that would be needed to obtain the same statistical accuracy as this estimator.  For independent samples,
    the estimator variance (squared MCSE) \f$\hat{\sigma}^2\f=\sigma^2/ N\f$.   Given the estimator variance \f$\hat{\sigma}^2\f$,
    the effective sample size is then given by the ratio of the target distribution variance and the estimator variance:
    \f[
        \text{ESS} = \frac{\sigma^2}{\hat{\sigma}^2}.
    \f]
   
    @param[in] blockInd Specifies the block of the sampling state we're interested in.  Defaults to -1, which will result in all blocks of the sampling state being concatenated in the MCSE estimate.
    @param[in] method A string describing what method should be used to estimate the MCSE.  Defaults to "Batch".   Other options include "MultiBatch" or "Wolff".   For details, see the SampleCollection class.
    @return A vector containing either the MCSE for each component (if method!="MultiBatch") or a single component vector containing the square root of the generalized estimator variance (if method=="MultiBatch").
    */
    virtual Eigen::VectorXd ESS(int                blockInd, 
                                std::string const& method) const  = 0;
    virtual Eigen::VectorXd ESS(int blockDim) const {return ESS(blockDim,"Batch");};
    virtual Eigen::VectorXd ESS(std::string const& method="Batch") const {return ESS(-1,method);};

  }; // class SampleEstimator

}
}
#endif