#ifndef SAMPLECOLLECTION_H
#define SAMPLECOLLECTION_H

#include <memory>
#include <vector>
#include <set>

#include <Eigen/Core>

#include "MUQ/Modeling/ModPiece.h"

#include "MUQ/SamplingAlgorithms/SamplingState.h"
#include "MUQ/SamplingAlgorithms/SampleEstimator.h"

#include "MUQ/Utilities/HDF5/HDF5File.h"

#include <boost/property_tree/ptree.hpp>

namespace muq{
  namespace SamplingAlgorithms{

    class SamplingStateIdentity{
    public:
      SamplingStateIdentity(int blockIndIn) : blockInd(blockIndIn){};

      virtual ~SamplingStateIdentity() = default;

      Eigen::VectorXd const& operator()(SamplingState const& a);

      const int blockInd;

    private:
      Eigen::VectorXd output;
    };

    class ExpectedModPieceValue {
    public:
      ExpectedModPieceValue(std::shared_ptr<muq::Modeling::ModPiece> const& f, std::vector<std::string> const& metains);

      virtual ~ExpectedModPieceValue() = default;

      Eigen::VectorXd const& operator()(SamplingState const& a);

    private:
      std::shared_ptr<muq::Modeling::ModPiece> f;

      const std::vector<std::string> metains;
    };

    class SamplingStatePartialMoment{
    public:
      SamplingStatePartialMoment(int                    blockIndIn,
                                 int                    momentPowerIn,
                                 Eigen::VectorXd const& muIn) : blockInd(blockIndIn), momentPower(momentPowerIn), mu(muIn){};

      virtual ~SamplingStatePartialMoment() = default;

      Eigen::VectorXd const& operator()(SamplingState const& a);

      const int blockInd;
      const int momentPower;
      const Eigen::VectorXd& mu;

    private:
      Eigen::VectorXd output;
    };

    /** @class SampleCollection
        @brief A class to hold and analyze a collection of SamplingState objects
    */
    class SampleCollection : public SampleEstimator, public std::enable_shared_from_this<SampleCollection>{
    public:
      SampleCollection() = default;

      virtual ~SampleCollection() = default;

      virtual void Add(std::shared_ptr<SamplingState> newSamp);

      virtual std::shared_ptr<SamplingState> at(unsigned i);
      virtual const std::shared_ptr<SamplingState> at(unsigned i) const;

      /** Returns the number of samples in this SampleCollection. */
      virtual unsigned int size() const;

      /** Returns a new sample collection with the first \f$N\f$ states of this collection.
          Equivalent to a python command like list[0:N]
      */
      virtual std::shared_ptr<SampleCollection> head(unsigned int N) const{return segment(0,N,1);}

      /** Returns a new sample collection with the last \f$N\f$ states of this collection.
          Equivalent to a python command like list[-N:]
      */
      virtual std::shared_ptr<SampleCollection> tail(unsigned int N) const{return segment(size()-N,N,1);}


      /** Returns a new sample collection containing a segment of this sample collection.
          The
          Equivalent to a python command like list[startInd:startInd+length:skipBy]
      */
      virtual std::shared_ptr<SampleCollection> segment(unsigned int startInd, unsigned int length, unsigned int skipBy=1) const;

      virtual const std::shared_ptr<SamplingState> back() const;

      ///  Computes the componentwise central moments (e.g., variance, skewness, kurtosis, etc..) of a specific order given that we already know the mean
      virtual Eigen::VectorXd CentralMoment(unsigned               order, 
                                            Eigen::VectorXd const& mean,
                                            int                    blockNum=-1) const override;
      virtual Eigen::VectorXd CentralMoment(unsigned int order, 
                                            int          blockNum=-1) const override{ return CentralMoment(order, Mean(blockNum), blockNum);};

      virtual Eigen::VectorXd Mean(int blockInd=-1) const override;
      virtual Eigen::MatrixXd Covariance(int blockInd=-1) const override{return SampleEstimator::Covariance(blockInd);};
      virtual Eigen::MatrixXd Covariance(Eigen::VectorXd const& mean, int blockInd=-1) const override;

      /** Computes running estimates of the covariance.

          This function returns a vector of matrices where index \f$n\f$ of the
          vector contains a Monte Carlo estimate of the covariance that is constructed
          using the first \f$n\f$ samples in this SampleCollection.
      */
      virtual std::vector<Eigen::MatrixXd> RunningCovariance(int blockInd=-1) const;
      virtual std::vector<Eigen::MatrixXd> RunningCovariance(Eigen::VectorXd const& mean, int blockInd=-1) const;

      /** @brief Returns the effective sample size of the samples
          @details For almost all random variables of interest, the central limit
                   theorem states that the variance of a Monte Carlo estimator
                   \f[\hat{\mu}_N = \frac{1}{N}\sum_{i=1}^N \theta^{(i)},\f]
                   satisfies
                   \f[ \sqrt{N}\mathbb{V}\left[\hat{\mu}_N-\mu\right] \rightarrow N(0,\sigma^2),\f]
                   as the number of samples \f$N\f$ increases.  Here, \f$mu\f$ is
                   the true mean of the random variable, \f$\sigma^2\f$ is the true
                   variance of the random variable.  This assumes that each \f$\theta^{(i)}\f$
                   is an independent sample with unity weight.

                   When the samples are not independent or the weights are not unity,
                   the variance of the estimator will generally be larger.   The
                   "effective sample size" (ESS) describes how many independent
                   samples would have been needed to obtain the same estimator
                   variance.  In particular, let \f$\tilde{\mu}_N\f$ be a Monte
                   Carlo estimator based on \f$N\f$ correlated samples.  

        If method=="Batch" (default) The overlapping batch method (OBM) described in \cite Flegal2010 
            is used.  This method is also applied to each component independently,
            resulting in an ESS estimate for each component.

        If method=="MultiBatch",  The multivariate method of \cite Vats2019 is employed.  This 
            method takes into account the joint correlation of all components of the chain and 
            returns a single ESS.   This approach is preferred in high dimensional settings.

        @param[in] method A string specifying the method used to estimate the ESS.  Options in SampleCollection are "Batch" or "MultiBatch".   The MarkovChain class also supports a "Wolff" method.
        @return Either a vector of length \f$D\f$ containing an ESS estimate for each component or a vector of length 1 containing a single multivariate ESS estimate.
      */
      virtual Eigen::VectorXd ESS(std::string const& method="Batch") const override{return ESS(-1,method);};

      /**
      @param[in] blockDim The block of the SampleState that we want to use in the ESS computation.
      @return Either a vector of length \f$D\f$ containing an ESS estimate for each component or a vector of length 1 containing a single multivariate ESS estimate.
      */
      virtual Eigen::VectorXd ESS(int blockDim) const override{return ESS(blockDim,"Batch");};

      /**
      @param[in] blockDim The block of the SampleState that we want to use in the ESS computation.
      @param[in] method A string specifying the method used to estimate the ESS.  Options in SampleCollection are "Batch" or "MultiBatch".   The MarkovChain class also supports a "Wolff" method.
      @return Either a vector of length \f$D\f$ containing an ESS estimate for each component or a vector of length 1 containing a single multivariate ESS estimate.
      */
      virtual Eigen::VectorXd ESS(int blockDim, std::string const& method) const override;

      /**
      Computes the effective sample size using the overlapping or nonoverlapping 
      batch method.  (See e.g., \cite Flegal2010).
      
      @param[in] blockInd The block of the SampleState that we want to use in the ESS computation.
      @param[in] batchSize The size of the batches \f$b_n\f$ to use.  Defaults to \f$N^{1/3}\f$.  Note that to ensure the convergence of the ESS estimate as \f$N\rightarrow\infty\f$, the batch size must increase with \f$N\f$.
      @param[in] overlap How much the batches overlap.   Defaults to \f$b_n/2\f$, where \f$b_n\f$ is the batch size.  A nonoverlapping batch estimate will be use is overlap=0.  This value bust be smaller than \f$b_n\f$.
      @return A vector of length \f$D\f$ containing an ESS estimate for each component
      */
      Eigen::VectorXd BatchESS(int blockInd=-1, int batchSize=-1, int overlap=-1) const;

      /**
        Computes the multivariate effective sample size of \cite Vats2019.  This implementation is 
        adapted from the python implementation at https://github.com/Gabriel-p/multiESS .

        The standard definition of a univariate effective sample size is given in terms of the 
        the ratio between the target distribution variance and the estimator variance.  More 
        precisely, \f$\text{ESS}_u\f$ is given by
        \f[
          \text{ESS}_u = N \frac{\sigma^2}{\hat{\sigma}^2},
        \f]
        where \f$N\f$ is the total number of samples in the Monte Carlo estimate,
        \f$\sigma^2\f$ is the variance of the target random variable, and \f$\hat{\sigma}^2\f$ 
        is the variance of a Monte Carlo estimator for the mean of the target distribution.

        The multivariate ESS is defined similarly, but using generalized variances.  Let \f$\Sigma\f$ 
        denote the target distribution covariance and \f$\hat{\Sigma}\f$ the covariance 
        of the Monte Carlo mean estimator.    The multivariate ESS of \cite Vats2019 is then given by
        \f[
          \text{ESS}_m = N \left( \frac{|\Sigma|}{|\hat{\Sigma}|} \right)^{1/D},
        \f]
        where \f$|\cdot|\f$ denotes the determinant of the matrix and \f$D\f$ is the dimension of 
        target random variable.

        As proposed in \cite Vats2019, it is possible to estimate \f$\text{ESS}_m\f$ using overlapping 
        batches of samples to calculate the estimator covariance \f$\hat{\Sigma}\f$.

        @param[in] blockInd The block of the SampleState that we want to use in the ESS computation.
        @param[in] batchSize The size of the batches \f$b_n\f$ to use.  Defaults to \f$N^{1/3}\f$.  Note that to ensure the convergence of the ESS estimate as \f$N\rightarrow\infty\f$, the batch size must increase with \f$N\f$.
        @param[in] overlap How much the batches overlap.   Defaults to \f$b_n/2\f$, where \f$b_n\f$ is the batch size.  A nonoverlapping batch estimate will be use is overlap=0.  This value bust be smaller than \f$b_n\f$.
        @return A double containing the multivariate ESS estimate.
      */
      double MultiBatchESS(int blockInd=-1, int batchSize=-1, int overlap=-1) const;

      /**
        Returns an estimate of the the Monte Carlo standard error (MCSE) \f$\hat{\sigma}\f$ for a Monte Carlo estimate of the mean derived using this SampleEstimator.
        Recall at the MCSE is the standard deviation of the estimator variance employed in the Central Limit Theorem.

        @param[in] blockInd Specifies the block of the sampling state we're interested in.  Defaults to -1, which will result in all blocks of the sampling state being concatenated in the MCSE estimate.
        @param[in] method A string describing what method should be used to estimate the MCSE.  Defaults to "Batch".  Options are the same as the ESS function: "Batch", "MultiBatch", and "Wolff".  The "Wolff" option is only valid for the "MarkovChain" and "MultiIndexEstimator" classes.
        @return A vector containing either the MCSE for each component.
      */
      virtual Eigen::VectorXd StandardError(int                blockInd, 
                                            std::string const& method) const override;
      virtual Eigen::VectorXd StandardError(int blockInd) const override{return StandardError(blockInd,"Batch");};
      virtual Eigen::VectorXd StandardError(std::string const& method="Batch") const override{return StandardError(-1,method);};

      /**
        Estimates the Monte Carlo standard error (square root of estimator variance) using overlapping batch means (OBM).   For details, consult  \cite Flegal2010 .
      
        @param[in] blockInd Specifies the block of the sampling state we're interested in.  Defaults to -1, which will result in all blocks of the sampling state being concatenated in the MCSE estimate.
        @param[in] batchSize The size of the batches \f$b_n\f$ to use.  Defaults to \f$\sqrt{N}\f$.  Note that to ensure the convergence of the MCSE estimate as \f$N\rightarrow\infty\f$, the batch size must increase with \f$N\f$.
        @param[in] overlap How much the batches overlap.   Defaults to \f$0.75*b_n\f$, where \f$b_n\f$ is the batch size.  A nonoverlapping batch estimate will be use is overlap=0.  This value bust be smaller than \f$b_n\f$.
        @return A vector containing either the MCSE for each component of the chain.
      */
      virtual Eigen::VectorXd BatchError(int blockInd=-1, int batchSize=-1, int overlap=-1) const;

      /**
        Estimates the Monte Carlo standard error using the effective sample size coming from the multivariate overlapping batch method of \cite Vats2019 .   
      
        @param[in] blockInd Specifies the block of the sampling state we're interested in.  Defaults to -1, which will result in all blocks of the sampling state being concatenated in the MCSE estimate.
        @param[in] batchSize The size of the batches \f$b_n\f$ to use.  Defaults to \f$N^{1/2}\f$.  Note that to ensure the convergence of the MCSE estimate as \f$N\rightarrow\infty\f$, the batch size must increase with \f$N\f$.
        @param[in] overlap How much the batches overlap.   Defaults to \f$0.75*b_n\f$, where \f$b_n\f$ is the batch size.  A nonoverlapping batch estimate will be use is overlap=0.  This value bust be smaller than \f$b_n\f$.
        @return A  vector containing \f$\sqrt{\sigma_i / N_{ess}}\f$, where \f$N_{ess}\f$ is the multivariate ESS estimate return by MultiBatchESS.
      */
      virtual Eigen::VectorXd MultiBatchError(int blockInd=-1, int batchSize=-1, int overlap=-1) const;

      /** Returns the samples in this collection as a matrix.  Each column of the
          matrix will correspond to a single state in the chain.
      */
      virtual Eigen::MatrixXd AsMatrix(int blockDim=-1) const;

      /** Constructs a SampleCollection from a matrix of samples.  Each column of the matrix is a sample.
      */
      static std::shared_ptr<SampleCollection> FromMatrix(Eigen::MatrixXd const& samps);

      /** Returns a vector of unnormalized sample weights for computing empirical expectations.
          If the samples were generated with MC or MCMC, the weights will all be
          1.  Importance sampling will generate non-unit weights.   Note that these
          weights are unnormalized and will not generally sum to one.
      */
      virtual Eigen::VectorXd Weights() const;

      /**
      Writes the samples, weights, and sample metadata to a group in an HDFfile.
      - The samples themselves will be written as a \f$DxN\f$ matrix to the "/samples" dataset.
      - The sample weights will be written as a length \f$1xN\f$ row vector to the "/weights" dataset.
      - Each additional metadata field (e.g., logdensity) will be written to a \f$KxN\f$ matrix, where \f$K\f$ is the dimension of the vector-valued metadata and \f$N|f$ is the number of samples.
	     @param[in] filename The name of the HDF file to write to
	     @param[in] group The name of the group within the HDF file.  Defaults to the root directory "/".
       */
      virtual void WriteToFile(std::string const& filename, std::string const& group = "/") const;

      /**
      @param[in] name Need the this piece of meta data for each sample
      \return A matrix of meta data associated with in the input string
      */
      Eigen::MatrixXd GetMeta(std::string const& name) const;

      /**
      Returns a vector containing the names of metadata that is available.  If
      requireAll is true, then the returned list contains metadata that is available
      for all samples.  If requireAll is false, then this list contains metadata
      that is stored with any sample even if not all samples have that metadata.
      */
      std::set<std::string> ListMeta(bool requireAll=true) const;

      /**
      Using samples of \f$x\f$ stored in this sample collection, this function
      computes the expected value of \f$f(x)\f$ for some function \f$f\f$ defined
      as a muq::Modeling::ModPiece.  The output is a vector containing the expected
      value of \f$f\f$.
      */
      virtual Eigen::VectorXd ExpectedValue(std::shared_ptr<muq::Modeling::ModPiece> const& f, 
                                            std::vector<std::string> const& metains = std::vector<std::string>()) const override;

      /**
      Computes running estimates of the expected value.  Returns a std::vector
      of Eigen::VectorXd instances.   Index \f$n\f$ of the std::vector contains
      a Monte Carlo  estimate of the expected value of \f$f\f$ using the first
      \f$n\f$ samples of the sample collection.
      */
      std::vector<Eigen::VectorXd> RunningExpectedValue(std::shared_ptr<muq::Modeling::ModPiece> const& f, std::vector<std::string> const& metains = std::vector<std::string>()) const;


      /** Returns the size \f$N_i\f$ of each block.   If `blockInd==-1`, the size \f$N\f$ of the joint random variable is 
          returned.
      */ 
      virtual unsigned int BlockSize(int blockInd) const override;

      /** Returns the nubmer of block \f$M\f$. */
      virtual unsigned int NumBlocks() const override;

      /**
      To help assess the convergence of a MCMC chain, this function returns either an estimate of the standard scale reduction \f$\hat{R}\f$ diagnostic from \cite Gelman2013,
      one of the modifications presented in \cite Vehtari2021, the multivariate potential scale reduction factor
      (MPSRF) of \cite Brooks1998, or multivariate adapations of MPSRF that are similar to the split and ranked
      methods of \cite Vehtari2021.   

      These scale reduction factors are typically computed using multiple chains.  This function instead splits 
      a single chain into multiple segments and then treats each segment as chain in the usual \f$\hat{R}\f$ estimators.
      This approach is similar to the split techniques of \cite Vehtari2021.

      For more details on the definition and computation of \f$\hat{R}\f$, see Diagnostics::Rhat.

      Parameter Key | Type | Default Value | Description |
      ------------- | ------------- | ------------- | ------------- |
      "Transform"   | boolean | False  | If the parameters should be rank-transformed before computing Rhat, as in \cite Vehtari2021. |
      "Multivariate" | boolean | False | If the MPSRF value should be returned instead of the componentwise \$\hat{R}\f$ statistic. If `true`, the output vector will have a single component.  The MPSRF serves as a worse case estimate of \f$\hat{R}\f$ over all linear combinations of the parameters. |

      @param[in] blockDim  Specifies the block of the sampling state we're interested in. If blockDim=-1, then all blocks will be concatenated together.
      @param[in] options (optional) A property tree possibly containing settings for the "Multivariate" or "Transform" parameters listed above.
      @returns If Multivariate==False, a vector of \f$\hat{R}\f$ values for each component of the parameters.  If Multivariate==true, then a length 1 vector containing the MPSRF.

      */
      virtual Eigen::VectorXd Rhat(int                         blockDim, 
                                   unsigned int                numSegments=4, 
                                   boost::property_tree::ptree options = boost::property_tree::ptree()) const;

      /** Same as the other Rhat function, except all blocks are concatenated together. */
      virtual Eigen::VectorXd Rhat(unsigned int                numSegments=4, 
                                   boost::property_tree::ptree options = boost::property_tree::ptree()) const{return Rhat(-1,numSegments,options);};
                                  
    protected:

      std::vector<std::shared_ptr<SamplingState>> samples;

      /** Returns the sum of the weights and the sum of the squared weights. */
      static std::pair<double,double> RecursiveWeightSum(std::vector<std::shared_ptr<SamplingState>>::const_iterator start,
                                                         std::vector<std::shared_ptr<SamplingState>>::const_iterator end);


      template<typename FuncType>
      static std::pair<double,Eigen::VectorXd> RecursiveSum(std::vector<std::shared_ptr<SamplingState>>::const_iterator                         start,
                                                            std::vector<std::shared_ptr<SamplingState>>::const_iterator                         end,
                                                            FuncType& f)
      {
        int numSamps = std::distance(start,end);
        const int maxSamps = 20;

        // If the number of samples is small enough, we can safely add them up directly
        if(numSamps<maxSamps){

          Eigen::VectorXd sum = (*start)->weight * f(**start);
          double weightSum = (*start)->weight;

          for(auto it=start+1; it!=end; ++it){
              sum += (*it)->weight * f(**it);
              weightSum += (*it)->weight;
          }
          return std::make_pair(weightSum, sum);

        // Otherwise, it's more numerically stable to add things pairwise
        }else{
          int halfDist = std::floor(0.5*numSamps);
          double weight1, weight2;
          Eigen::VectorXd sum1, sum2;
          std::tie(weight1,sum1) = RecursiveSum(start, start+halfDist, f);
          std::tie(weight2,sum2) = RecursiveSum(start+halfDist, end, f);

          return std::make_pair(weight1+weight2, (sum1+sum2).eval());
        }
      }

      /**
	 \return A map from meta data name to a matrix where each column corresponds to a sample
       */
      std::unordered_map<std::string, Eigen::MatrixXd> GetMeta() const;

    private:

      /**
	 @param[in] hdf5file The hdf5 file where the data will be written
	 @param[in] dataname The name of the data set we (may) want to create
	 @param[in] dataSize The number of rows (the size of the data in one sample)
	 @param[in] totSamps The total number of samples we need to write to the file (max. number of samples)
	 \return true: the data set exists and is the right size, false: the data set does not exist or is the wrong size
       */
      bool CreateDataset(std::shared_ptr<muq::Utilities::HDF5File> hdf5file, std::string const& dataname, int const dataSize, int const totSamps) const;
    };
  }
}

#endif // SAMPLECOLLECTION_H
