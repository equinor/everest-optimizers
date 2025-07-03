#ifndef MarkovChain_H
#define MarkovChain_H

#include "MUQ/SamplingAlgorithms/SampleCollection.h"

namespace muq {
  namespace SamplingAlgorithms{

    /**
    @ingroup MCMC
    @class MarkovChain
    @brief A class for storing and working with the results of Markov chain Monte Carlo algorithms.
    @details The MarkovChain class is a child of SampleCollection where the sample
    weights correspond to the number of consecutive steps taking the same value,
    and the weights are unnormalized (i.e., do not sum to one).  This is a useful
    class for storing the chain produced by an MCMC algorithm without storing the
    duplicate points that result from rejected proposals.
    */
    class MarkovChain : public SampleCollection
    {
    public:

      MarkovChain() = default;

      virtual ~MarkovChain() = default;

      /** Computes the effective sample size of the Markov chain.  

        If method=="Wolff", the spectral method described in
            "Monte Carlo errors with less error" by Ulli Wolff is employed.
            This returns an ESS for each component of the chain.

        If method=="Batch" (default) The overlapping batch method (OBM) described in \cite Flegal2010 
            is used.  This method is also applied to each component independently,
            resulting in an ESS estimate for each component.

        If method=="MultiBatch",  The multivariate method of \cite Vats2019 is employed.  This 
            method takes into account the joint correlation of all components of the chain and 
            returns a single ESS.   This approach is preferred in high dimensional settings.
      */
      virtual Eigen::VectorXd ESS(std::string const& method="Batch") const override{return ESS(-1,method);};
      virtual Eigen::VectorXd ESS(int blockDim) const override{return ESS(blockDim,"");};
      virtual Eigen::VectorXd ESS(int blockDim, std::string const& method) const override;

      /** Computes the Monte Carlo standard error (MCSE) of the Markov chain.  

        If method=="Wolff", the spectral method described in
            "Monte Carlo errors with less error" by Ulli Wolff \cite Wolff2004 is employed.
            This returns an MCSE for each component of the chain.

        If method=="Batch" (default) The overlapping batch method (OBM) described in \cite Flegal2010 
            is used.  This method is also applied to each component independently,
            resulting in an MCSE estimate for each component.

        If method=="MultiBatch",  The multivariate method of \cite Vats2019 is employed.  This 
            method takes into account the joint correlation of all components of the chain and 
            returns a single generalized MCSE.   This approach is preferred in high dimensional settings.
      */
      virtual Eigen::VectorXd StandardError(std::string const& method="Batch") const override{return StandardError(-1,method);};
      virtual Eigen::VectorXd StandardError(int blockDim) const override{return StandardError(blockDim,"Batch");};
      virtual Eigen::VectorXd StandardError(int blockDim, std::string const& method) const override;
      
      /** Computes the effective sample size using the spectral method of \cite Wolff2004 

        @param[in] blockInd Specifies the block of the sampling state we're interested in.  Defaults to -1, which will result in all blocks of the sampling state being concatenated in the ESS estimate.
        @return A vector containing the ESS for each component of the chain.
      */
      Eigen::VectorXd WolffESS(int blockDim) const;

      /** Computes the MCSE based on the effective sample size returned by WolffESS

        @param[in] blockInd Specifies the block of the sampling state we're interested in.  Defaults to -1, which will result in all blocks of the sampling state being concatenated in the MCSE estimate.
        @return A vector containing the MCSE for each component of the chain.
      */
      Eigen::VectorXd WolffError(int blockDim) const;

      /** Computes the effective sample size given a vector containing a single component of the Markov chain. */
      static double SingleComponentWolffESS(Eigen::Ref<const Eigen::VectorXd> const& trace);

      virtual std::shared_ptr<SampleCollection> segment(unsigned int startInd, unsigned int length, unsigned int skipBy=1) const override;

    private:


      std::vector<std::unordered_map<std::string, boost::any> > meta;

    }; // class MarkovChain
  }
}

#endif
