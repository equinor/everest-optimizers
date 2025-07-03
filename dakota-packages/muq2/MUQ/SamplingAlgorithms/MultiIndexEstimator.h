#ifndef MULTIINDEXESTIMATOR_H
#define MULTIINDEXESTIMATOR_H

#include "MUQ/SamplingAlgorithms/SampleEstimator.h"
#include "MUQ/SamplingAlgorithms/MIMCMCBox.h"
#include "MUQ/SamplingAlgorithms/MarkovChain.h"

namespace muq{
namespace SamplingAlgorithms{

    /** @class MultiIndexEstimator
        @ingroup MCMC
        @brief Class for estimating expectations using multi-index samples from MC or MCMC.
    */
    class MultiIndexEstimator : public SampleEstimator
    {
    public:

        /** Construct the multiindex estimator using MIMCMC boxes.  These boxes are typically constructed by 
            a MIMCMC methods such as the GreedyMLMCMC or MIMCMC classes.

            @param[in] boxesIn "Boxes" holding the differences between chains at different indices
            @param[in] useQoisIn (optional) Whether this estimator should use the QOIs in the chains or
                               the parameters themselves.  Defaults to false, which implies the parameters
                               will be used in the estimates.
         */ 
        MultiIndexEstimator(std::vector<std::shared_ptr<MIMCMCBox>> const& boxesIn, 
                            bool                                           useQoisIn = false);

        virtual ~MultiIndexEstimator() = default;

        virtual unsigned int BlockSize(int blockInd) const override;

        virtual unsigned int NumBlocks() const override;

        virtual Eigen::VectorXd ExpectedValue(std::shared_ptr<muq::Modeling::ModPiece> const& f,
                                              std::vector<std::string> const& metains = std::vector<std::string>()) const override;

        virtual Eigen::MatrixXd Covariance(int blockInd=-1) const override { return SampleEstimator::Covariance(blockInd);};
        virtual Eigen::MatrixXd Covariance(Eigen::VectorXd const& mean, 
                                           int                    blockInd=-1) const override;


        
        /** Computes the standard deviation mean value returned by theis MultiIndexEstimator.  The estimator variances for each term in the 
            telescoping series are summed and the square root of this quantity is returned.  This process assumes that the terms in the 
            series are independent.   The value of Method is passed on to the underlying SampleCollection classes to compute the 
            variance of each term.   Valid options are `Batch`, `MultiBatch`, and `Wolff`.   

            @param[in] blockDim Specifies the block that we wish to use in the MCSE estimator.  Defaults to -1, which results in all blocks of the chain being concatenated in the MCSE estimate.
            @param[in] method Specifies the type of MCSE estimator used by the single level chains to estimate the variance of each term in the multiindex telescoping series.   See MarkovChain::StandardError for more details.
            @return If `method=="MultiBatch"`, this function returns length 1 vector containing a single multivariate MCSE estimate.  If `method!="MultiBatch"`, this function returns the MCSE for each component of the chain.
        */  
        virtual Eigen::VectorXd StandardError(int blockDim, std::string const& method) const override;
        virtual Eigen::VectorXd StandardError(std::string const& method="Batch") const override{return StandardError(-1,method);};
        virtual Eigen::VectorXd StandardError(int blockDim) const override{return StandardError(blockDim,"Batch");};
        
        /** This function returns an estimate of the Effective Sample Size (ESS) of this estimator.  The ESS here refers to the number of 
            indepedent samples that would be required in a classic single-level Monte Carlo estimate to achieve the same statistical accuracy
            as this multi-index estimator.    

            This function computes the ESS by computing the ratio of the sample variance with the squared MCSE.   The MCSE is computed by 
            the MultiIndex::StandardError function.    

            @param[in] method Specifies the type of MCSE estimator used by the single level chains to estimate the variance of each term in the multiindex telescoping series.   See MarkovChain::ESS for more details.
            @return If `method=="MultiBatch"`, this function returns length 1 vector containing a single multivariate effective sample size estimate.  If `method!="MultiBatch"`, this function returns an ESS for each component of the chain.
        */
        virtual Eigen::VectorXd ESS(std::string const& method="Batch") const override{return ESS(-1,method);};
        virtual Eigen::VectorXd ESS(int blockDim) const override{return ESS(blockDim,"Batch");};
        virtual Eigen::VectorXd ESS(int blockDim, std::string const& method) const override;
        
    private:
        
        /** Creates a Markov chain for each term in the telescoping sum.
            @param[in] blockInd (optional) The index of the block we're interested in.  If -1, then all blocks in the state are concatenated.    This is not used if the f argument is specified.
            @param[in] f (optional), A ModPiece that evaluates a quantity of interest if we're interested in chains over the QOI difference.
            @return A vector of Markov chains for each term in the series. 
        */
        std::vector<std::shared_ptr<MarkovChain>> GetDiffChains(int blockInd=-1,
                                                                std::shared_ptr<muq::Modeling::ModPiece> const& f=nullptr) const;



        const Eigen::VectorXi blockSizes;
        const bool useQois;

        std::vector<std::shared_ptr<MIMCMCBox>> boxes;
        std::vector<std::shared_ptr<MarkovChain>> diffChains;
        
    };
}
}



#endif 