#include "MUQ/SamplingAlgorithms/MultiIndexEstimator.h"
#include "MUQ/SamplingAlgorithms/MarkovChain.h"

using namespace muq::SamplingAlgorithms;


MultiIndexEstimator::MultiIndexEstimator(std::vector<std::shared_ptr<MIMCMCBox>> const& boxesIn,
                                         bool                                           useQoisIn) : boxes(boxesIn),
                                                                                                    blockSizes(useQoisIn ? boxesIn.at(0)->GetFinestProblem()->blockSizesQOI : boxesIn.at(0)->GetFinestProblem()->blockSizes),
                                                                                                    useQois(useQoisIn)
{
    
}

unsigned int MultiIndexEstimator::BlockSize(int blockInd) const
{   
    if(blockInd<0){
        return blockSizes.sum();
    }else{
        return blockSizes(blockInd);
    }
}


unsigned int MultiIndexEstimator::NumBlocks() const
{
    return blockSizes.size();
}

Eigen::VectorXd MultiIndexEstimator::StandardError(int blockDim, std::string const& method) const
{   
    // Construct Markov chain objects for each term in the telescoping series
    std::vector<std::shared_ptr<MarkovChain>> chains = GetDiffChains();

    Eigen::VectorXd estVar = chains.at(0)->StandardError(blockDim, method).array().square();
    for(unsigned int i=1; i<chains.size(); ++i)
        estVar += chains.at(i)->StandardError(blockDim, method).array().square().matrix();
    
    return estVar.array().sqrt();
}


Eigen::VectorXd MultiIndexEstimator::ESS(int blockDim, std::string const& method) const
{   
    return Variance(blockDim).array() / StandardError(blockDim, method).array().square();
}



std::vector<std::shared_ptr<MarkovChain>> MultiIndexEstimator::GetDiffChains(int blockInd, std::shared_ptr<muq::Modeling::ModPiece> const& f) const
{
    // Chains to hold the summand for each term in the multi-index telescoping series
    std::vector<std::shared_ptr<MarkovChain>> diffChains(boxes.size());

    unsigned int dim;
    if(useQois){
        dim = boxes.at(0)->FinestChain()->GetQOIs()->BlockSize(blockInd);
    }else{
        dim = boxes.at(0)->FinestChain()->GetSamples()->BlockSize(blockInd);
    }

    // Extract information from each box
    for(unsigned int boxInd = 0; boxInd<boxes.size(); ++boxInd){

        diffChains.at(boxInd) = std::make_shared<MarkovChain>();

        auto& box = boxes.at(boxInd);
        auto boxIndices = box->GetBoxIndices();

        unsigned int numSamps;
        if(useQois){
            numSamps = box->FinestChain()->GetQOIs()->size();
        }else{
            numSamps = box->FinestChain()->GetSamples()->size();
        }

        Eigen::VectorXd diff;
        for(unsigned int sampInd =0; sampInd < numSamps; ++sampInd)
        {
            // Compute the difference for one term and one sample in the telescoping series
            diff = Eigen::VectorXd::Zero(dim);

            for (int i = 0; i < boxIndices->Size(); i++) {

                std::shared_ptr<MultiIndex> boxIndex = (*boxIndices)[i];
                auto chain = box->GetChain(boxIndex);
                std::shared_ptr<MarkovChain> samps;
                if(useQois){
                    samps = chain->GetQOIs();
                }else{
                    samps = chain->GetSamples();
                } 

                MultiIndex index = *(box->GetLowestIndex()) + *boxIndex;
                MultiIndex indexDiffFromTop = *(box->GetHighestIndex()) - index;

                double mult = (indexDiffFromTop.Sum() % 2 == 0) ? 1.0 : -1.0;
                
                if(f){
                    diff += mult * f->Evaluate(samps->at(sampInd)->state).at(0);
                }else{
                    diff += mult * samps->at(sampInd)->ToVector(blockInd);
                }
                 
            }

            diffChains.at(boxInd)->Add(std::make_shared<SamplingState>(diff));
        }
    }

    return diffChains;
}


Eigen::VectorXd MultiIndexEstimator::ExpectedValue(std::shared_ptr<muq::Modeling::ModPiece> const& f,
                                                   std::vector<std::string>                 const& metains) const
{
    assert(f->outputSizes.size()==1);

    Eigen::VectorXd telescopingSum = Eigen::VectorXd::Zero(f->outputSizes(0));

    // Add up the telescoping series of MI boxes
    for (auto& box : boxes) {

        // Compute the expected difference for one term in the telescoping series
        Eigen::VectorXd diffMean = Eigen::VectorXd::Zero(f->outputSizes(0));

        auto boxIndices = box->GetBoxIndices();
        for (int i = 0; i < boxIndices->Size(); i++) {

            std::shared_ptr<MultiIndex> boxIndex = (*boxIndices)[i];
            auto chain = box->GetChain(boxIndex);
            std::shared_ptr<MarkovChain> samps;
            if(useQois){
                samps = chain->GetQOIs();
            }else{
                samps = chain->GetSamples();
            } 

            MultiIndex index = *(box->GetLowestIndex()) + *boxIndex;
            MultiIndex indexDiffFromTop = *(box->GetHighestIndex()) - index;

            if (indexDiffFromTop.Sum() % 2 == 0) {
                diffMean += samps->ExpectedValue(f,metains);
            } else {
                diffMean -= samps->ExpectedValue(f,metains);
            }
        }

        // Add this term of the series to the running total      
        telescopingSum += diffMean;
    }

    return telescopingSum;
} 

Eigen::MatrixXd MultiIndexEstimator::Covariance(Eigen::VectorXd const& mean, 
                                                int                    blockInd) const
{
    Eigen::MatrixXd telescopingSum = Eigen::MatrixXd::Zero(BlockSize(blockInd),BlockSize(blockInd));

    // Add up the telescoping series of MI boxes
    for (auto& box : boxes) {

        // Compute the expected difference for one term in the telescoping series
        Eigen::MatrixXd diffMean = Eigen::MatrixXd::Zero(BlockSize(blockInd),BlockSize(blockInd));

        auto boxIndices = box->GetBoxIndices();
        for (int i = 0; i < boxIndices->Size(); i++) {

            std::shared_ptr<MultiIndex> boxIndex = (*boxIndices)[i];
            auto chain = box->GetChain(boxIndex);
            std::shared_ptr<MarkovChain> samps;
            if(useQois){
                samps = chain->GetQOIs();
            }else{
                samps = chain->GetSamples();
            } 

            MultiIndex index = *(box->GetLowestIndex()) + *boxIndex;
            MultiIndex indexDiffFromTop = *(box->GetHighestIndex()) - index;

            if (indexDiffFromTop.Sum() % 2 == 0) {
                diffMean += samps->Covariance(mean);
            } else {
                diffMean -= samps->Covariance(mean);
            }
        }

        // Add this term of the series to the running total      
        telescopingSum += diffMean;
    }

    return 0.5*(telescopingSum + telescopingSum.transpose());
}