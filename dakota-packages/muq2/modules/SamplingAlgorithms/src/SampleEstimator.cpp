#include "MUQ/SamplingAlgorithms/SampleEstimator.h"
#include "MUQ/Modeling/LinearAlgebra/IdentityOperator.h"

#include <Eigen/Core>

using namespace muq::Modeling;
using namespace muq::SamplingAlgorithms;

Eigen::VectorXd SampleEstimator::CentralMoment(unsigned int order, 
                                               int          blockNum) const
{
    return CentralMoment(order, Mean(blockNum), blockNum);
}


Eigen::VectorXd SampleEstimator::CentralMoment(unsigned int           order, 
                                               Eigen::VectorXd const& mean, 
                                               int                    blockNum) const
{   
    class Integrand : public muq::Modeling::ModPiece
    {
    public:
        Integrand(Eigen::VectorXd const& meanIn,
                  unsigned int           orderIn,
                  int                    blockIndIn,
                  Eigen::VectorXi const& blockSizes) : muq::Modeling::ModPiece(blockSizes, GetBlockSize(blockSizes,blockIndIn)*Eigen::VectorXi::Ones(1)),
                                                       localMean(meanIn), localOrder(orderIn), localBlockInd(blockIndIn){};

        virtual ~Integrand() = default;

        virtual void EvaluateImpl(ref_vector<Eigen::VectorXd> const& inputs) override
        {
            if(localBlockInd<0){
                outputs.resize(1);
                outputs.at(0).resize(inputSizes.sum());
                unsigned int cumSum = 0;

                for(unsigned int i=0; i<inputSizes.size(); ++i){
                    outputs.at(0).segment(cumSum,inputSizes(i)) = (inputs.at(i).get()-localMean.segment(cumSum,inputSizes(i))).array().pow(localOrder);
                    cumSum += inputSizes(i);
                }
                
            }else{
                outputs.resize(1);
                outputs.at(0) = (inputs.at(localBlockInd).get()-localMean).array().pow(localOrder);
            }
        };

        static int GetBlockSize(Eigen::VectorXi const& blockSizes, int blockInd){
            if(blockInd<0){
                return blockSizes.sum();
            }else{
                return blockSizes(blockInd);
            }
        };

    private:
        Eigen::VectorXd const& localMean;
        unsigned int localOrder;
        int localBlockInd;

    }; // class integrand

    // Get the size of each block
    Eigen::VectorXi blockSizes(NumBlocks());
    for(int i=0; i<NumBlocks(); ++i)
        blockSizes(i) = BlockSize(i);

    // Create an integrand for the central moment
    auto integrand = std::make_shared<Integrand>(mean, order, blockNum, blockSizes);

    return ExpectedValue(integrand);
}


Eigen::VectorXd SampleEstimator::Mean(int blockInd) const
{   
    assert(BlockSize(blockInd)>0);
    return CentralMoment(1, Eigen::VectorXd::Zero(BlockSize(blockInd)), blockInd);
}
    
Eigen::VectorXd SampleEstimator::Variance(int blockInd) const
{   
    return Variance(Mean(blockInd), blockInd);
}

Eigen::VectorXd SampleEstimator::Variance(Eigen::VectorXd const& mean, 
                                          int                    blockInd) const
{
    return CentralMoment(2, mean, blockInd);
}

Eigen::VectorXd SampleEstimator::StandardizedMoment(unsigned int order, int blockInd) const
{
    return StandardizedMoment(order, Mean(blockInd), blockInd);
}

Eigen::VectorXd SampleEstimator::StandardizedMoment(unsigned int           order,
                                                    Eigen::VectorXd const& mean,
                                                    int                    blockInd) const
{   
    return StandardizedMoment(order, mean, Variance(mean,blockInd).array().sqrt(), blockInd);
}

Eigen::VectorXd SampleEstimator::StandardizedMoment(unsigned int           order,
                                                    Eigen::VectorXd const& mean,
                                                    Eigen::VectorXd const& stdDev,
                                                    int                    blockInd) const
{
    Eigen::VectorXd moment = CentralMoment(order, mean, blockInd);
    return moment.array() / stdDev.array().pow(order);
}


Eigen::VectorXd SampleEstimator::Skewness(int blockInd) const
{
    return StandardizedMoment(3,blockInd);
}

Eigen::VectorXd SampleEstimator::Skewness(Eigen::VectorXd const& mean,
                                          int                    blockInd) const
{   
    return StandardizedMoment(3,mean, blockInd);
}

Eigen::VectorXd SampleEstimator::Skewness(Eigen::VectorXd const& mean,
                                          Eigen::VectorXd const& stdDev,
                                          int                    blockInd) const
{   
    return StandardizedMoment(3,mean, stdDev, blockInd);
}

Eigen::VectorXd SampleEstimator::Kurtosis(int blockInd) const
{
    return StandardizedMoment(4,blockInd);
}

Eigen::VectorXd SampleEstimator::Kurtosis(Eigen::VectorXd const& mean,
                                          int                    blockInd) const
{   
    return StandardizedMoment(4,mean, blockInd);
}

Eigen::VectorXd SampleEstimator::Kurtosis(Eigen::VectorXd const& mean,
                                          Eigen::VectorXd const& stdDev,
                                          int                    blockInd) const
{   
    return StandardizedMoment(4,mean, stdDev, blockInd);
}


Eigen::MatrixXd SampleEstimator::Covariance(int blockInd) const
{
    return Covariance(Mean(blockInd),blockInd);
}