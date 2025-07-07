#include "MUQ/SamplingAlgorithms/SampleEstimator.h"

#include "MUQ/Utilities/RandomGenerator.h"

#include <gtest/gtest.h>

using namespace muq::SamplingAlgorithms;
using namespace muq::Modeling;
using namespace muq::Utilities;

class MonteCarloEstimatorTest : public SampleEstimator 
{
public:

    MonteCarloEstimatorTest(Eigen::MatrixXd const& sampsIn) : samps(sampsIn), dim(samps.rows()){};
    
    virtual ~MonteCarloEstimatorTest() = default;

    virtual unsigned int NumBlocks() const override 
    {
        return 1;
    };

    virtual unsigned int BlockSize(int blockInd) const override 
    {
        return dim;
    };

    virtual Eigen::MatrixXd Covariance(Eigen::VectorXd const& mean, 
                                       int                    blockInd=-1) const override
    {
        Eigen::MatrixXd diff = samps.colwise() - mean;

        return (diff * diff.transpose())/(samps.cols()-1.0);   
    }

    virtual Eigen::VectorXd ExpectedValue(std::shared_ptr<muq::Modeling::ModPiece> const& f,
                                          std::vector<std::string> const& metains = std::vector<std::string>()) const override 
    {
        Eigen::MatrixXd results(f->outputSizes(0), samps.cols());
        for(unsigned int i=0; i<samps.cols(); ++i)
            results.col(i) = f->Evaluate(samps.col(i).eval()).at(0);

        return results.rowwise().mean();
    }

    virtual Eigen::VectorXd ESS(int blockInd, std::string const& method) const override
    {
        return samps.cols()*Eigen::VectorXd::Ones(dim);
    };

    virtual Eigen::VectorXd StandardError(int blockInd, std::string const& method) const override
    {
        return (Variance(blockInd).array() / (samps.cols()*Eigen::VectorXd::Ones(dim)).array()).sqrt();
    }

private:
    Eigen::MatrixXd samps;
    unsigned int dim;
};

TEST(SampleEstimatorTest, Gaussian){

    const double exactTol = 1e-12;
    const double inexactTol = 0.07;

    unsigned int dim = 10;
    unsigned int numSamps = 50000;

    Eigen::MatrixXd samps = RandomGenerator::GetNormal(dim,numSamps); 

    MonteCarloEstimatorTest estimator(samps);

    Eigen::VectorXd result = estimator.Mean();
    Eigen::VectorXd trueResult = samps.rowwise().mean();
    for(unsigned int i=0; i<dim; ++i)
        EXPECT_NEAR(trueResult(i),result(i), exactTol);

    result = estimator.Variance();
    trueResult = (samps.colwise() - samps.rowwise().mean()).array().square().rowwise().mean();
    for(unsigned int i=0; i<dim; ++i)
        EXPECT_NEAR(trueResult(i), result(i), exactTol);

    result = estimator.Variance(Eigen::VectorXd::Zero(dim));
    for(unsigned int i=0; i<dim; ++i)
        EXPECT_NEAR(1.0,result(0), inexactTol);

    result = estimator.CentralMoment(3);
    for(unsigned int i=0; i<dim; ++i)
        EXPECT_NEAR(0.0,result(i), inexactTol);

    result = estimator.Skewness();
    for(unsigned int i=0; i<dim; ++i)
        EXPECT_NEAR(0.0,result(i), inexactTol);

    result = estimator.Kurtosis();
    for(unsigned int i=0; i<dim; ++i)
        EXPECT_NEAR(3.0, result(i), inexactTol);
    
}