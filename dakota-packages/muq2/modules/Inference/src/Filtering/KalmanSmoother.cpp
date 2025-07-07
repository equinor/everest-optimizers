#include "MUQ/Inference/Filtering/KalmanSmoother.h"

#include <Eigen/Dense>

using namespace muq::Modeling;
using namespace muq::Inference;


std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanSmoother::Analyze(std::pair<Eigen::VectorXd, Eigen::MatrixXd> const& currDist_t,
                                                                    std::pair<Eigen::VectorXd, Eigen::MatrixXd> const& nextDist_t,
                                                                    std::pair<Eigen::VectorXd, Eigen::MatrixXd> const& nextDist_n,
                                                                    std::shared_ptr<muq::Modeling::LinearOperator>     F)
{
    std::pair<Eigen::VectorXd, Eigen::MatrixXd> output;
    
    Eigen::MatrixXd C = ComputeC(currDist_t.second, nextDist_t.second, F);
    
    output.first = currDist_t.first + C*(nextDist_n.first - nextDist_t.first);
    output.second = currDist_t.second + C*(nextDist_n.second - nextDist_t.second).selfadjointView<Eigen::Lower>()*C.transpose();
    
    return output;

}


Eigen::MatrixXd KalmanSmoother::ComputeC(Eigen::MatrixXd                          const& currDist_t_cov,
                                         Eigen::MatrixXd                          const& nextDist_t_cov,
                                         std::shared_ptr<muq::Modeling::LinearOperator> const& F)
{   
    return  nextDist_t_cov.ldlt().solve( F->Apply(currDist_t_cov) ).transpose();
}
