#ifndef KALMANSMOOTHER_H
#define KALMANSMOOTHER_H

#include "MUQ/Inference/Filtering/KalmanFilter.h"

namespace muq
{
namespace Inference
{

    /** @class KalmanSmoother
        @details Implements the Rauch–Tung–Striebel smoother.
    */
    class KalmanSmoother
    {

    public:

        /** @param[in] currDist_t The distribution at time t after the forward Kalman filtering step (i.e., using all data up to and including time t).
            @param[in] nextDist_t The distribution at time t+1 from the prediction phase of the Kalman filtering step (i.e., using all data up to time t).
            @param[in] nextDist_n The distribution at time t+1 after the RTS smoothing step (i.e., using all data).
            @param[in] F The linear operator acting on the state at time t, to produce the state at time t+1
            @returns A distribution (mean and covariance) at time t after accounting for all data.
        */
        static std::pair<Eigen::VectorXd, Eigen::MatrixXd> Analyze(std::pair<Eigen::VectorXd, Eigen::MatrixXd> const& currDist_t,
                                                                   std::pair<Eigen::VectorXd, Eigen::MatrixXd> const& nextDist_t,
                                                                   std::pair<Eigen::VectorXd, Eigen::MatrixXd> const& nextDist_n,
                                                                   std::shared_ptr<muq::Modeling::LinearOperator>     F);

        /**
         * @tparam MatrixType A type that can be converted to a MUQ LinearOperator.  Examples include Eigen::MatrixXd and Eigen::SparseMatrix
         */
        template<typename MatrixType>
        static std::pair<Eigen::VectorXd, Eigen::MatrixXd> Analyze(std::pair<Eigen::VectorXd, Eigen::MatrixXd> const& currDist_t,
                                                                   std::pair<Eigen::VectorXd, Eigen::MatrixXd> const& nextDist_t,
                                                                   std::pair<Eigen::VectorXd, Eigen::MatrixXd> const& nextDist_n,
                                                                   MatrixType                                  const& F)
        {
            return Analyze(currDist_t, nextDist_t, nextDist_n, muq::Modeling::LinearOperator::Create(F));
        };

        
    private:                                                           

        static Eigen::MatrixXd ComputeC( Eigen::MatrixXd                          const& currDist_t_cov,
                                         Eigen::MatrixXd                          const& nextDist_t_cov,
                                         std::shared_ptr<muq::Modeling::LinearOperator> const& F);
        

    }; // class KalmanSmoother 
    
}// namespace Inference
}// namespace muq


#endif 
