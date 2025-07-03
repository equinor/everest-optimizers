#ifndef KALMANFILTER_H
#define KALMANFILTER_H

#include "MUQ/Modeling/LinearAlgebra/LinearOperator.h"
#include "MUQ/Modeling/LinearAlgebra/EigenLinearOperator.h"

#include <Eigen/Core>

#include <memory>


namespace muq
{
namespace Inference
{

    class KalmanFilter
    {

    public:
        
        /** @param[in] dist A pair containing the mean vector and covariance matrix of the Gaussian prior distribution at time t before incorporating the data from time t.
            @param[in] H The observation operator, which relates the state \f$x\f$ and observation \f$y_{obs}\f$ through  \f$y_{obs} = Hx + \epsilon\f$, where \f$\epsilon\f$ is additive Gaussian noise.
            @param[in] obsMean The value of \f$y_{obs}\f$
            @param[in] obsCov The covariance of the additive noise \f$\epsilon\f$.
            @returns A pair containing the mean and covariance of the posterior distribution.
        */
        static std::pair<Eigen::VectorXd, Eigen::MatrixXd> Analyze(std::pair<Eigen::VectorXd, Eigen::MatrixXd> const& dist,
                                                                   std::shared_ptr<muq::Modeling::LinearOperator>     H,
                                                                   Eigen::Ref<const Eigen::VectorXd> const&           obsMean,
                                                                   Eigen::Ref<const Eigen::MatrixXd> const&           obsCov);

        /**
         * @tparam MatrixType A type that can be converted to a MUQ LinearOperator.  Examples include Eigen::MatrixXd and Eigen::SparseMatrix
         */
        template<typename MatrixType>
        static std::pair<Eigen::VectorXd, Eigen::MatrixXd> Analyze(std::pair<Eigen::VectorXd, Eigen::MatrixXd> const& dist,
                                                                   MatrixType                                  const& H,
                                                                   Eigen::Ref<const Eigen::VectorXd> const&           obsMean,
                                                                   Eigen::Ref<const Eigen::MatrixXd> const&           obsCov)
        {
            return Analyze(dist, muq::Modeling::LinearOperator::Create(H), obsMean, obsCov);
        };


    private:
        
        static Eigen::MatrixXd ComputeGain(Eigen::MatrixXd                           const& HP,
                                           std::shared_ptr<muq::Modeling::LinearOperator>  H,
                                           Eigen::Ref<const Eigen::MatrixXd> const&         obsCov);
        
        
    }; // class KalmanFilter

} // namespace Inference
} // namespace muq

#endif
