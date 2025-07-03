#ifndef LINEARSDE_H
#define LINEARSDE_H

#include "MUQ/Modeling/LinearAlgebra/EigenLinearOperator.h"

#include "MUQ/Utilities/RandomGenerator.h"

#include <boost/property_tree/ptree.hpp>
#include <random>

namespace muq
{
namespace Modeling
{

    /** @brief Defines a linear time invariant stochastic differential equation with Gaussian process noise.
        @details This class defines a LTI SDE of the form
        \f[
            \frac{\partial f(t)}{\partial t} = F f(t) + L w(t),
        \f]
        where \f$f(t)\f$ is the solution in \f$\mathbb{R}^M\f$, \f$F\f$ is an \f$M\timesM\f$ matrix, \f$L\f$ is an \f$M\times N\f$ matrix,
        and \f$w(t)\f$ is a white noise process with an \f$N\timesN\f$ covariance matrix \f$Q\f$.

        It is possible to ignore the process noise by setting the matrix \f$L\f$ to a nullptr in the constructor.  This is used in the 
        statespace representation of the constant Gaussian process covariance kernel.
    */
    class LinearSDE
    {

    public:

        template<typename EigenType1, typename EigenType2>
        LinearSDE(EigenType1                   const& Fin,
                  EigenType2                   const& Lin,
                  Eigen::MatrixXd              const& Qin,
                  boost::property_tree::ptree         options) : LinearSDE(muq::Modeling::LinearOperator::Create(Fin),
                                                                           muq::Modeling::LinearOperator::Create(Lin),
                                                                           Qin,
                                                                           options)
        {};
        
        template<typename Type1, typename Type2>
        LinearSDE(std::shared_ptr<Type1>      Fin,
                  std::shared_ptr<Type2>      Lin,
                  Eigen::MatrixXd      const& Qin,
                  boost::property_tree::ptree options) : LinearSDE(std::dynamic_pointer_cast<LinearOperator>(Fin),std::dynamic_pointer_cast<LinearOperator>(Lin), Qin, options){};


        LinearSDE(std::shared_ptr<muq::Modeling::LinearOperator>     Fin,
                  std::shared_ptr<muq::Modeling::LinearOperator>     Lin,
                  Eigen::MatrixXd                             const& Qin,
                  boost::property_tree::ptree                        options);

        
        LinearSDE(std::shared_ptr<muq::Modeling::LinearOperator>    Fin,
                  boost::property_tree::ptree                        options);


        /** Given \f$f(t)\f$, the state of the system at time \f$t\f$, return a random realization of the state at time \f$t+\delta t\f$.
         */
        template<typename EigenRefVector>
        Eigen::VectorXd EvolveState(EigenRefVector const& f0,
                                    double                 T) const
        {
            Eigen::VectorXd fnext(f0.size());
            EvolveState(f0, T, Eigen::Ref<Eigen::VectorXd>(fnext));
            return fnext;
        }

        template<typename EigenRefVector1, typename EigenRefVector2>
        void EvolveState(EigenRefVector1 const& f0,
                         double                 T,
                         EigenRefVector2        f) const
        {   
            f = f0;

            if(T<std::numeric_limits<double>::epsilon()){
                return;
            }

            const int numTimes = std::ceil(T/dt);

            Eigen::VectorXd z;
            
            // Take all but the last step.  The last step might be a partial step
            for(int i=0; i<numTimes-1; ++i)
            {   
                if(L){
                    z = sqrt(dt) * (sqrtQ.triangularView<Eigen::Lower>() * muq::Utilities::RandomGenerator::GetNormal(sqrtQ.cols()) ).eval();
                    f += dt*F->Apply(f) + L->Apply( z );
                }else{
                    f += dt*F->Apply(f);
                }
            }

            // Now take the last step
            double lastDt = T-(numTimes-1)*dt;
            if(L){
                z = sqrt(lastDt) * (sqrtQ.triangularView<Eigen::Lower>() * muq::Utilities::RandomGenerator::GetNormal(sqrtQ.cols())).eval();
                f += lastDt*F->Apply(f) + L->Apply( z );
            }else{
                f += lastDt*F->Apply(f);
            }
        }


        /** Given the mean and covariance of the solution at time \f$t\f$, compute the mean and covariance of the solution at time \f$t+T\f$.
         */
        template<typename EigenRefVector, typename EigenRefMatrix>
        std::pair<Eigen::VectorXd, Eigen::MatrixXd> EvolveDistribution(EigenRefVector const& mu0,
                                                                       EigenRefMatrix const& gamma0,
                                                                       double                T) const
        {
            Eigen::VectorXd mu(mu0.size());
            Eigen::MatrixXd cov(gamma0.rows(), gamma0.cols());

            EvolveDistribution(mu0,gamma0, T, Eigen::Ref<Eigen::VectorXd>(mu), Eigen::Ref<Eigen::MatrixXd>(cov));
            return std::make_pair(mu,cov);
        }

        template<typename EigenRefVector1, typename EigenRefMatrix1, typename EigenRefVector2, typename EigenRefMatrix2>
        void EvolveDistribution(EigenRefVector1 const& mu0,
                                EigenRefMatrix1 const& gamma0,
                                double                 T,
                                EigenRefVector2        mu,
                                EigenRefMatrix2        gamma) const
        {
            
            if(mu0.size()!=mu.size()){
                throw std::runtime_error("In LinearSDE::EvolveDistribution: mu0 and mu have different sizes.");
            }
            if((gamma0.rows()!=gamma.rows())||(gamma0.cols()!=gamma.cols())){
                throw std::runtime_error("In LinearSDE::EvolveDistribution: gamma0 and gamma have different sizes.");
            }

            mu = mu0;
            gamma = gamma0;
            
            if(T<std::numeric_limits<double>::epsilon()){
                return;
            }

            const int numTimes = std::ceil(T/dt);

            // Fourth-Order Stochastic Runge-Kutta method
            if(integratorType=="SRK4"){

                Eigen::MatrixXd Fgamma, k1, k2, k3, k4;

                // Take all but the last step because the last step might be a partial step.
                for(int i=0; i<numTimes-1; ++i)
                {
                    k1 = F->Apply(mu);
                    k2 = F->Apply(mu + 0.5*dt*k1);
                    k3 = F->Apply(mu + 0.5*dt*k2);
                    k4 = F->Apply(mu + dt*k3);
                    mu = mu + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);
                    
                    Fgamma = F->Apply(gamma);
                    k1 = Fgamma + Fgamma.transpose() + LQLT;
                    Fgamma = F->Apply(gamma + 0.5*dt*k1);
                    k2 = Fgamma + Fgamma.transpose() + LQLT;
                    Fgamma = F->Apply(gamma + 0.5*dt*k2);
                    k3 = Fgamma + Fgamma.transpose() + LQLT;
                    Fgamma = F->Apply(gamma + dt*k3);
                    k4 = Fgamma + Fgamma.transpose() + LQLT;

                    gamma = gamma + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);
                }

                // Take the last step
                double lastDt = T-(numTimes-1)*dt;

                k1 = F->Apply(mu);
                k2 = F->Apply(mu + 0.5*lastDt*k1);
                k3 = F->Apply(mu + 0.5*lastDt*k2);
                k4 = F->Apply(mu + lastDt*k3);
                mu = mu + (lastDt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);
                
                Fgamma = F->Apply(gamma);
                k1 = Fgamma + Fgamma.transpose() + LQLT;
                Fgamma = F->Apply(gamma + 0.5*lastDt*k1);
                k2 = Fgamma + Fgamma.transpose() + LQLT;
                Fgamma = F->Apply(gamma + 0.5*lastDt*k2);
                k3 = Fgamma + Fgamma.transpose() + LQLT;
                Fgamma = F->Apply(gamma + lastDt*k3);
                k4 = Fgamma + Fgamma.transpose() + LQLT;
                
                gamma = gamma + (lastDt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4);

            // Euler-Maruyama method
            }else{

                // Take all but the last step because the last step might be a partial step.
                for(int i=0; i<numTimes-1; ++i){
                    mu += dt*F->Apply(mu);
                    gamma += dt*dt*(F->Apply(F->Apply(gamma).transpose().eval()).transpose()) + dt*LQLT;
                }

                // Take the last step
                double lastDt = T-(numTimes-1)*dt;

                mu += lastDt*F->Apply(mu);
                gamma += lastDt*lastDt*(F->Apply(F->Apply(gamma).transpose().eval()).transpose()) + lastDt*LQLT;

            }
        }

        /** Evolve the mean and covariance of the system using a std::pair to hold the distribution.
         */
        template<typename EigenRefVector, typename EigenRefMatrix>
        std::pair<Eigen::VectorXd, Eigen::MatrixXd> EvolveDistribution(std::pair<EigenRefVector,EigenRefMatrix> const& muCov,
                                                                       double                                          T) const{
            return EvolveDistribution(muCov.first, muCov.second, T);
        }; 


        /** 
           Compute a matrix A and covariance Q such that \f$x(t+\delta t) = A x(t) + q\f$ where \f$q\f$ is a normal random variable with covariance \f$Q\f$.
         */
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd> Discretize(double deltaT);
        
        /** @brief Combines the states of multiple SDEs into a single monolitch SDE.
            @details Consider \f$N\f$ different stochastic differential equations defined by matrices \f$F_i\f$, \f$L_i\f$, and process covariances \f$Q_i\f$.   This function creates a new SDE defined by block diagonal matrices \f$F\f$, \f$L\f$, and \f$Q\f$:
\f[
F = \left[\begin{array}{cccc} F_1 & 0 & \cdots & \\ 0 & F_2 & 0 \\ \vdots & & \ddots & \\ 0 & \cdots & & F_N \end{array}\right]
\f]
\f[
L = \left[\begin{array}{cccc} L_1 & 0 & \cdots & \\ 0 & L_2 & 0 \\ \vdots & & \ddots & \\ 0 & \cdots & & L_N \end{array}\right]
\f]
\f[
Q = \left[\begin{array}{cccc} Q_1 & 0 & \cdots & \\ 0 & Q_2 & 0 \\ \vdots & & \ddots & \\ 0 & \cdots & & Q_N \end{array}\right]
\f]
         */
        static std::shared_ptr<LinearSDE> Concatenate(std::vector<std::shared_ptr<LinearSDE>> const& sdes,
                                                      boost::property_tree::ptree                    options = boost::property_tree::ptree());
        

        /// The dimension of the state variable \f$f(t)\f$.
        const int stateDim;


        std::shared_ptr<muq::Modeling::LinearOperator> GetF() const{return F;};
        std::shared_ptr<muq::Modeling::LinearOperator> GetL() const{return L;};
        Eigen::MatrixXd const& GetQ() const{return Q;};
        
        
    protected:

        void ExtractOptions(boost::property_tree::ptree options);
        
        std::shared_ptr<muq::Modeling::LinearOperator> F;
        std::shared_ptr<muq::Modeling::LinearOperator> L;

        Eigen::MatrixXd Q;
        Eigen::MatrixXd sqrtQ;

        double dt; // time step used in SDE integration
        std::string integratorType; // Type of integratin to use (either EM for "Euler Maruyama" or "SRK4" for Stochastic Runge Kutta)

        Eigen::MatrixXd LQLT;
    };


}// namespace Modeling
}// namespace muq




#endif
