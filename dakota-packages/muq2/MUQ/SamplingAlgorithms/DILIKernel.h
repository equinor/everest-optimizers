#ifndef DILIKERNEL_H_
#define DILIKERNEL_H_

#include "MUQ/SamplingAlgorithms/TransitionKernel.h"

#include "MUQ/SamplingAlgorithms/MCMCProposal.h"
#include "MUQ/Modeling/Distributions/GaussianBase.h"
#include "MUQ/Modeling/LinearAlgebra/LinearOperator.h"

namespace muq {
  namespace SamplingAlgorithms {

    /**
    @class AverageHessian
    @details During the construction of the LIS, we need the average Hessian.
             This class can be used to compute the action of the average Hessian
             given a generalized eigenvalue decomposition of the old Hessian and
             a linear operator for the new Hessian, i.e.,
             $$
             \bar{H} = \frac{m}{m+1}H_{old} + \frac{1}{m+1}H_{new}
             $$
             where
             $$
             H_{old}u_i = \lambda_i\Gamma^{-1}u_i = \lambda_i L^{-T}L^{-1} u_i
             $$
             and
             $$
             L^T H_{old} u_i = \lambda_i L^{-1}u_i
             $$
             now, with $u_i$ = L v_i$, we have
             $$
             L^T H_{old} L v_i = \lambda_i v_i
             $$
             which has eigenvalue decomposition
             $$
             L^T H_{old} L  = V \Lambda V^T
             $$
             and thus
             $$
             H_{old} = L^{-T} L^{-1}U\Lambda U^TL^{-T} L^{-1}
                     = \Gamma^{-1} U \Lambda U^T \Gamma^{-1}
             $$


    */
    class AverageHessian : public muq::Modeling::LinearOperator
    {
    public:
      AverageHessian(unsigned int                            numOldSamps,
                     std::shared_ptr<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>> const& uQRIn,
                     std::shared_ptr<Eigen::MatrixXd> const& oldWIn,
                     std::shared_ptr<Eigen::VectorXd> const& oldValsIn,
                     std::shared_ptr<muq::Modeling::LinearOperator> const& newHess);

      /** Apply the linear operator to a vector */
      virtual Eigen::MatrixXd Apply(Eigen::Ref<const Eigen::MatrixXd> const& x) override;

      /** Apply the transpose of the linear operator to a vector. */
      virtual Eigen::MatrixXd ApplyTranspose(Eigen::Ref<const Eigen::MatrixXd> const& x) override;

    private:
      const double numSamps;
      std::shared_ptr<Eigen::MatrixXd> oldW;
      std::shared_ptr<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>> uQR; // holds the QR decomp of U

      std::shared_ptr<Eigen::VectorXd> oldEigVals;
      std::shared_ptr<muq::Modeling::LinearOperator> newHess;
      Eigen::MatrixXd thinQ;

    };

    class CSProjector : public muq::Modeling::LinearOperator
    {
    public:

      CSProjector(std::shared_ptr<Eigen::MatrixXd> const& Uin,
                  std::shared_ptr<Eigen::MatrixXd> const& Win,
                  unsigned int                            lisDimIn) : LinearOperator(Uin->rows(), Win->rows()),
                                                                 U(Uin), W(Win), lisDim(lisDimIn){};

      virtual ~CSProjector() = default;

      /** Apply the linear operator to a vector */
      virtual Eigen::MatrixXd Apply(Eigen::Ref<const Eigen::MatrixXd> const& x) override;

      /** Apply the transpose of the linear operator to a vector. */
      virtual Eigen::MatrixXd ApplyTranspose(Eigen::Ref<const Eigen::MatrixXd> const& x) override;

    private:
      std::shared_ptr<Eigen::MatrixXd> U, W;
      unsigned int lisDim;
    };


    class LIS2Full : public muq::Modeling::LinearOperator
    {
    public:
      LIS2Full(std::shared_ptr<Eigen::MatrixXd> const& Uin,
               std::shared_ptr<Eigen::VectorXd> const& Lin) : LinearOperator(Uin->rows(), Lin->rows()),
                                                              U(Uin), L(Lin), lisDim(Lin->rows()){};

      virtual ~LIS2Full() = default;

      /** Apply the linear operator to a vector */
      virtual Eigen::MatrixXd Apply(Eigen::Ref<const Eigen::MatrixXd> const& x) override;

      /** Apply the transpose of the linear operator to a vector. */
      virtual Eigen::MatrixXd ApplyTranspose(Eigen::Ref<const Eigen::MatrixXd> const& x) override;

    private:
      std::shared_ptr<Eigen::MatrixXd> U;
      std::shared_ptr<Eigen::VectorXd> L;
      const unsigned int lisDim;
    };


    /**
      @ingroup MCMCKernels
      @class DILIKernel
      @brief An implementation of the Dimension Independent Likelihood Informed subspace (DILI) MCMC sampler
      @details

      Cui, T., Law, K. J., & Marzouk, Y. M. (2016). Dimension-independent likelihood-informed MCMC. Journal of Computational Physics, 304, 109-137.

      <B>Configuration Parameters:</B>

      Parameter Key | Type | Default Value | Description |
      ------------- | ------------- | ------------- | ------------- |
      "LIS Block"  | String | - | A string pointing to a block of kernel/proposal options for the Likelihood informed subspace. |
      "CS Block"   | String | - | A string pointing to a block of kernel/proposal options for the Complementary space.  Typically this will be a Crank-Nicolson proposal. |
      "HessianType" | String | GaussNewton | The type of posterior Hessian to use.  Either "Exact" or "GaussNewton"
      "Adapt Interval" | int | -1 | How many MCMC steps to take before updating the average Hessian and the likelihood informed subspace.  If negative, the LIS will be fixed to the initial value and will not be udpated. |
      "Adapt Start" | int | 1 | The number of MCMC steps taken before updating the LIS. |
      "Adapt End" | int | -1 | No LIS updates will occur after this number of MCMC steps.  If negative, the LIS will continue to be updated until the end of the chain. |
      "Initial Weight" | int | 100 | "Weight" or number of samples given to the to initial Hessian.  The weight on the previous average Hessian estimate is given by $(N+W)/(N+W+1)$, where $N$ is the number of MCMC steps taken and $W$ is this parameter. |
      "Hess Tolerance" | double | 1e-4 | The lower bound on eigenvalues used to construct the low-rank approximation of the global expected Hessian. |
      "LIS Tolerance" | double | 0.1 | Lower bound on eigenvalues used to construct local and global likelihood informed subspaces.  Must be larger than Hess Tolerance. |
      "Eigensolver Block" | String | - | A string pointing to a block of eigensolver options for solving the generalized eigenvalue problems. |

     */
    class DILIKernel : public TransitionKernel {
    public:

      DILIKernel(boost::property_tree::ptree       const& pt,
                 std::shared_ptr<AbstractSamplingProblem> problem);

      DILIKernel(boost::property_tree::ptree       const& pt,
                 std::shared_ptr<AbstractSamplingProblem> problem,
                 Eigen::VectorXd                   const& genEigVals,
                 Eigen::MatrixXd                   const& genEigVecs);

      DILIKernel(boost::property_tree::ptree                  const& pt,
                 std::shared_ptr<AbstractSamplingProblem>            problem,
                 std::shared_ptr<muq::Modeling::GaussianBase> const& prior,
                 std::shared_ptr<muq::Modeling::ModPiece>     const& noiseModel,
                 std::shared_ptr<muq::Modeling::ModPiece>     const& forwardModelIn);

      DILIKernel(boost::property_tree::ptree                  const& pt,
                 std::shared_ptr<AbstractSamplingProblem>            problem,
                 std::shared_ptr<muq::Modeling::GaussianBase> const& prior,
                 std::shared_ptr<muq::Modeling::ModPiece>     const& noiseModel,
                 std::shared_ptr<muq::Modeling::ModPiece>     const& forwardModelIn,
                 Eigen::VectorXd                              const& genEigVals,
                 Eigen::MatrixXd                              const& genEigVecs);


      DILIKernel(boost::property_tree::ptree                  const& pt,
                 std::shared_ptr<AbstractSamplingProblem>            problem,
                 std::shared_ptr<muq::Modeling::GaussianBase> const& prior,
                 std::shared_ptr<muq::Modeling::ModPiece>     const& likelihood);

      DILIKernel(boost::property_tree::ptree                  const& pt,
                 std::shared_ptr<AbstractSamplingProblem>            problem,
                 std::shared_ptr<muq::Modeling::GaussianBase> const& prior,
                 std::shared_ptr<muq::Modeling::ModPiece>     const& likelihood,
                 Eigen::VectorXd                              const& genEigVals,
                 Eigen::MatrixXd                              const& genEigVecs);

      virtual ~DILIKernel() = default;

      virtual inline std::shared_ptr<TransitionKernel> LISKernel() {return lisKernel;};
      virtual inline std::shared_ptr<TransitionKernel> CSKernel() {return csKernel;};

      virtual void PostStep(unsigned int const t,
                            std::vector<std::shared_ptr<SamplingState>> const& state) override;

      virtual std::vector<std::shared_ptr<SamplingState>> Step(unsigned int const t,
                                                               std::shared_ptr<SamplingState> prevState) override;

      virtual void PrintStatus(std::string prefix) const override;

      /** From a ModGraphPiece defining the posterior log density, this function
          extracts a ModGraphPiece defining the likelihood function.
      */
      static std::shared_ptr<muq::Modeling::ModPiece> ExtractLikelihood(std::shared_ptr<AbstractSamplingProblem> const& problem,
                                                                        std::string                              const& nodeName);


      /** From a ModGraphPiece defining the posterior log density, this function
          extracts a GaussianBase instance defining the prior.
      */
      static std::shared_ptr<muq::Modeling::GaussianBase> ExtractPrior(std::shared_ptr<AbstractSamplingProblem> const& problem,
                                                                       std::string                              const& nodeName);

      static std::shared_ptr<muq::Modeling::ModPiece> ExtractNoiseModel(std::shared_ptr<muq::Modeling::ModPiece> const& likelihood);

      // Extract the forward model from the likelihood.  This function assumes the graph has only one node between the output of the likelihood function and the output of the forward model
      static std::shared_ptr<muq::Modeling::ModPiece> ExtractForwardModel(std::shared_ptr<muq::Modeling::ModPiece> const& likelihoodIn);

      static std::shared_ptr<muq::Modeling::ModPiece> CreateLikelihood(std::shared_ptr<muq::Modeling::ModPiece> const& forwardModel,
                                                                       std::shared_ptr<muq::Modeling::ModPiece> const& noiseDensity);

      Eigen::MatrixXd const& LISVecs() const{return *hessU;};
      Eigen::VectorXd const& LISVals() const{return *hessEigVals;};
      Eigen::MatrixXd const& LISW() const{return *hessW;};
      Eigen::VectorXd const& LISL() const{return *lisL;};

      /** Returns the dimension of the LIS. */
      unsigned int LISDim() const{return lisDim;};

      /** Create the likelihood informed subspace for the first time. */
      void CreateLIS(std::vector<Eigen::VectorXd> const& currState);

      /** Returns the low dimensional LIS representation of a vector \f$x\f$
          defined in the full parameter space.
      */
      Eigen::VectorXd ToLIS(Eigen::VectorXd const& x) const;

      /** Returns a vector in the full space given a vector \f$r\f$ in the
          low dimensional LIS.
      */
      Eigen::VectorXd FromLIS(Eigen::VectorXd const& r) const;

      /** Projects a point \f$x\in\mathbb{R}^N\f$ in the full parameter space to
          a point \f$y\in\mathbb{R}^N\f$ that has the same dimension as \f$x\f$
          but lies on the complementary space.
      */
      Eigen::VectorXd ToCS(Eigen::VectorXd const& x) const;


    protected:

      std::pair<Eigen::MatrixXd, Eigen::VectorXd> ComputeLocalLIS(std::vector<Eigen::VectorXd> const& currState);


      /** Sets up the LIS based on the eigenvalues and eigenvectors solving the problem $Hu = \lambda \Gamma^{-1}u$
      */
      void SetLIS(Eigen::VectorXd const& eigVals, Eigen::MatrixXd const& eigVecs);



      /** Update the likelihood informed subspace using Hessian information at a
          new point.
          @param[in] numSamps The number of samples that have been used to estimate the current LIS.
          @param[in] currState The current state of the chain.
      */
      void UpdateLIS(unsigned int                        numSamps,
                     std::vector<Eigen::VectorXd> const& currState);

      // Used to reconstruct the LIS and CS kernels when lisU, lisW, or lisL changes.
      void UpdateKernels();

      boost::property_tree::ptree lisKernelOpts;
      boost::property_tree::ptree csKernelOpts;

      std::shared_ptr<muq::Modeling::ModPiece> logLikelihood;
      std::shared_ptr<muq::Modeling::GaussianBase> prior;

      std::shared_ptr<muq::Modeling::ModPiece> forwardModel;
      std::shared_ptr<muq::Modeling::ModPiece> noiseDensity;

      // A matrix containing the eigenvectors of the generalized eigenvalue problem Hv = lam*\Gamma^{-1}v
      std::shared_ptr<Eigen::MatrixXd> hessU;

      // The following matrices hold a QR decomposition of hessU
      std::shared_ptr<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>> hessUQR;

      // A vector of eigenvalues of the generalized eigenvalue problem.  Only values above hessValTol are kept.
      std::shared_ptr<Eigen::VectorXd> hessEigVals;

      // W = \Gamma^{-1} v
      std::shared_ptr<Eigen::MatrixXd> hessW;

      // L is the Cholesky factor of the approximate posterior covariance projected onto the LIS
      std::shared_ptr<Eigen::VectorXd> lisL;

      // Defines the operation from the LIS to the full space
      std::shared_ptr<muq::Modeling::LinearOperator> lisToFull;

      // Defines the projection onto the complementary space
      std::shared_ptr<muq::Modeling::LinearOperator> fullToCS;

      // Transition kernel for updating the LIS
      std::shared_ptr<TransitionKernel> lisKernel;

      // Transition kernel for updating the CS
      std::shared_ptr<TransitionKernel> csKernel;

      // The type of Hessian approximation to use.  "Exact" or "GaussNewton"
      std::string hessType;

      // Options for the Generalized eigenvalue solver
      boost::property_tree::ptree eigOpts;

      const int updateInterval;
      const int adaptStart;
      const int adaptEnd;
      const int initialHessSamps;

      /* The size of the LIS.  Note that the LIS can have a smaller dimension
       than the number of eigenvalue/eigenvector pairs kept to approximate the
       global hessian.  The number of eigenvalues greater than lisValTol dictates
       the LIS dimension
      */
      unsigned int lisDim=0;

      /* Threshold on eigenvalues defining the low rank approximation of the global
        hessian.  Note that this toleran is typically smaller than the tolerance
        used to define the LIS.  In Cui et al., this was set to 1e-4.
      */
      const double hessValTol;

      /* Threshold on eigenvalues used to define local and global LIS
         approximations.  Modes with eigenavalues greater than this threshold
         will be used.  In Cui et al., this was set to 0.1
      */
      const double lisValTol;

      unsigned int numLisUpdates;
    };
  } // namespace SamplingAlgorithms
} // namespace muq

#endif
