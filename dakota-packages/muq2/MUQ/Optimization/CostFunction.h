#ifndef COSTFUNCTION_H_
#define COSTFUNCTION_H_

#include "MUQ/Utilities/VariadicMacros.h"

#include "MUQ/Modeling/ModPiece.h"

namespace muq {
  namespace Optimization {

    /// The cost function for an optimization routine
    /**
       The cost function has the form:
       \f{equation}{
       c = J(x),
       \f}
       where, \f$c \in \mathbb{R}\f$.

       This class defines an interface for evaluating such a cost function, as
       well as evaluating the gradient and Hessian.  There are two ways to use
       this class.  The first, is just to call the Cost, Gradient, and Hessian
       functions with the point we want to evaluate.  This is shown below:
       @code
       std::shared_ptr<CostFunction> cost = ...

       Eigen::VectorXd evalPt = ...

       // Compute the cost
       double c = cost->Cost(evalPt);

       // Compute the gradient
       Eigen::VectorXd grad = cost->Gradient(evalPt);

       // Compute the Hessian matrix H
       Eigen::MatrixXd hess = cost->Hessian(evalPt);

       // Compute the matrix product Hg
       Eigen::VectorXd hessApp = cost->ApplyHessian(evalPt, grad);
       @endcode
       The above code is straightforward, but makes it difficult to share
       calculations between the Cost, Gradient, and Hessian functions.  For example,
       all of these functions might require the same model evaluation. To help
       overcome this issue, a second way to use this class is by first setting
       the evaluation point and then calling the Cost, Gradient, or Hessian functions.
       For example,
       @code
       std::shared_ptr<CostFunction> cost = ...

       Eigen::VectorXd evalPt = ...

       // Set the shared evaluation point
       cost->SetPoint(evalPt);

       // Extract the cost
       double c = cost->Cost();

       // Compute the gradient
       Eigen::VectorXd grad = cost->Gradient();

       // Compute the Hessian matrix H
       Eigen::MatrixXd hess = cost->Hessian();

       // Compute the matrix product Hg
       Eigen::VectorXd hessApp = cost->ApplyHessian(grad);
       @endcode
       By setting the evaluation point first, we can precompute any information that
       might be shared across calls to Cost, Gradient, Hessian, etc...

       To define a child of the CostFunction class, the user needs to implement
       the Cost() function that does not take any arguments.  If information can
       be shared between Cost, Gradient, and Hessian functions, then the user
       will also want to override the SetPoint function.
     */
    class CostFunction : public muq::Modeling::ModPiece {
    public:

      /**
	       @param[in] dim The number of the decision variables in \f$x\f$.
       */
      CostFunction(unsigned int dim) :
        muq::Modeling::ModPiece(Eigen::VectorXi::Constant(1,dim), Eigen::VectorXi::Ones(1)) {};

      virtual ~CostFunction() = default;

      virtual void SetPoint(Eigen::VectorXd const& evalPt);

      /// The value of the cost function
      /**
	      @param[in] input The inputs \f$x\f$, \f$\theta_{1:n}\f$
	      \return The value of the cost function
       */
      virtual double Cost(Eigen::VectorXd const& x){SetPoint(x); return Cost();};
      virtual double Cost() = 0;

      virtual Eigen::VectorXd Gradient(Eigen::VectorXd const& evalPt){SetPoint(evalPt); return Gradient();};
      virtual Eigen::VectorXd Gradient();


      /// The Hessian of the cost function
      /**
         @param[in] inputDimWrt Which input are we taking the 2nd derivative with respect to?
         @param[in] input The inputs \f$x\f$, \f$\theta_{1:n}\f$
         \return The Hessian of the cost function
      */
      virtual Eigen::MatrixXd Hessian(Eigen::VectorXd const& evalPt){SetPoint(evalPt); return Hessian();};
      virtual Eigen::MatrixXd Hessian();

      /// The Hessian of the cost function using finite difference
      /**
         @param[in] inputDimWrt Which input are we taking the 2nd derivative with respect to?
         @param[in] input The inputs \f$x\f$, \f$\theta_{1:n}\f$
         \return The Hessian of the cost function
      */
      virtual Eigen::MatrixXd HessianByFD(Eigen::VectorXd const& evalPt){SetPoint(evalPt); return  HessianByFD();};
      virtual Eigen::MatrixXd HessianByFD();

      /// The Hessian of the cost function
      /**
         @param[in] inputDimWrt Which input are we taking the 2nd derivative with respect to?
         @param[in] input The inputs \f$x\f$, \f$\theta_{1:n}\f$
         @param[in] vec Vector to which the Hessian is applied
         \return The Hessian action on vec
      */
      virtual Eigen::VectorXd ApplyHessian(Eigen::VectorXd const& evalPt,
                                           Eigen::VectorXd const& vec){SetPoint(evalPt); return ApplyHessian(vec);};

      virtual Eigen::VectorXd ApplyHessian(Eigen::VectorXd const& vec);

    protected:
      Eigen::VectorXd x;

    private:

      /// The value of the cost function
      /**
	 @param[in] args The inputs \f$x\f$, \f$\theta_{1:n}\f$
	 \return The value of the cost function
       */
      virtual void EvaluateImpl(muq::Modeling::ref_vector<Eigen::VectorXd> const& input) override;

      /// Compute the gradient of the cost function
      /**
	 @param[in] inputDimWrt Which input are we taking the derivative with respect to?
	 @param[in] args The inputs \f$x\f$, \f$\theta_{1:n}\f$ and the sensitivity vector
       */
      virtual void GradientImpl(unsigned int outputDimWrt, 
                                unsigned int inputDimWrt, 
                                muq::Modeling::ref_vector<Eigen::VectorXd> const& input, 
                                Eigen::VectorXd const& sensitivity) override;

      virtual void JacobianImpl(unsigned int outputDimWrt, 
                                unsigned int inputDimWrt, 
                                muq::Modeling::ref_vector<Eigen::VectorXd> const& input) override;

      virtual void ApplyHessianImpl(unsigned int outWrt,
                                    unsigned int inWrt1,
                                    unsigned int inWrt2,
                                    muq::Modeling::ref_vector<Eigen::VectorXd> const& input,
                                    Eigen::VectorXd const& sensitivity,
                                    Eigen::VectorXd const& vec) override;

    };
  } // namespace Optimization
} // namespace muq

#endif
