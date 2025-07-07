#ifndef MODPIECECOSTFUNCTION_H_
#define MODPIECECOSTFUNCTION_H_

#include "MUQ/Optimization/CostFunction.h"

namespace muq {
  namespace Optimization {
    class ModPieceCostFunction : public CostFunction {
    public:

      ModPieceCostFunction(std::shared_ptr<muq::Modeling::ModPiece> cost, double scaleIn=1.0);

      virtual ~ModPieceCostFunction() = default;

      /// The value of the cost function
      virtual double Cost() override;

      /// Compute the gradient of the cost function
      virtual Eigen::VectorXd Gradient() override;

      // Apply the Hessian to a vector
      virtual Eigen::VectorXd ApplyHessian(Eigen::VectorXd const& vec) override;

    private:
         
      // The muq::Modeling::ModPiece that holds the cost
      std::shared_ptr<muq::Modeling::ModPiece> cost;

      const double scale;
    };
  } // namespace Optimization
} // namespace muq

#endif
