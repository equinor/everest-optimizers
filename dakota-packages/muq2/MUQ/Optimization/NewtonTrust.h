#ifndef NEWTONTRUST_H_
#define NEWTONTRUST_H_

#include <boost/property_tree/ptree.hpp>

#include "MUQ/Optimization/Optimizer.h"
#include "MUQ/Optimization/CostFunction.h"

namespace muq {
  namespace Optimization {

    /** @class NewtonTrust
        @ingroup Optimization
        @brief Newton optimizer with trust region to ensure global convergence.
        @details Implements a trust region optimizer with quadratic model
                 subproblems for use on unconstrainted optimization problems.
                 Uses a Steihaug-CG method to approximately solve the subproblem.
    */
    class NewtonTrust : public Optimizer {

    public:

      /**

      Let \f$m_k(p)\f$ denote the quadratic model at iteration \f$k\f$ and \f$f(x)\f$ the trust cost function.
      After minimizing the quadratic model in the trust region, this algorithm
      measures the approximation quality of the quadratic model by looking at
      the change in the quadratic model compared to the change in the true objective.
      Let \f$\rho\f$ define the ratio of these changes:
      \f[
      \rho = \frac{f(x_k) - f(x_k_p_k)}{m_k(0)-m_k(p_k)},
      \f]
      where \f$p_k\f$ is the step that minimizes the quadratic approximation \f$m_k(p)\f$
      inside the trust region.

      <h3>Options</h3>
      <tr><th>Option Key <th> Optional/Required <th> Type <th> Possible Values <th> Default <th> Description
      <tr><td> PrintLevel <td> Optional <td> integer <td> \f${0,1}\f$ <td> 0 <td> Verbosity of output to std::cout.  If 0, no outut is printed.  If 1, messages are printed for every step of the optimizer.
      <tr><td> Ftol.AbsoluteTolerance <td> Optional <td> double <td> Any nonnegative real number. <td> 1e-8 <td> Termination criterion based on value of function value.  If the change in function value is less than this, the method terminates.
      <tr><td> Xtol.AbsoluteTolerance <td> Optional <td> double <td> Any nonnegative real number. <td> 1e-8 <td> Termination criterion based on the change of optimization variables or gradient. If the norm of the optimization step is less than this, the method terminates.
      <tr><td> MaxEvaluations <td> Optional <td> unsigned int <td> Any natural number. <td> 100 <td> The maximum number of optimization steps allowed.
      <tr><td> InitialRadius <td> Optional <td> double <td> Any positive real number. <td> min(1.0, MaxRadius) <td> Initial trust region radius.
      <tr><td> MaxRadius <td> Optional <td> double <td> Any positive real number.  <td> inf <td> The maximum allowed trust region radius.
      <tr><td> AcceptRatio <td> Optional <td> double <td> \f$[0,\eta_2]\f$ <td> 0.05 <td> Threshold \f$\eta_1\f$ on approximation quality \f$\rho\f$ needed to accept the subproblem solution as the next step.  If \f$\rho > \eta_1\f$, then \f$x_{k+1} = x_k + p_k\f$.
      <tr><td> ShrinkRatio <td> Optional <td> double <td> \f$[0,\infty)\f$ <td> 0.25 <td> Threshold \f$\eta_2\f$ indicating when the trust region should be shrunk.  If \f$\rho < \eta_2\f$, then the trust region size is multiplied by \f$t_1\f$.
      <tr><td> GrowRatio <td> Optional <td> double <td> \f$[0,\infty)\f$ <td> 0.75 <td> Threshold \f$\eta_3\f$ indicating when the trust region should be grown.  If \f$\rho>\eta_3\f$, then the trust region size is multiplied by \f$t_2\f$.
      <tr><td> ShrinkRate <td> Optional <td> double <td> \f$(0,1)\f$ <td> 0.25 <td> Multiplier \f$t_1\f$ used to shrink the trust region size: \f$\Delta_{k+1} = t_1 \Delta_k\f$.
      <tr><td> GrowRate <td> Optional <td> double <td> \f$(1,\infty)\f$ <td> 2.0 <td> Multiplier \f$t_2\f$ used to grow the trust region size: \f$\Delta_{k+1} = min(t_2 \Delta_k, \Delta_{max})\f$.
      <tr><td> TrustTol <td> Optional <td> double <td> Positive real number. <td> min(1e-4, xtol_abs) <td> Termination tolerance for Steihaug-CG solver in quadratic subproblem.  If the norm of the subproblem gradient is less than this value, the Steihaug solver terminates.
      */
      NewtonTrust(std::shared_ptr<muq::Modeling::ModPiece> const& cost,
                  boost::property_tree::ptree              const& pt);

      virtual ~NewtonTrust() = default;

      virtual std::pair<Eigen::VectorXd, double> Solve(std::vector<Eigen::VectorXd> const& inputs) override;


    private:

      // Applies a preconditioner to s
      Eigen::VectorXd Prec(Eigen::VectorXd const& s) const;

      /** Computes the distance to the trust region boundary from the point x and in the direction d. */
      double Dist2Bndry(Eigen::VectorXd const& x, Eigen::VectorXd const& d) const;

      /** Solve the model trust region subproblem using Steihaug's method */
      Eigen::VectorXd SolveSub(double                 fval,
                               Eigen::VectorXd const& x0,
                               Eigen::VectorXd const& grad);

      double trustRadius;
      const double maxRadius;
      const double initialRadius;
      const double acceptRatio;
      const double shrinkRatio;
      const double growRatio;
      const double shrinkRate;
      const double growRate;
      const double trustTol;
      const unsigned int printLevel;

    };// class NewtonTrust

  }
}

#endif
