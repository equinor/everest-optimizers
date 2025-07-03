#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <boost/property_tree/ptree.hpp>

#include "MUQ/Modeling/ModPiece.h"
#include "MUQ/Optimization/ModPieceCostFunction.h"

#include "MUQ/Utilities/RegisterClassName.h"

namespace muq {
namespace Optimization {
  /// Solve an optimization problem
  /**
     \f{eqnarray}{
     c &=& \min{J(x; \theta_1, ..., \theta_1)} \        \
     f_i(x) &\leq& 0 \                                  \
     g_i(x) &=& 0
     \f}
  */
  class Optimizer : public muq::Modeling::WorkPiece {
  public:

    Optimizer(std::shared_ptr<muq::Modeling::ModPiece> const& cost,
              boost::property_tree::ptree              const& pt);



    virtual ~Optimizer() = default;

    /** Constructs an optimizer given a cost function and ptree of options.

        <h3>Options:</h3>
        <table>
        <tr><th>Option Key <th> Optional/Required <th> Type <th> Possible Values <th> Default <th> Description
        <tr><td> Method <td> Required <td> string <td> Many.  Typically the name of the Optimizer child or the name of an NLOPT algorithm, but run Optimizer::ListMethod() for a complete list. <td> -- <td> Specifies the optimization method to use.
        <tr><td> Ftol.AbsoluteTolerance <td> Optional <td> double <td> Any nonnegative real number. <td> 1e-8 <td> Termination criterion based on value of function value.  Note: this may not be used by all algorithms.
        <tr><td> Ftol.RelativeTolerance <td> Optional <td> double <td> Any nonnegative real number. <td> 1e-8 <td> Termination criterion based on the relative change in the function value.  Note: this may not be used by all algorithms.
        <tr><td> Xtol.AbsoluteTolerance <td> Optional <td> double <td> Any nonnegative real number. <td> 1e-8 <td> Termination criterion based on the change of optimization variables or gradient. Note: this may not be used by all algorithms.
        <tr><td> Xtol.RelativeTolerance <td> Optional <td> double <td> Any nonnegative real number. <td> 1e-8 <td> Termination criterion based on the relative change of optimization variables or gradient. Note: this may not be used by all algorithms.
        <tr><td> ConstraintTolerance <td> Optional <td> double <td> Any nonnegative real number.  <td> 1e-8 <td> Tolerance on equality constraints.
        <tr><td> MaxEvaluations <td> Optional <td> unsigned int <td> Any natural number. <td> 100 <td> Depending on the algorithm, either the number of cost function evaluations allowed or the number of optimization iterations allowed.
        </table>
    */
    static std::shared_ptr<Optimizer> Construct(std::shared_ptr<muq::Modeling::ModPiece>   const& cost,
                                                boost::property_tree::ptree                const& options);


    typedef std::function<std::shared_ptr<Optimizer>(std::shared_ptr<muq::Modeling::ModPiece> const&, boost::property_tree::ptree)> OptimizerConstructor;
    typedef std::map<std::string, OptimizerConstructor> OptimizerMap;
    static std::shared_ptr<OptimizerMap> GetOptimizerMap();

    /** Prints the list of registered optimization method to std::cout
    */
    static void ListMethods(std::string prefix="");

    /// Add an inequality constraint to the optimization
    /**
       @param[in] ineq The constraint
    */
    virtual void AddInequalityConstraint(std::vector<std::shared_ptr<muq::Modeling::ModPiece>> const& ineq);

    /// Add an inequality constraint to the optimization
    /**
       @param[in] ineq The constraint
    */
    virtual void AddInequalityConstraint(std::shared_ptr<muq::Modeling::ModPiece> const& ineq);


    /// Clear all inequality constraints
    void ClearInequalityConstraint();

    /// Add an equality constraint to the optimization
    /**
       NOTE: the NLOPT algorithm used must be able to handle equality constraints
       @param[in] ineq The constraint
    */
    virtual void AddEqualityConstraint(std::vector<std::shared_ptr<muq::Modeling::ModPiece>> const& eq);

    /// Add an equality constraint to the optimization
    /**
       NOTE: the NLOPT algorithm used must be able to handle equality constraints
       @param[in] ineq The constraint
    */
    virtual void AddEqualityConstraint(std::shared_ptr<muq::Modeling::ModPiece> const& eq);

    /// Clear all equality constraints
    void ClearEqualityConstraint();

    /// Solve the optimization problem
    /**
       @param[in] inputs The first input is the variable we are optimizing over, second input are the
                  cost function parameters, and the third input are the constraint parameters
       \return First: the argmin, second: the minimum cost
    */
    virtual std::pair<Eigen::VectorXd, double> Solve(std::vector<Eigen::VectorXd> const& inputs) = 0;

  protected:

    virtual void EvaluateImpl(muq::Modeling::ref_vector<boost::any> const& inputs) override;

    static std::shared_ptr<CostFunction> ConvertModPiece(std::shared_ptr<muq::Modeling::ModPiece> const& modCost);

    /// The cost function that we are trying to minimize
    std::shared_ptr<CostFunction> opt;

    /// Inequality constraints
    std::vector<std::shared_ptr<muq::Modeling::ModPiece>> ineqConstraints;

    /// Equality constraints
    /**
       NOTE: the solver muq::Optimization::Optimization::algorithm must be able to handle equality constraints
    */
    std::vector<std::shared_ptr<muq::Modeling::ModPiece>> eqConstraints;

    /// Relative and absolute tolerances on the cost function value and on the difference between successive values of the state
    const double ftol_rel, ftol_abs, xtol_rel, xtol_abs;

    /// Tolerance on the constraints
    const double constraint_tol;

    /// Maximum number of cost function evaluations
    const unsigned int maxEvals;
  };

} // namespace Optimization
} // namespace muq

#define REGISTER_OPTIMIZER(OPTNAME, CLASSNAME) static auto optReg ##OPTNAME = muq::Optimization::Optimizer::GetOptimizerMap()->insert(std::make_pair(#OPTNAME, muq::Utilities::shared_factory<CLASSNAME>()));



#endif
