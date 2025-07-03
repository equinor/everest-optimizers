#ifndef DEFAULTCOMPONENTFACTORY_H
#define DEFAULTCOMPONENTFACTORY_H

#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"
#include "MUQ/SamplingAlgorithms/MCMCProposal.h"

#include <boost/property_tree/ptree_fwd.hpp>

namespace muq{
namespace SamplingAlgorithms{


/** @class DefaultComponentFactory
    @ingroup MIMCMC
    @brief Provides a high level interface for the sampling problems on each MIMCMC level.
*/
class DefaultComponentFactory : public MIComponentFactory {
public:

  DefaultComponentFactory(boost::property_tree::ptree options,
                          Eigen::VectorXd startingPoint,
                          std::vector<std::shared_ptr<AbstractSamplingProblem>> const& problems);
                          
  DefaultComponentFactory(boost::property_tree::ptree options, 
                          Eigen::VectorXd startingPoint, 
                          std::shared_ptr<MultiIndexSet> const& problemIndices,
                          std::vector<std::shared_ptr<AbstractSamplingProblem>> const& problems);

  virtual ~DefaultComponentFactory() = default;

  virtual std::shared_ptr<MCMCProposal> Proposal (std::shared_ptr<MultiIndex> const& index, 
                                                  std::shared_ptr<AbstractSamplingProblem> const& samplingProblem) override ;

  virtual std::shared_ptr<MultiIndex> FinestIndex() override;

  virtual std::shared_ptr<MCMCProposal> CoarseProposal (std::shared_ptr<MultiIndex> const& fineIndex,
                                                        std::shared_ptr<MultiIndex> const& coarseIndex,
                                                        std::shared_ptr<AbstractSamplingProblem> const& coarseProblem,
                                                        std::shared_ptr<SingleChainMCMC> const& coarseChain) override;

  virtual std::shared_ptr<AbstractSamplingProblem> SamplingProblem (std::shared_ptr<MultiIndex> const& index) override;

  virtual std::shared_ptr<MIInterpolation> Interpolation (std::shared_ptr<MultiIndex> const& index) override;

  virtual Eigen::VectorXd StartingPoint (std::shared_ptr<MultiIndex> const& index) override;
  
private:
  boost::property_tree::ptree options;
  Eigen::VectorXd startingPoint;
  std::shared_ptr<MultiIndexSet> problemIndices;
  std::vector<std::shared_ptr<AbstractSamplingProblem>> problems;
};

}
}



#endif // #ifndef DEFAULTCOMPONENTFACTORY_H