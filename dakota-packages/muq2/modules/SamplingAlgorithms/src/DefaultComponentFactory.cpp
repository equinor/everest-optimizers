#include "MUQ/SamplingAlgorithms/DefaultComponentFactory.h"
#include "MUQ/SamplingAlgorithms/SubsamplingMIProposal.h"
#include "MUQ/SamplingAlgorithms/ConcatenatingInterpolation.h"

#include "MUQ/Utilities/MultiIndices/MultiIndexFactory.h"

#include <boost/property_tree/ptree.hpp>

using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

DefaultComponentFactory::DefaultComponentFactory(boost::property_tree::ptree pt, 
                                                 Eigen::VectorXd startingPoint, 
                                                 std::vector<std::shared_ptr<AbstractSamplingProblem>> const& problemsIn)
                        : DefaultComponentFactory(pt, startingPoint, MultiIndexFactory::CreateFullTensor(1,problemsIn.size() - 1), problemsIn)
{

}

DefaultComponentFactory::DefaultComponentFactory(boost::property_tree::ptree optionsIn, 
                                                 Eigen::VectorXd startingPointIn, 
                                                 std::shared_ptr<MultiIndexSet> const& problemIndicesIn, 
                                                 std::vector<std::shared_ptr<AbstractSamplingProblem>> const& problemsIn)
                        : options(optionsIn),
                          startingPoint(startingPointIn),
                          problemIndices(problemIndicesIn),
                          problems(problemsIn)
  {
  }


std::shared_ptr<MCMCProposal> DefaultComponentFactory::Proposal(std::shared_ptr<MultiIndex> const& index,
                                                                std::shared_ptr<AbstractSamplingProblem> const& problem) 
{

    boost::property_tree::ptree subTree = options.get_child("Proposal");
    subTree.put("BlockIndex",0);

    // Construct the proposal
    std::shared_ptr<MCMCProposal> proposal = MCMCProposal::Construct(subTree, problem);
    assert(proposal);
    return proposal;
}

std::shared_ptr<MultiIndex> DefaultComponentFactory::FinestIndex()
{
    return std::make_shared<MultiIndex>(problemIndices->GetMaxOrders());
}

std::shared_ptr<MCMCProposal> DefaultComponentFactory::CoarseProposal (std::shared_ptr<MultiIndex> const& fineIndex,
                                                                       std::shared_ptr<MultiIndex> const& coarseIndex,
                                                                       std::shared_ptr<AbstractSamplingProblem> const& coarseProblem,
                                                                       std::shared_ptr<SingleChainMCMC> const& coarseChain)
{
    boost::property_tree::ptree ptProposal = options;
    ptProposal.put("BlockIndex",0);
    return std::make_shared<SubsamplingMIProposal>(ptProposal, coarseProblem, coarseIndex, coarseChain);
}

std::shared_ptr<AbstractSamplingProblem> DefaultComponentFactory::SamplingProblem (std::shared_ptr<MultiIndex> const& index)
{
    for(int i = 0; i < problemIndices->Size(); i++) {
        if (*(problemIndices->at(i)) == *index)
            return problems.at(i);
    }

    std::cout << "Undefined problem! " << *index << std::endl;
    assert(false);
    return nullptr;
}

std::shared_ptr<MIInterpolation> DefaultComponentFactory::Interpolation (std::shared_ptr<MultiIndex> const& index)
{
    return std::make_shared<ConcatenatingInterpolation>(index);
}

Eigen::VectorXd DefaultComponentFactory::StartingPoint (std::shared_ptr<MultiIndex> const& index)
{
    return startingPoint;
}