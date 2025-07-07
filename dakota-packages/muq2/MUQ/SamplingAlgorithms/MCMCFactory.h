#ifndef MCMCFACTORY_H_
#define MCMCFACTORY_H_

#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"


namespace muq {
  namespace SamplingAlgorithms {

    class MCMCFactory {

    public:

      static std::shared_ptr<SingleChainMCMC>
      CreateSingleChain(boost::property_tree::ptree             pt,
                        std::shared_ptr<AbstractSamplingProblem> problem);

    };

  } // namespace SamplingAlgoirthms
} // namespace muq

#endif
