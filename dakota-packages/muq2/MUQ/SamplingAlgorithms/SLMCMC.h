#ifndef SLMCMC_H
#define SLMCMC_H

#include <boost/property_tree/ptree.hpp>

#include "MUQ/SamplingAlgorithms/MIMCMCBox.h"
#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"

#include "MUQ/SamplingAlgorithms/MarkovChain.h"

namespace pt = boost::property_tree;

namespace muq {
  namespace SamplingAlgorithms {

    /** @brief Single-level MCMC for multiindex sampling problems.
        @details A wrapper generating a single-chain MCMC
        based on the finest problem of a multiindex sampling problem.
        This is mostly needed for computing reference solutions to
        multilevel/multiindex MCMC methods.
    */

    class SLMCMC {

    public:
      SLMCMC (pt::ptree pt, std::shared_ptr<MIComponentFactory> componentFactory, std::shared_ptr<MultiIndex> index);
      SLMCMC (pt::ptree pt, std::shared_ptr<MIComponentFactory> componentFactory);

      virtual std::shared_ptr<MarkovChain> GetSamples() const;
      virtual std::shared_ptr<MarkovChain> GetQOIs() const;

      Eigen::VectorXd MeanQOI();

      Eigen::VectorXd MeanParameter();
      
      void WriteToFile(std::string filename);

      virtual std::shared_ptr<MarkovChain> Run();

    protected:
      

    private:
      std::shared_ptr<MIComponentFactory> componentFactory;
      std::shared_ptr<SingleChainMCMC> single_chain;
    };

  }
}

#endif
