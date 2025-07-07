#ifndef MIMCMC_H
#define MIMCMC_H

#include <boost/property_tree/ptree.hpp>

#include "MUQ/SamplingAlgorithms/MIMCMCBox.h"
#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"
#include "MUQ/SamplingAlgorithms/MultiIndexEstimator.h"

namespace pt = boost::property_tree;


namespace muq {
  namespace SamplingAlgorithms {

    /** @brief Multiindex MCMC method.
        @ingroup MIMCMC
        @details A basic MIMCMC method based on a fixed
        number of samples for all model indices.
     */
    class MIMCMC {
    public:
      MIMCMC(boost::property_tree::ptree options,
             std::shared_ptr<MIComponentFactory> const& componentFactory);

      MIMCMC(boost::property_tree::ptree                                  pt, 
             Eigen::VectorXd                                       const& startPt,
             std::vector<std::shared_ptr<muq::Modeling::ModPiece>> const& models,
             std::shared_ptr<MultiIndexSet>                        const& multis = nullptr);
      
      MIMCMC(boost::property_tree::ptree                                  pt, 
             Eigen::VectorXd                                       const& startPt,
             std::vector<std::shared_ptr<AbstractSamplingProblem>> const& problems,
             std::shared_ptr<MultiIndexSet>                        const& multis = nullptr);

      virtual std::shared_ptr<MultiIndexEstimator> GetSamples() const;
      virtual std::shared_ptr<MultiIndexEstimator> GetQOIs() const;

      /**
       * @brief Access an MIMCMCBox
       *
       * @param index Model index for which to retrieve the box.
       * @return std::shared_ptr<MIMCMCBox> The MIMCMCBox representing the Multiindex telescoping sum component
       * associated with that model.
       */
      std::shared_ptr<MIMCMCBox> GetBox(std::shared_ptr<MultiIndex> index);

      /**
       * @brief Draw MI structure (mostly debugging purposes)
       */
      void Draw(bool drawSamples = true);

      std::shared_ptr<MIMCMCBox> GetMIMCMCBox(std::shared_ptr<MultiIndex> index);

      /**
       * @brief Get set of indices of boxes set up by the method.
       */
      std::shared_ptr<MultiIndexSet> GetIndices();

      virtual std::shared_ptr<MultiIndexEstimator> Run();

      /**
       * @brief Write HDF5 output for the entire MIMCMC method
       */
      void WriteToFile(std::string filename);

    private:
      pt::ptree pt;
      std::shared_ptr<MultiIndexSet> gridIndices;
      std::shared_ptr<MIComponentFactory> componentFactory;
      std::vector<std::shared_ptr<MIMCMCBox>> boxes;

      std::string multiindexToConfigString (std::shared_ptr<MultiIndex> index);

      static std::vector<std::shared_ptr<AbstractSamplingProblem>> CreateProblems(std::vector<std::shared_ptr<muq::Modeling::ModPiece>> const& models);
      static std::shared_ptr<MultiIndexSet> ProcessMultis(std::shared_ptr<MultiIndexSet> const& multis, unsigned int numLevels);
    };

  }
}

#endif
