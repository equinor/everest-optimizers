#ifndef GreedyMLMCMC_H
#define GreedyMLMCMC_H

#include <boost/property_tree/ptree.hpp>

#include "MUQ/SamplingAlgorithms/MIMCMCBox.h"
#include "MUQ/SamplingAlgorithms/MIComponentFactory.h"

#include "MUQ/SamplingAlgorithms/MultiIndexEstimator.h"

#include "MUQ/Modeling/ModPiece.h"
#include "MUQ/SamplingAlgorithms/AbstractSamplingProblem.h"
#include "MUQ/Utilities/MultiIndices/MultiIndexSet.h"

namespace muq {
  namespace SamplingAlgorithms {

    /** @brief Greedy Multilevel MCMC method.
        @details A Multilevel MCMC method choosing
        the number of samples adaptively at runtime,
        estimating the most profitable level from
        statistical information on samples.
     */
    class GreedyMLMCMC {
    public:
      GreedyMLMCMC (boost::property_tree::ptree                                  pt, 
                    Eigen::VectorXd                                       const& startPt,
                    std::vector<std::shared_ptr<muq::Modeling::ModPiece>> const& models,
                    std::shared_ptr<MultiIndexSet>                        const& multis = nullptr);
      
      GreedyMLMCMC (boost::property_tree::ptree                                  pt, 
                    Eigen::VectorXd                                       const& startPt,
                    std::vector<std::shared_ptr<AbstractSamplingProblem>> const& problems,
                    std::shared_ptr<MultiIndexSet>                        const& multis = nullptr);

      GreedyMLMCMC (boost::property_tree::ptree pt, 
                    std::shared_ptr<MIComponentFactory> componentFactory);

      virtual std::shared_ptr<MultiIndexEstimator> GetSamples() const;
      virtual std::shared_ptr<MultiIndexEstimator> GetQOIs() const;

      void Draw(bool drawSamples = true);

      std::shared_ptr<MIMCMCBox> GetBox(int index);
      std::vector<std::shared_ptr<MIMCMCBox>> GetBoxes();

      void WriteToFile(std::string filename);

      virtual std::shared_ptr<MultiIndexEstimator> Run();

    protected:
      
    private:
      static std::vector<std::shared_ptr<AbstractSamplingProblem>> CreateProblems(std::vector<std::shared_ptr<muq::Modeling::ModPiece>> const& models);
      static std::shared_ptr<MultiIndexSet> ProcessMultis(std::shared_ptr<MultiIndexSet> const& multis, unsigned int numLevels);


      std::shared_ptr<MIComponentFactory> componentFactory;
      const int numInitialSamples;
      const double e;
      const double beta;
      const int levels;
      int verbosity;
      std::vector<std::shared_ptr<MIMCMCBox>> boxes;
      bool useQOIs; // <- Whether or not the sampling problems have QOIs.  If not, the parameters themselves are used for assessing estimator variance.
    };

  }
}

#endif
