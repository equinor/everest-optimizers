#ifndef PARALLELMIMCMCWORKER_H_
#define PARALLELMIMCMCWORKER_H_

#include "MUQ/config.h"

#if MUQ_HAS_MPI

#if !MUQ_HAS_PARCER
#error
#endif

#include <chrono>
#include <list>
#include <thread>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"
#include "MUQ/SamplingAlgorithms/MarkovChain.h"
#include "MUQ/SamplingAlgorithms/DistributedCollection.h"
#include "MUQ/SamplingAlgorithms/ParallelFlags.h"
#include "MUQ/SamplingAlgorithms/ParallelizableMIComponentFactory.h"
#include "MUQ/SamplingAlgorithms/ParallelMIComponentFactory.h"
#include "MUQ/SamplingAlgorithms/ParallelMIMCMCBox.h"
#include "MUQ/Utilities/AnyHelpers.h"
#include "MUQ/Utilities/Cereal/MultiIndexSerializer.h"

namespace muq {
  namespace SamplingAlgorithms {

    /**
     * @brief High-level communication wrapper for controlling SampleCollectors.
     * @details This takes care about assigning workers to a set of collectors for a specific
     * model index/level, sending commands to them and finally unassigning them again.
     */
    class CollectorClient {
    public:
      CollectorClient(std::shared_ptr<parcer::Communicator> comm, 
                      std::vector<int> subgroup, 
                      std::shared_ptr<MultiIndex> modelindex);

      std::shared_ptr<MultiIndex> GetModelIndex() const;

      void CollectSamples (int numSamples);

      void ComputeMeans();

      void WriteToFile(std::string filename);

      bool Receive (ControlFlag command, const MPI_Status& status);

      Eigen::VectorXd GetQOIMean();

      void Unassign();

      bool IsSampling();

      bool IsComputingMeans();

    private:

      std::shared_ptr<parcer::Communicator> comm;
      std::vector<int> subgroup;

      bool sampling = false;
      bool computingMeans = false;
      std::shared_ptr<MultiIndexSet> boxIndices;
      Eigen::VectorXd boxQOIMean;

      std::shared_ptr<MultiIndex> boxHighestIndex;
      std::shared_ptr<MultiIndex> boxLowestIndex;

      std::map<std::shared_ptr<MultiIndex>, Eigen::VectorXd, MultiPtrComp> means;

    };

    /**
     * @brief High-level communication wrapper for controlling worker processes.
     * @details This takes care about assigning workers to a worker group for a specific
     * model index/level, sending commands to them and finally unassigning them again.
     */
    class WorkerClient {
    public:
      WorkerClient(std::shared_ptr<parcer::Communicator> comm, 
                   std::shared_ptr<PhonebookClient> phonebookClient,
                  int RootRank);

      void assignGroup (std::vector<int> subgroup, std::shared_ptr<MultiIndex> modelindex);

      std::vector<int> UnassignGroup (std::shared_ptr<MultiIndex> modelIndex, int groupRootRank);

      void UnassignAll();

      void Finalize();

    private:
      std::shared_ptr<parcer::Communicator> comm;
      std::shared_ptr<PhonebookClient> phonebookClient;
    };

    /**
     * @brief Implements the actual sampling / collecting logic for parallel MIMCMC.
     * @details Workers will, in a loop until finalized, wait for instructions to
     * join a set of collectors or a worker group for sampling. They then listen
     * for commands to execute these tasks until unassigned from that task again.
     * As a collector, workers will request MCMC samples (more specifically,
     * samples and their coarser ancesters in analogy to the sequential MIMCMCBox)
     * from sampling worker groups. As part of a sampling worker group, they will
     * compute MCMC samples and provide them to other processes via the phonebook.
     */
    class WorkerServer {
    public:
      WorkerServer(boost::property_tree::ptree const& pt, 
                   std::shared_ptr<parcer::Communicator> comm, 
                   std::shared_ptr<PhonebookClient> phonebookClient, 
                   int RootRank, 
                   std::shared_ptr<ParallelizableMIComponentFactory> componentFactory, 
                   std::shared_ptr<muq::Utilities::OTF2TracerBase> tracer);

    private:
      std::string multiindexToConfigString (std::shared_ptr<MultiIndex> index);
      
    };

  }
}

#endif

#endif
