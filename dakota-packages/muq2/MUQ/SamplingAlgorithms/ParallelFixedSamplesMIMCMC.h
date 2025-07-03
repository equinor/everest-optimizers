#ifndef PARALLELFIXEDSAMPLESMIMCMC_H_
#define PARALLELFIXEDSAMPLESMIMCMC_H_

#include "MUQ/config.h"

#if MUQ_HAS_MPI

#if !MUQ_HAS_PARCER
#error
#endif

#include "spdlog/spdlog.h"
#include <boost/property_tree/ptree.hpp>

#include "MUQ/SamplingAlgorithms/ParallelMIMCMCWorker.h"
#include "MUQ/SamplingAlgorithms/ParallelizableMIComponentFactory.h"
#include "MUQ/SamplingAlgorithms/SamplingAlgorithm.h"

namespace muq {
  namespace SamplingAlgorithms {

    /**
     * @brief Base class handling the static load balancing of parallel MIMCMC.
     * @details The user can implement a custom load balancing strategy and pass it to the
     * parallel MIMCMC method.
     */
    class StaticLoadBalancer {
    public:
      struct WorkerAssignment {
        int numGroups;
        int numWorkersPerGroup;
      };

      virtual void setup(std::shared_ptr<ParallelizableMIComponentFactory> componentFactory, uint availableRanks) = 0;

      /**
       * @brief Number of collector processes assigned to a model index.
       */
      virtual int numCollectors(std::shared_ptr<MultiIndex> modelIndex) = 0;

      /**
       * @brief Number of worker groups and number of workers per group for a given model index.
       */
      virtual WorkerAssignment numWorkers(std::shared_ptr<MultiIndex> modelIndex) = 0;
    };

    /**
     * @brief Simple default static load balancing strategy suitable for many cases.
     *
     * @details This load balancing strategy assigns one collector rank for each model
     * index, one worker per group and proceeds to evenly split groups across indices.
     * It clearly makes no further assumptions on the model, and would best be used
     * together with dynamic load balancing during runtime.
     */
    class RoundRobinStaticLoadBalancer : public StaticLoadBalancer {
    public:
      void setup(std::shared_ptr<ParallelizableMIComponentFactory> componentFactory, uint availableRanks) override;

      int numCollectors(std::shared_ptr<MultiIndex> modelIndex) override;

      WorkerAssignment numWorkers(std::shared_ptr<MultiIndex> modelIndex) override;

    private:
      uint ranks_remaining;
      uint models_remaining;

    };

    /**
     * @brief A parallel MIMCMC method.
     * @details This parallelized MIMCMC method begins by assigning tasks to processes according
     * to a StaticLoadBalancer. It then proceeds to collect a pre-defined number of samples per level.
     * The main result is a mean estimate quantity of interest, computed via a telescoping sum across
     * model indices. When applied to one-dimensional multiindices, this is equivalent to a MLMCMC method.
     *
     * Optionally, more control can be taken: For example, samples can be analyzed on the fly and possibly additional
     * samples requested in order to adaptively ensure high-quality estimates and optimize computational cost.
     * To the same end, dynamic scheduling can be activated in the case of same-size work groups, which optimizes
     * machine utilization by reassigning processes from less busy model indices to ones with higher load.
     */
    class StaticLoadBalancingMIMCMC : public SamplingAlgorithm {
    public:
      StaticLoadBalancingMIMCMC (pt::ptree pt,
                                 std::shared_ptr<ParallelizableMIComponentFactory> componentFactory,
                                 std::shared_ptr<StaticLoadBalancer> loadBalancing = std::make_shared<RoundRobinStaticLoadBalancer>(),
                                 std::shared_ptr<parcer::Communicator> comm = std::make_shared<parcer::Communicator>(MPI_COMM_WORLD),
                                 std::shared_ptr<muq::Utilities::OTF2TracerBase> tracer = std::make_shared<OTF2TracerDummy>());

      /**
       * @brief Get mean quantity of interest estimate computed via telescoping sum.
       */
      Eigen::VectorXd MeanQOI();

      /**
       * @brief Dummy implementation; required by interface, has no meaning in ML/MI setting.
       */
      virtual std::shared_ptr<SampleCollection> GetSamples() const;

      /**
       * @brief Dummy implementation; required by interface, has no meaning in ML/MI setting.
       */
      virtual std::shared_ptr<SampleCollection> GetQOIs() const;

      /**
       * @brief Cleanup parallel method, wait for all ranks to finish.
       *
       */
      void Finalize();

      /**
       * @brief Request additional samples to be compute for a given model index.
       */
      void RequestSamples(std::shared_ptr<MultiIndex> index, int numSamples);

      /**
       * @brief Request an additional number of samples to be computed on each level.
       */
      void RequestSamplesAll(int numSamples);

      /**
       * @brief Run the parallel method.
       *
       * @details Note that this internally also handles reassigning tasks for scheduling purposes.
       * The phonebook cannot be responsible for this, since the phonebook itself needs to be available
       * during the reassignment process.
       */
      void RunSamples();

      void WriteToFile(std::string filename);


    protected:
      virtual std::shared_ptr<SampleCollection> RunImpl(std::vector<Eigen::VectorXd> const& x0);

    private:

      std::string multiindexToConfigString (std::shared_ptr<MultiIndex> index);

      const int rootRank = 0;
      const int phonebookRank = 1;
      pt::ptree pt;
      std::shared_ptr<parcer::Communicator> comm;
      std::shared_ptr<ParallelizableMIComponentFactory> componentFactory;
      std::shared_ptr<PhonebookClient> phonebookClient;
      std::vector<CollectorClient> collectorClients;
      WorkerClient workerClient;
    };
  }
}

#endif // #if MUQ_HAS_MPI
#endif // #ifndef PARALLELFIXEDSAMPLESMIMCMC_H_
