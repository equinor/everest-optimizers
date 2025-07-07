#include "MUQ/SamplingAlgorithms/ParallelFixedSamplesMIMCMC.h"

#include "MUQ/SamplingAlgorithms/SamplingState.h"

using namespace muq::SamplingAlgorithms;


void RoundRobinStaticLoadBalancer::setup(std::shared_ptr<ParallelizableMIComponentFactory> componentFactory, uint availableRanks) 
{
    ranks_remaining = availableRanks;
    spdlog::info("Balancing load across {} ranks", availableRanks);
    auto indices = MultiIndexFactory::CreateFullTensor(componentFactory->FinestIndex()->GetVector());
    models_remaining = indices->Size();
}

int RoundRobinStaticLoadBalancer::numCollectors(std::shared_ptr<MultiIndex> modelIndex) 
{
    ranks_remaining--;
    return 1;
}

StaticLoadBalancer::WorkerAssignment RoundRobinStaticLoadBalancer::numWorkers(std::shared_ptr<MultiIndex> modelIndex)
{
    WorkerAssignment assignment;
    assignment.numWorkersPerGroup = 1;
    assignment.numGroups = ranks_remaining / models_remaining;

    spdlog::debug("Of {}, assigning {} to model {}", ranks_remaining, assignment.numGroups * assignment.numWorkersPerGroup, *modelIndex);

    assert (assignment.numGroups * assignment.numWorkersPerGroup > 0);

    models_remaining--;
    ranks_remaining -= assignment.numGroups * assignment.numWorkersPerGroup;

    return assignment;
}

StaticLoadBalancingMIMCMC::StaticLoadBalancingMIMCMC (pt::ptree pt,
                                                      std::shared_ptr<ParallelizableMIComponentFactory> componentFactory,
                                                      std::shared_ptr<StaticLoadBalancer> loadBalancing,
                                                      std::shared_ptr<parcer::Communicator> comm,
                                                      std::shared_ptr<muq::Utilities::OTF2TracerBase> tracer)
                                                            : SamplingAlgorithm(nullptr),
                                                                pt(pt),
                                                                comm(comm),
                                                                componentFactory(componentFactory),
                                                                phonebookClient(std::make_shared<PhonebookClient>(comm, phonebookRank)),
                                                                workerClient(comm, phonebookClient, rootRank) 
{

    spdlog::debug("Rank: {}", comm->GetRank());

    if (comm->GetRank() == rootRank) {

        auto comm_self = std::make_shared<parcer::Communicator>(MPI_COMM_SELF);
        componentFactory->SetComm(comm_self);

        auto indices = MultiIndexFactory::CreateFullTensor(componentFactory->FinestIndex()->GetVector());

        assert(comm->GetSize() - 2 >= 0);
        loadBalancing->setup(componentFactory, comm->GetSize() - 2);

        int rank = 2;

        // Assign collectors
        spdlog::trace("Assigning collectors");
        for (int i = 0; i < indices->Size(); i++) {
        std::shared_ptr<MultiIndex> index = (*indices)[i];
        std::vector<int> collectorRanks;
        int numCollectors = loadBalancing->numCollectors(index);
        for (int r = 0; r < numCollectors; r++) {
            collectorRanks.push_back(rank);
            rank++;
        }
        collectorClients.push_back(CollectorClient(comm, collectorRanks, index));
        }

        // Assign workers
        spdlog::trace("Assigning workers");
        for (int i = 0; i < indices->Size(); i++) {

        std::shared_ptr<MultiIndex> index = (*indices)[i];
        StaticLoadBalancer::WorkerAssignment assignment = loadBalancing->numWorkers(index);

        for (int group = 0; group < assignment.numGroups; group++) {
            std::vector<int> groupRanks;
            for (int r = 0; r < assignment.numWorkersPerGroup; r++) {
                groupRanks.push_back(rank);
                rank++;
            }
            workerClient.assignGroup(groupRanks, index);
        }

        assert (rank <= comm->GetSize());
        }


    } else if (comm->GetRank() == phonebookRank) {
        PhonebookServer phonebook(comm, pt.get<bool>("MLMCMC.Scheduling"), tracer);
        phonebook.Run();
    } else {
        auto phonebookClient = std::make_shared<PhonebookClient>(comm, phonebookRank);
        WorkerServer worker(pt, comm, phonebookClient, rootRank, componentFactory, tracer);
    }

}

std::shared_ptr<SampleCollection> StaticLoadBalancingMIMCMC::GetSamples() const
{ 
    return nullptr; 
};

std::shared_ptr<SampleCollection> StaticLoadBalancingMIMCMC::GetQOIs() const
{ 
    return nullptr; 
};

      
Eigen::VectorXd StaticLoadBalancingMIMCMC::MeanQOI() 
{
    if (comm->GetRank() != rootRank) {
        return Eigen::VectorXd::Zero(1);
    }

    for (CollectorClient& client : collectorClients) {
        client.ComputeMeans();
    }

    while (true) {
        MPI_Status status;
        ControlFlag command = comm->Recv<ControlFlag>(MPI_ANY_SOURCE, ControlTag, &status);

        for (CollectorClient& client : collectorClients) {
            if (client.Receive(command, status))
                break;
        }

        bool isComputingMeans = false;
        for (CollectorClient& client : collectorClients) {
            isComputingMeans = isComputingMeans || client.IsComputingMeans();
        }
        if (!isComputingMeans)
            break;
    }
    spdlog::info("Computing means completed");


    Eigen::VectorXd mean_box = collectorClients[0].GetQOIMean();
    mean_box.setZero();
    Eigen::VectorXd mean = mean_box;

    for (CollectorClient& client : collectorClients) {
        mean_box = client.GetQOIMean();
        mean += mean_box;
        //std::cout << "Mean level:\t" << mean_box.transpose() << " adding up to:\t" << mean.transpose() << std::endl;
    }
    return mean;
}

void StaticLoadBalancingMIMCMC::Finalize() 
{
    if (comm->GetRank() == rootRank) {
        phonebookClient->SchedulingStop();
        std::cout << "Starting unassign sequence" << std::endl;
        for (CollectorClient& client : collectorClients) {
        client.Unassign();
        }
        workerClient.UnassignAll();
        std::cout << "Finished unassign sequence" << std::endl;

        workerClient.Finalize();
        std::cout << "Rank " << comm->GetRank() << " quit" << std::endl;
    }
}

void StaticLoadBalancingMIMCMC::RequestSamples(std::shared_ptr<MultiIndex> index, int numSamples) 
{
    if (comm->GetRank() != rootRank) {
        return;
    }
    for (CollectorClient& client : collectorClients) {
        client.GetModelIndex();
        if (client.GetModelIndex() == index) {
        client.CollectSamples(numSamples);
        return;
        }
    }
    std::cerr << "Requested samples from nonexisting collector!" << std::endl;
}

void StaticLoadBalancingMIMCMC::RequestSamplesAll(int numSamples) 
{
    if (comm->GetRank() != rootRank) {
        return;
    }
    // TODO: Get indices from collectors, then request samples for each index
    for (CollectorClient& client : collectorClients) {
        client.CollectSamples(numSamples);
    }
}

      
void StaticLoadBalancingMIMCMC::RunSamples() 
{
    if (comm->GetRank() != rootRank) {
        return;
    }
    while (true) {
        MPI_Status status;
        ControlFlag command = comm->Recv<ControlFlag>(MPI_ANY_SOURCE, ControlTag, &status);

        bool command_handled = false;
        for (CollectorClient& client : collectorClients) {
            if (client.Receive(command, status)) {
                command_handled = true;
                break;
            }
        }

        if (!command_handled) {
            if (command == ControlFlag::SCHEDULING_NEEDED) {
                spdlog::debug("SCHEDULING_NEEDED entered!");
                // TODO: Phonebook client receive analog zu CollectorClient statt manuellem Empangen!
                auto idle_index = std::make_shared<MultiIndex>(comm->Recv<MultiIndex>(status.MPI_SOURCE, ControlTag));
                int rescheduleRank = comm->Recv<int>(status.MPI_SOURCE, ControlTag);
                auto busy_index = std::make_shared<MultiIndex>(comm->Recv<MultiIndex>(status.MPI_SOURCE, ControlTag));

                spdlog::debug("SCHEDULING_NEEDED Unassigning {}!", rescheduleRank);
                std::vector<int> groupMembers = workerClient.UnassignGroup(idle_index, rescheduleRank);

                workerClient.assignGroup(groupMembers, busy_index);

                phonebookClient->SchedulingDone();
                spdlog::debug("SCHEDULING_NEEDED left!");
                command_handled = true;
            }
        }


        if (!command_handled) {
        std::cerr << "Unexpected command!" << std::endl;
        exit(43);
        }

        bool isSampling = false;
        for (CollectorClient& client : collectorClients) {
        isSampling = isSampling || client.IsSampling();
        }
        if (!isSampling)
        break;
    }
    spdlog::debug("Sampling completed");
}

void StaticLoadBalancingMIMCMC::WriteToFile(std::string filename) 
{
    if (comm->GetRank() != rootRank) {
        return;
    }
    for (CollectorClient& client : collectorClients) {
        client.WriteToFile(filename);
    }
}

std::shared_ptr<SampleCollection> StaticLoadBalancingMIMCMC::RunImpl(std::vector<Eigen::VectorXd> const& x0) 
{
    for (CollectorClient& client : collectorClients) {
        int numSamples = pt.get<int>("NumSamples" + multiindexToConfigString(client.GetModelIndex()));
        client.CollectSamples(numSamples);
    }

    RunSamples();
    return nullptr;
}

std::string StaticLoadBalancingMIMCMC::multiindexToConfigString(std::shared_ptr<MultiIndex> index) 
{
    std::stringstream strs;
    for (int i = 0; i < index->GetLength(); i++) {
        strs << "_" << index->GetValue(i);
    }
    return strs.str();
}