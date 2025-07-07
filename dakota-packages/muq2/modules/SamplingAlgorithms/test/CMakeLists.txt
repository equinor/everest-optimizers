CreateTestGroup(
    SAMPLING_ALGORITHM

    SampleCollectionTests.cpp
    SampleEstimatorTest.cpp
    MarkovChainTests.cpp
    MIMCMCTests.cpp
    MLMCMCTests.cpp

    GMHKernelTests.cpp

    #MonteCarloTests.cpp
    ImportanceSamplingTests.cpp
    MCMCTests.cpp
    DelayedRejectionTests.cpp
    DILITests.cpp
    CrankNicolsonTests.cpp
    InfMALATests.cpp
    InverseGammaProposalTests.cpp
    IndependenceProposalTests.cpp
    ParallelTemperingTests.cpp
)

CreateTestGroup(
  SAMPLING_PROBLEMS

  SamplingProblemTests.cpp
  InferenceProblemTests.cpp
)

CreateTestGroup(
  FANCY_SAMPLING_PROBLEMS

  ExpensiveSamplingProblemTests.cpp
)

CreateParallelTestGroup(
    SAMPLING_ALGORITHM

    SamplingStateParallelTests.cpp
    DistributedCollectionTests.cpp
    GMHKernelParallelTests.cpp
    SingleChainMCMCParallelTests.cpp
    ExpensiveSamplingProblemParallelTests.cpp
    ImportanceSamplingParallelTests.cpp
)
