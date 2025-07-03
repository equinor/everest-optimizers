#include "MUQ/SamplingAlgorithms/SLMCMC.h"

namespace muq {
  namespace SamplingAlgorithms {

    SLMCMC::SLMCMC (pt::ptree pt, std::shared_ptr<MIComponentFactory> componentFactory, std::shared_ptr<MultiIndex> index)
     : componentFactory(componentFactory)
    {
      auto finestIndex = componentFactory->FinestIndex(); 
      
      assert(index->GetLength() == finestIndex->GetLength());
      assert(*index <= *(componentFactory->FinestIndex()));

      pt::ptree ptBlockID;
      ptBlockID.put("BlockIndex",0);
      
      auto problem = componentFactory->SamplingProblem(index);
      auto proposal = componentFactory->Proposal(index, problem);

      std::vector<std::shared_ptr<TransitionKernel>> kernels(1);
      kernels[0] = std::make_shared<MHKernel>(ptBlockID,problem,proposal);
      
      Eigen::VectorXd startingPoint = componentFactory->StartingPoint(index);

      single_chain = std::make_shared<SingleChainMCMC>(pt,kernels);
      single_chain->SetState(startingPoint);
    }
    
    SLMCMC::SLMCMC (pt::ptree pt, std::shared_ptr<MIComponentFactory> componentFactory)
     : SLMCMC(pt,componentFactory, componentFactory->FinestIndex()) { }

    std::shared_ptr<MarkovChain> SLMCMC::GetSamples() const {
      return single_chain->GetSamples();
    }
    std::shared_ptr<MarkovChain> SLMCMC::GetQOIs() const {
      return single_chain->GetQOIs();
    }

    std::shared_ptr<MarkovChain> SLMCMC::Run() {
      return single_chain->Run();
    }
    
    void SLMCMC::WriteToFile(std::string filename){
        auto samps = single_chain->GetSamples();
        auto QOI = single_chain->GetQOIs();
        if(QOI != nullptr)
          QOI->WriteToFile(filename,"/qois");
        samps->WriteToFile(filename,"/samples");
    }
    
  }
}
