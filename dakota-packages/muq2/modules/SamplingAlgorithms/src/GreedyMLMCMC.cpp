#include "MUQ/SamplingAlgorithms/GreedyMLMCMC.h"
#include "MUQ/SamplingAlgorithms/DefaultComponentFactory.h"

#include "MUQ/Utilities/MultiIndices/MultiIndexFactory.h"

using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;
using namespace muq::Modeling;

GreedyMLMCMC::GreedyMLMCMC (boost::property_tree::ptree                                  opts, 
                            Eigen::VectorXd                                       const& startPt,
                            std::vector<std::shared_ptr<muq::Modeling::ModPiece>> const& models,
                            std::shared_ptr<MultiIndexSet>                        const& multis) : GreedyMLMCMC(opts, startPt, CreateProblems(models), ProcessMultis(multis,models.size()))
{
}
  
GreedyMLMCMC::GreedyMLMCMC (boost::property_tree::ptree                                  opts, 
                            Eigen::VectorXd                                       const& startPt,
                            std::vector<std::shared_ptr<AbstractSamplingProblem>> const& problems,
                            std::shared_ptr<MultiIndexSet>                        const& multis) : GreedyMLMCMC(opts, std::make_shared<DefaultComponentFactory>(opts,startPt,ProcessMultis(multis,problems.size()),problems))
{ 
}

GreedyMLMCMC::GreedyMLMCMC (boost::property_tree::ptree opts, std::shared_ptr<MIComponentFactory> componentFactory)
: componentFactory(componentFactory),
  numInitialSamples(opts.get("NumInitialSamples",1000)),
  e(opts.get("GreedyTargetVariance",0.1)),
  beta(opts.get("GreedyResamplingFactor",0.5)),
  levels(componentFactory->FinestIndex()->GetValue(0)),
  verbosity(opts.get("verbosity",0)),
  useQOIs(componentFactory->SamplingProblem(componentFactory->FinestIndex())->numBlocksQOI>0)
{

  for (int level = 0; level <= levels; level++) {
    if (verbosity > 0)
      std::cout << "Setting up level " << level << std::endl;

    auto boxHighestIndex = std::make_shared<MultiIndex>(1,level);
    auto box = std::make_shared<MIMCMCBox>(componentFactory, boxHighestIndex);
    boxes.push_back(box);
  }
}

std::vector<std::shared_ptr<AbstractSamplingProblem>> GreedyMLMCMC::CreateProblems(std::vector<std::shared_ptr<muq::Modeling::ModPiece>> const& models)
{
  std::vector<std::shared_ptr<AbstractSamplingProblem>> output(models.size());
  for(unsigned int i=0; i<models.size(); ++i)
    output.at(i) = std::make_shared<SamplingProblem>(models.at(i));
  
  return output;
}
      

std::shared_ptr<MultiIndexEstimator> GreedyMLMCMC::GetSamples() const {
  return std::make_shared<MultiIndexEstimator>(boxes);
}

std::shared_ptr<MultiIndexEstimator> GreedyMLMCMC::GetQOIs() const {
  return std::make_shared<MultiIndexEstimator>(boxes, true);
}

std::shared_ptr<MultiIndexEstimator> GreedyMLMCMC::Run() {

  const int levels = componentFactory->FinestIndex()->GetValue(0);

  if (verbosity > 0)
    std::cout << "Computing " << numInitialSamples << " initial samples per level" << std::endl;

  for (auto box : boxes) {
    for (int samp = 0; samp < numInitialSamples; samp++) {
      box->Sample();
    }
  }
  if (verbosity > 0)
    std::cout << "Initial samples done" << std::endl;

  while(true) {

    double var_mle = 0.0;
    for (int i = 0; i <= levels; i++) {

      std::shared_ptr<SampleCollection> chain;
      if(useQOIs){
         chain = boxes.at(i)->FinestChain()->GetQOIs();
      }else{
        chain = boxes.at(i)->FinestChain()->GetSamples();
      }
      var_mle += chain->Variance().cwiseQuotient(chain->ESS()).maxCoeff();
    }

    if (var_mle <= std::pow(e,2)) {
      if (verbosity > 0)
        std::cout << "val_mle " << var_mle << " below " << std::pow(e,2) << std::endl;
      break;
    }

    // Find level with largest payoff ratio
    int l = -1;
    double payoff_l = -1;
    for (int i = 0; i <= levels; i++) {
      std::shared_ptr<SampleCollection> chain;
      if(useQOIs){
         chain = boxes.at(i)->FinestChain()->GetQOIs();
      }else{
        chain = boxes.at(i)->FinestChain()->GetSamples();
      }
      
      double my_payoff = chain->Variance().maxCoeff() / boxes.at(i)->FinestChain()->TotalTime();

      if (my_payoff > payoff_l) {
        l = i;
        payoff_l = my_payoff;
      }
    }

    // Beta percent new samples on largest payoff level
    double weight_sum = 0.0;
    auto finestChain = boxes[l]->FinestChain();
    for (int s = 0; s < finestChain->GetSamples()->size(); s++) {
      std::shared_ptr<SamplingState> sample = finestChain->GetSamples()->at(s);
      weight_sum += sample->weight;
    }
    int n_samples = std::ceil(weight_sum);
    int n_new_samples = std::ceil(n_samples * beta);

    if (verbosity > 0)
      std::cout << "var_mle " << var_mle << "\t" << n_new_samples << " new samples on level " << l << std::endl;
    for (int i = 0; i < n_new_samples; i++)
      boxes[l]->Sample();
  }

  if (verbosity > 0) {
    for (int l = 0; l <= levels; l++)
      boxes[l]->FinestChain()->PrintStatus("lvl " + std::to_string(l) + " ");
  }

  return GetSamples();
}

std::shared_ptr<MIMCMCBox> GreedyMLMCMC::GetBox(int index) {
  return boxes[index];
}

std::vector<std::shared_ptr<MIMCMCBox>> GreedyMLMCMC::GetBoxes() {
  return boxes;
}

void GreedyMLMCMC::WriteToFile(std::string filename) {
  for (auto box : boxes) {
    box->WriteToFile(filename);
  }
}

void GreedyMLMCMC::Draw(bool drawSamples) {
  std::ofstream graphfile;
  graphfile.open ("graph");
  graphfile << "digraph {" << std::endl;
  graphfile << "nodesep=1.2;" << std::endl;
  graphfile << "splines=false;" << std::endl;
  for (auto box : boxes) {
    box->Draw(graphfile, drawSamples);
  }
  graphfile << "}" << std::endl;
  graphfile.close();
}

std::shared_ptr<MultiIndexSet> GreedyMLMCMC::ProcessMultis(std::shared_ptr<MultiIndexSet> const& multis, 
                                                           unsigned int numLevels)
{
  if(multis){
    return multis;
  }else{
    return MultiIndexFactory::CreateFullTensor(1, numLevels-1);
  }
}
