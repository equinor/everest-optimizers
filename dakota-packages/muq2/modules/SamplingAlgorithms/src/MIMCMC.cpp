#include "MUQ/SamplingAlgorithms/MIMCMC.h"

#include "MUQ/SamplingAlgorithms/DefaultComponentFactory.h"

using namespace muq::SamplingAlgorithms;

MIMCMC::MIMCMC(boost::property_tree::ptree                                  pt, 
               Eigen::VectorXd                                       const& startPt,
               std::vector<std::shared_ptr<muq::Modeling::ModPiece>> const& models,
               std::shared_ptr<MultiIndexSet>                        const& multis) : MIMCMC(pt,startPt, CreateProblems(models),multis)
{
  
}
      
MIMCMC::MIMCMC(boost::property_tree::ptree                                  pt, 
               Eigen::VectorXd                                       const& startPt,
               std::vector<std::shared_ptr<AbstractSamplingProblem>> const& problems,
               std::shared_ptr<MultiIndexSet>                        const& multis) : MIMCMC(pt, std::make_shared<DefaultComponentFactory>(pt,startPt,ProcessMultis(multis,problems.size()),problems))
{

}


MIMCMC::MIMCMC (pt::ptree pt, std::shared_ptr<MIComponentFactory> const& componentFactory)
: pt(pt),
  componentFactory(componentFactory)
{
  gridIndices = MultiIndexFactory::CreateFullTensor(componentFactory->FinestIndex()->GetVector());

  for (int i = 0; i < gridIndices->Size(); i++) {
    std::shared_ptr<MultiIndex> boxHighestIndex = (*gridIndices)[i];
    auto box = std::make_shared<MIMCMCBox>(componentFactory, boxHighestIndex);
    boxes.push_back(box);
  }
}

std::shared_ptr<MIMCMCBox> MIMCMC::GetBox(std::shared_ptr<MultiIndex> index) {
  for (std::shared_ptr<MIMCMCBox> box : boxes) {
    if (*(box->GetHighestIndex()) == *index)
      return box;
  }
  return nullptr;
}

std::shared_ptr<MultiIndexEstimator> MIMCMC::GetSamples() const {
  return std::make_shared<MultiIndexEstimator>(boxes);
}
std::shared_ptr<MultiIndexEstimator> MIMCMC::GetQOIs() const {
  return std::make_shared<MultiIndexEstimator>(boxes,true);
}

std::shared_ptr<MultiIndexEstimator> MIMCMC::Run() {
  for (auto box : boxes) {
    assert(box);
    int numSamples = pt.get<int>("NumSamples" + multiindexToConfigString(box->GetHighestIndex()));
    for (int samp = 0; samp < numSamples; samp++) {
      box->Sample();
    }
  }

  return GetSamples();
}

std::shared_ptr<MIMCMCBox> MIMCMC::GetMIMCMCBox(std::shared_ptr<MultiIndex> index) {
  for (auto box : boxes) {
    if (*(box->GetHighestIndex()) == *index)
      return box;
  }
  return nullptr;
}

void MIMCMC::WriteToFile(std::string filename) {
  for (auto box : boxes) {
    box->WriteToFile(filename);
  }
}


std::shared_ptr<MultiIndexSet> MIMCMC::GetIndices() {
  return gridIndices;
}


std::string MIMCMC::multiindexToConfigString (std::shared_ptr<MultiIndex> index) {
  std::stringstream strs;
  for (int i = 0; i < index->GetLength(); i++) {
    strs << "_" << index->GetValue(i);
  }
  return strs.str();
}

void MIMCMC::Draw(bool drawSamples) {
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

std::shared_ptr<MultiIndexSet> MIMCMC::ProcessMultis(std::shared_ptr<MultiIndexSet> const& multis, 
                                                           unsigned int numLevels)
{
  if(multis){
    return multis;
  }else{
    return MultiIndexFactory::CreateFullTensor(1, numLevels-1);
  }
}

std::vector<std::shared_ptr<AbstractSamplingProblem>> MIMCMC::CreateProblems(std::vector<std::shared_ptr<muq::Modeling::ModPiece>> const& models)
{
  std::vector<std::shared_ptr<AbstractSamplingProblem>> output(models.size());
  for(unsigned int i=0; i<models.size(); ++i)
    output.at(i) = std::make_shared<SamplingProblem>(models.at(i));
  
  return output;
}