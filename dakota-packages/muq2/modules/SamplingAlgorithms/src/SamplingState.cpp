#include "MUQ/SamplingAlgorithms/SamplingState.h"
#include <Eigen/Core>
#include "MUQ/Utilities/AnyHelpers.h"

using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

SamplingState::SamplingState(Eigen::VectorXd const& stateIn, double weight) : state({stateIn}), weight(weight) {}
SamplingState::SamplingState(std::vector<Eigen::VectorXd> const& stateIn, double const weight) : state(stateIn), weight(weight) {}

bool SamplingState::HasMeta(std::string const& metaKey) {
  auto iter = meta.find(metaKey);
  return iter!=meta.end();
}

int SamplingState::TotalDim() const {
  int sum = 0;
  for(auto& s : state){
    sum += s.size();
  }
  return sum;
}

Eigen::VectorXd SamplingState::ToVector(int blockInd) const
{
  if(blockInd>=0){
    return state.at(blockInd);
  }else{
    if(state.size()==1)
      return state.at(0);
    
    Eigen::VectorXd output(TotalDim());
    unsigned int currInd = 0;
    for(auto& s : state){
      output.segment(currInd, s.size()) = s;
      currInd += s.size();
    }
    return output;
  }
}


double SamplingState::StateValue(unsigned int totalInd) const
{
  unsigned int sum = 0;
  for(auto& s : state){

    if(totalInd < sum + s.size())
      return s(totalInd - sum);

    sum += s.size();
  }

  return std::numeric_limits<double>::quiet_NaN();
}


double& SamplingState::StateValue(unsigned int totalInd)
{
  unsigned int sum = 0;
  for(auto& s : state){

    if(totalInd < sum + s.size())
      return s(totalInd - sum);

    sum += s.size();
  }

  assert(false);
  return state.at(0)(0);
}
