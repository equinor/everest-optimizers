#include "MUQ/Utilities/MultiIndices/MultiIndexLimiter.h"


bool muq::Utilities::DimensionLimiter::IsFeasible(std::shared_ptr<MultiIndex> multi) const{

  for(auto pair = multi->GetNzBegin(); pair!=multi->GetNzEnd(); ++pair){
    if(((pair->first<lowerDim)||(pair->first>=lowerDim+length))&&(pair->second!=0))
      return false;
  }
  return true;
};

muq::Utilities::AnisotropicLimiter::AnisotropicLimiter(const Eigen::RowVectorXf& weightsIn, const double epsilonIn) : weights(weightsIn), epsilon(epsilonIn) {

  // validate weight vector
  for(int i = 0; i < weights.size(); ++i){
    if (weights(i) > 1 || weights[i] < 0)
      throw std::invalid_argument("AnisotropicLimiter requires all weights have to be in [0,1]. Got weight " + std::to_string(weights[i]));
  }
  // validate threshold
  if (epsilon >= 1 || epsilon <= 0)
      throw std::invalid_argument("AnisotropicLimiter requires epsilon to be in (0,1). Got epsilon = " + std::to_string(epsilon));
};

bool muq::Utilities::AnisotropicLimiter::IsFeasible(std::shared_ptr<MultiIndex> multi) const{

  double prod = 1;
  for(auto pair = multi->GetNzBegin(); pair!=multi->GetNzEnd(); ++pair){
    if(pair->first >= weights.size())
      return false;
    prod *= std::pow(weights(pair->first),(int)pair->second);
  }
  return prod >= epsilon;
};

bool muq::Utilities::MaxOrderLimiter::IsFeasible(std::shared_ptr<MultiIndex> multi) const{

  if(maxOrders.size()==0){
    return (multi->Max() <= maxOrder);
  }else{
    assert(multi->GetLength()<=maxOrders.size());

    if(multi->Max() <= vectorMin)
      return true;

    for(auto nzIter = multi->GetNzBegin(); nzIter!=multi->GetNzEnd(); ++nzIter){
      if(nzIter->second>maxOrders(nzIter->first)){
        return false;
      }
    }
    return true;
  }
};
