#include "MUQ/SamplingAlgorithms/Diagnostics.h"

#include "MUQ/SamplingAlgorithms/MultiIndexEstimator.h"
#include "MUQ/SamplingAlgorithms/MarkovChain.h"
#include "MUQ/SamplingAlgorithms/SampleCollection.h"

#include <boost/math/distributions/normal.hpp>
#include <algorithm>
#include <vector>

using namespace muq::SamplingAlgorithms;

template<typename EstimatorType>
Eigen::VectorXd muq::SamplingAlgorithms::Diagnostics::Rhat(std::vector<std::shared_ptr<EstimatorType>> const& collections,
                                                           boost::property_tree::ptree                         options)
{ 
  std::vector<std::shared_ptr<SampleEstimator>> chains;

  // Check to see if we can cast the estimator to SampleCollection
  auto samplePtr = std::dynamic_pointer_cast<SampleCollection>(collections.at(0)); 
  if(samplePtr){

    // Create a new vector of sample collections
    std::vector<std::shared_ptr<SampleCollection>> newCollections(collections.size());
    for(unsigned int i=0; i<collections.size(); ++i)
      newCollections.at(i) = std::dynamic_pointer_cast<SampleCollection>(collections.at(i));

    // Split and possibly transform the chains
    if(options.get("Split",true))
      newCollections = SplitChains(newCollections);

    if(options.get("Transform",false))
      newCollections = TransformChains(newCollections);

    // Now cast the split and ranked sample collections back to a SampleEstimator
    chains.resize(newCollections.size());
    for(unsigned int chainInd=0; chainInd<newCollections.size(); ++chainInd)
      chains.at(chainInd) = std::dynamic_pointer_cast<SampleEstimator>(newCollections.at(chainInd));
    
  }else{

    chains.resize(collections.size());
    for(unsigned int chainInd=0; chainInd<collections.size(); ++chainInd)
      chains.at(chainInd) = std::dynamic_pointer_cast<SampleEstimator>(collections.at(chainInd));

  }

  if(options.get("Multivariate",false)){
    return BasicMPSRF(chains) * Eigen::VectorXd::Ones(1);
  }else{
    return BasicRhat(chains);
  }
}

template Eigen::VectorXd muq::SamplingAlgorithms::Diagnostics::Rhat(std::vector<std::shared_ptr<MarkovChain>> const& collections,
                                                                    boost::property_tree::ptree                      options);
template Eigen::VectorXd muq::SamplingAlgorithms::Diagnostics::Rhat(std::vector<std::shared_ptr<MultiIndexEstimator>> const& collections,
                                                                    boost::property_tree::ptree                              options);
template Eigen::VectorXd muq::SamplingAlgorithms::Diagnostics::Rhat(std::vector<std::shared_ptr<SampleCollection>> const& collections,
                                                                    boost::property_tree::ptree                           options);


std::vector<std::shared_ptr<SampleCollection>> muq::SamplingAlgorithms::Diagnostics::SplitChains(std::vector<std::shared_ptr<SampleCollection>> const& origChains, 
                                                                                                 unsigned int numSegments)
{
  std::vector<std::shared_ptr<const SampleCollection>> constChains;
  for(int i=0; i<origChains.size(); ++i)
    constChains.push_back(std::const_pointer_cast<const SampleCollection>(origChains.at(i)));
  
  return SplitChains(constChains,numSegments);
}
/** Performs a split of the chains. */
std::vector<std::shared_ptr<SampleCollection>> muq::SamplingAlgorithms::Diagnostics::SplitChains(std::vector<std::shared_ptr<const SampleCollection>> const& origChains, 
                                                                                                 unsigned int numSegments)
{
  std::vector<std::shared_ptr<SampleCollection>> chains;

  double fraction = 1.0/double(numSegments);

  // Figure out how long the split chains will be
  unsigned int chainLength = std::floor(fraction*origChains.at(0)->size());
  unsigned int numChains = numSegments*origChains.size();
  const unsigned int dim = origChains.at(0)->at(0)->TotalDim();

  chains.resize(numChains);

  // Extract the split chains
  for(int i=0; i<origChains.size();++i){
    for(int j=0; j<numSegments; ++j){
      chains.at(numSegments*i+j) = origChains.at(i)->segment(j*chainLength, chainLength);
    }
  }

  return chains;
}

/** Performas a Gaussianization of the chains based on ranking the samples and applying a Gaussian transform. */
std::vector<std::shared_ptr<SampleCollection>> muq::SamplingAlgorithms::Diagnostics::TransformChains(std::vector<std::shared_ptr<SampleCollection>> const& origChains)
{
  const boost::math::normal std_normal(0.0, 1.0);

  const unsigned int dim = origChains.at(0)->at(0)->TotalDim();

  unsigned int numChains = origChains.size();
  unsigned int chainLength = origChains.at(0)->size();
  const unsigned int totalSamps = numChains*chainLength;

  std::vector<std::shared_ptr<SampleCollection>> chains;
  chains.insert(chains.begin(), origChains.begin(), origChains.end());
  
  for(unsigned int i=0; i<dim; ++i){

    // Compute the ranks
    std::vector<Eigen::VectorXd> ranks = ComputeRanks(chains,i);

    // Apply a normal transformation to the ranks and compute chain means and variances.  See eqn. (14) in https://arxiv.org/pdf/1903.08008.pdf
    for(unsigned int chainInd=0; chainInd<ranks.size(); ++chainInd){
      ranks.at(chainInd) = ( (ranks.at(chainInd).array()+0.625)/(totalSamps + 0.25) ).unaryExpr([&std_normal](double v){return boost::math::quantile(std_normal, v);});

      for(unsigned int sampInd=0; sampInd<chains.at(chainInd)->size(); ++sampInd)
        chains.at(chainInd)->at(sampInd)->StateValue(i) = ranks.at(chainInd)(sampInd);
    }
  }

  return chains;
}


double muq::SamplingAlgorithms::Diagnostics::BasicMPSRF(std::vector<std::shared_ptr<SampleEstimator>> const& chains)
{ 
  const unsigned int numChains = chains.size();
  const unsigned int dim = chains.at(0)->BlockSize(-1);

  double lengthScale = 1.0;

  // Check to see if the sample estimator is a SampleCollection, which has a length
  auto cast = std::dynamic_pointer_cast<SampleCollection>(chains.at(0));
  if(cast)
    lengthScale = (cast->size() - 1.0)/cast->size();

  // A matrix of the chain means.
  Eigen::MatrixXd chainMeans(dim,numChains);

  // All of the within-chain covariance matrices
  std::vector<Eigen::MatrixXd> chainCovs(numChains);
  
  // Compute the mean and covariance of each chain
  for(int i=0; i<numChains; ++i){
    chainMeans.col(i) = chains.at(i)->Mean();
    chainCovs.at(i) = chains.at(i)->Covariance();
  }

  // Now we're good to compute the MPSRF estimator of Brooks 1998
  Eigen::VectorXd globalMean = chainMeans.rowwise().mean();

  Eigen::MatrixXd W = Eigen::MatrixXd::Zero(chainCovs.at(0).rows(), chainCovs.at(0).cols());
  for(auto& cov : chainCovs) 
    W += cov;

  W /= chainCovs.size();

  // Compute the between-chain covariance
  Eigen::MatrixXd diff = chainMeans.colwise()-globalMean;
  Eigen::MatrixXd B = (1.0/(numChains-1)) * diff * diff.transpose();

  Eigen::MatrixXd Vhat = lengthScale * W + (1.0 + 1.0/numChains)*B;

  Eigen::VectorXd lambdas = Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd>(B,W,Eigen::EigenvaluesOnly).eigenvalues();
  
  if(lambdas.size()==1)
    return std::sqrt( lengthScale + lambdas(0)*(numChains+1)/numChains );

  if(lambdas(1)>lambdas(0)){
    // Get an estimate of the marginal posterior variance
    return std::sqrt( lengthScale + lambdas(lambdas.size()-1)*(numChains+1)/numChains );
  }else{
    return std::sqrt( lengthScale + lambdas(0)*(numChains+1)/numChains );
  }
}

Eigen::VectorXd muq::SamplingAlgorithms::Diagnostics::BasicRhat(std::vector<std::shared_ptr<SampleEstimator>> const& chains)
{
 
  
  // If we aren't working with SampleCollections, use the vanilla Rhat statistic of Gelman 2013
  const unsigned int numChains = chains.size();
  const unsigned int dim = chains.at(0)->BlockSize(-1);

  Eigen::MatrixXd mus(dim, numChains);
  Eigen::MatrixXd sbjs(dim, numChains);
  Eigen::MatrixXd vars(dim,numChains);

  for(unsigned int i=0; i<numChains; ++i){
    mus.col(i) = chains.at(i)->Mean();
    sbjs.col(i) = chains.at(i)->CentralMoment(2,mus.col(i)); // Possibly biased estimate
    vars.col(i) = chains.at(i)->Variance(mus.col(i)); // Possibly unbiased estimate
  }
  
  Eigen::VectorXd mumu = mus.rowwise().mean();
  Eigen::VectorXd muVar = (mus.colwise()-mumu).array().square().rowwise().sum() / (numChains-1.0);

  Eigen::VectorXd varEst = sbjs.rowwise().mean() + muVar;
  Eigen::VectorXd W = vars.rowwise().mean();

  return (varEst.array() / W.array()).array().sqrt();
}



std::vector<Eigen::VectorXd> muq::SamplingAlgorithms::Diagnostics::ComputeRanks(std::vector<std::shared_ptr<SampleCollection>> const& collections,
                                                                                unsigned int                                      dim)
{

  // A vector of sample indices [chainInd, sampInd]
  std::vector<std::pair<unsigned int, unsigned int>> sampInds;

  for(unsigned int chainInd=0; chainInd<collections.size(); ++chainInd){
    for(unsigned int sampInd=0; sampInd<collections.at(chainInd)->size(); ++sampInd)
      sampInds.push_back(std::make_pair(chainInd,sampInd));
  }

  // Sort the vector of indices according to the value of the parameters
  auto compLambda = [&collections, dim](std::pair<unsigned int, unsigned int> const& p1, std::pair<unsigned int, unsigned int> const& p2) {
                      return collections.at(p1.first)->at(p1.second)->StateValue(dim) < collections.at(p2.first)->at(p2.second)->StateValue(dim);
                    };

  std::stable_sort(sampInds.begin(), sampInds.end(), compLambda);

  // Set up empty vectors for storing the ranks
  std::vector<Eigen::VectorXd> ranks(collections.size());
  for(unsigned int i=0; i<ranks.size(); ++i)
    ranks.at(i).resize(collections.at(i)->size());

  // Figure out the rank of each sample
  unsigned int rawRank = 0;
  double currVal, nextVal;
  unsigned int numRepeat, chainInd, sampInd;

  while(rawRank < sampInds.size()){
    std::tie(chainInd, sampInd) = sampInds.at(rawRank);
    currVal = collections.at(chainInd)->at(sampInd)->StateValue(dim);

    // Look ahead and find the next sample with a new value
    numRepeat = 1;
    for(numRepeat=1; numRepeat<sampInds.size()-rawRank; ++numRepeat){
      std::tie(chainInd, sampInd) = sampInds.at(rawRank+numRepeat);
      nextVal = collections.at(chainInd)->at(sampInd)->StateValue(dim);

      if(std::abs(currVal-nextVal)>1e-15){
        break;
      }
    }

    // Compute the average rank across all of the duplicates
    double avgRank = 0.5*(rawRank + rawRank+numRepeat-1);

    // Set the ranks to the average value
    for(int i=rawRank; i<rawRank+numRepeat; ++i){
      std::tie(chainInd, sampInd) = sampInds.at(i);
      ranks.at(chainInd)(sampInd) = avgRank;
    }

    // Update how many samples we've moved through
    rawRank += numRepeat;
  }

  return ranks;
}
