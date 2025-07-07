#include "MUQ/SamplingAlgorithms/SampleCollection.h"

#include "MUQ/Utilities/AnyHelpers.h"

#include "MUQ/SamplingAlgorithms/Diagnostics.h"


using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

ExpectedModPieceValue::ExpectedModPieceValue(std::shared_ptr<muq::Modeling::ModPiece> const& f, std::vector<std::string> const& metains) : f(f), metains(metains) {
  assert(f->numOutputs==1);
}

void ExpectedValueInputs(SamplingState const& a, std::vector<std::string> const& metains, std::vector<Eigen::VectorXd>& ins) {
  ins = a.state;

  for( auto metain : metains ) {
    auto it = a.meta.find(metain);
    assert(it!=a.meta.end());

    if( it->second.type()==typeid(Eigen::Vector2d) ) {
      ins.push_back(boost::any_cast<Eigen::Vector2d const&>(it->second));
      continue;
    }
    if( it->second.type()==typeid(Eigen::Vector3d) ) {
      ins.push_back(boost::any_cast<Eigen::Vector3d const&>(it->second));
      continue;
    }
    if( it->second.type()==typeid(Eigen::Vector4d) ) {
      ins.push_back(boost::any_cast<Eigen::Vector4d const&>(it->second));
      continue;
    }
    if( it->second.type()==typeid(Eigen::VectorXd) ) {
      ins.push_back(boost::any_cast<Eigen::VectorXd const&>(it->second));
      continue;
    }
    if( it->second.type()==typeid(double) ) {
      ins.push_back(Eigen::VectorXd::Constant(1, boost::any_cast<double const>(it->second)));
      continue;
    }
    if( it->second.type()==typeid(float) ) {
      ins.push_back(Eigen::VectorXd::Constant(1, boost::any_cast<float const>(it->second)));
      continue;
    }
    if( it->second.type()==typeid(int) ) {
      ins.push_back(Eigen::VectorXd::Constant(1, boost::any_cast<int const>(it->second)));
      continue;
    }
    if( it->second.type()==typeid(unsigned int) ) {
      ins.push_back(Eigen::VectorXd::Constant(1, boost::any_cast<unsigned int const>(it->second)));
      continue;
    }
  }
}

Eigen::VectorXd const& ExpectedModPieceValue::operator()(SamplingState const& a) {
  assert(f->numInputs==a.state.size()+metains.size());
  for( unsigned int i=0; i<a.state.size(); ++i ) { assert(a.state[i].size()==f->inputSizes(i)); }

  std::vector<Eigen::VectorXd> ins;
  ExpectedValueInputs(a, metains, ins);

  return f->Evaluate(ins) [0];
}

Eigen::VectorXd const& SamplingStateIdentity::operator()(SamplingState const& a)
{
  if(blockInd<0){
    const int totalSize = a.TotalDim();
    const int numBlocks = a.state.size();

    output.resize(totalSize);
    int currInd = 0;
    for(int i=0; i<numBlocks; ++i){
      output.segment(currInd,a.state.at(i).size()) = a.state.at(i);
      currInd += a.state.at(i).size();
    }
    return output;

  }else{
    output.resize(0);
    return a.state.at(blockInd);
  }
}

Eigen::VectorXd const& SamplingStatePartialMoment::operator()(SamplingState const& a)
{
  if(blockInd<0){
    const int totalSize = a.TotalDim();
    const int numBlocks = a.state.size();

    output.resize(totalSize);
    int currInd = 0;
    for(int i=0; i<numBlocks; ++i){
      output.segment(currInd,a.state.at(i).size()) = (a.state.at(i)-mu.segment(currInd,a.state.at(i).size())).array().pow(momentPower).matrix();
      currInd += a.state.at(i).size();
    }
    return output;

  }else{
    output = (a.state.at(blockInd)-mu).array().pow(momentPower).matrix();
    return output;
  }
}

void SampleCollection::Add(std::shared_ptr<SamplingState> newSamp)
{
  // copy the sample pointer
  samples.push_back(newSamp);//std::make_shared<SamplingState>(*newSamp));
}

std::shared_ptr<SamplingState> SampleCollection::at(unsigned i)
{
  return samples.at(i);
}
const std::shared_ptr<SamplingState> SampleCollection::at(unsigned i) const
{
  return samples.at(i);
}

std::shared_ptr<SampleCollection> SampleCollection::segment(unsigned int startInd, unsigned int length, unsigned int skipBy) const
{
  assert(startInd<size());
  assert(startInd+length<=size());

  std::shared_ptr<SampleCollection> output = std::make_shared<SampleCollection>();
  for(unsigned int i=startInd; i<startInd+length; i+=skipBy)
    output->Add(at(i));

  return output;
}

const std::shared_ptr<SamplingState> SampleCollection::back() const {
  return samples.back();
}


Eigen::VectorXd SampleCollection::CentralMoment(unsigned order, Eigen::VectorXd const& mean, int blockNum) const {
  SamplingStatePartialMoment op(blockNum, order, mean);

  Eigen::VectorXd stateSum;
  double weightSum;

  std::tie(weightSum, stateSum) = RecursiveSum(samples.begin(), samples.end(), op);
  return (stateSum / weightSum).eval();
}

Eigen::VectorXd SampleCollection::Mean(int blockNum) const
{
    SamplingStateIdentity op(blockNum);

    Eigen::VectorXd stateSum;
    double weightSum;

    std::tie(weightSum, stateSum) = RecursiveSum(samples.begin(), samples.end(), op);
    return (stateSum / weightSum).eval();
}


Eigen::MatrixXd SampleCollection::Covariance(Eigen::VectorXd const& mean, int blockInd) const {
  const int numSamps = samples.size();
  Eigen::MatrixXd samps;
  Eigen::VectorXd weights(numSamps);

  if(blockInd<0){

    const int totalSize = samples.at(0)->TotalDim();
    const int numBlocks = samples.at(0)->state.size();

    samps.resize(totalSize, numSamps);

    for(int i=0; i<numSamps; ++i){
      weights(i) = samples.at(i)->weight;

      int currInd = 0;
      for(int block = 0; block<numBlocks; ++block){
        samps.col(i).segment(currInd, samples.at(i)->state.at(block).size()) = samples.at(i)->state.at(block);
        currInd += samples.at(i)->state.at(block).size();
      }

    }

  }else{

    const int blockSize = samples.at(0)->state.at(blockInd).size();

    samps.resize(blockSize, numSamps);

    for(int i=0; i<numSamps; ++i){
      weights(i) = samples.at(i)->weight;
      samps.col(i) = samples.at(i)->state.at(blockInd);
    }
  }

  return (samps.colwise() - mean) * weights.asDiagonal() * (samps.colwise()-mean).transpose() / weights.sum();
}

std::vector<Eigen::MatrixXd> SampleCollection::RunningCovariance(int blockInd) const {
  return RunningCovariance(Mean(blockInd), blockInd);
}

std::vector<Eigen::MatrixXd> SampleCollection::RunningCovariance(Eigen::VectorXd const& mean, int blockInd) const {
  const int numSamps = size();
  std::vector<Eigen::MatrixXd> runCov(numSamps);

  if(blockInd<0){
    std::shared_ptr<SamplingState> s = at(0);

    const int totalSize = s->TotalDim();
    const int numBlocks = s->state.size();
    double totWeight = s->weight;

    Eigen::VectorXd samp(totalSize);

    int currInd = 0;
    for(int block = 0; block<numBlocks; ++block){
      samp.segment(currInd, s->state.at(block).size()) = s->state.at(block);
      currInd += s->state.at(block).size();
    }

    samp -= mean;
    runCov[0] = totWeight*samp*samp.transpose();

    for(int i=1; i<numSamps; ++i){
      s = at(i);

      int currInd = 0;
      for(int block = 0; block<numBlocks; ++block){
        samp.segment(currInd, s->state.at(block).size()) = s->state.at(block);
        currInd += s->state.at(block).size();
      }

      totWeight += s->weight;
      samp -= mean;
      runCov[i] = ((totWeight-s->weight)*runCov[i-1] + s->weight*samp*samp.transpose())/totWeight;
    }

  }else{
    std::shared_ptr<SamplingState> s = at(0);

    const int blockSize = s->state.at(blockInd).size();
    double totWeight = s->weight;

    Eigen::VectorXd samp = s->state.at(blockInd)-mean;
    runCov[0] = totWeight*samp*samp.transpose();

    for(int i=1; i<numSamps; ++i){
      s = at(i);
      totWeight += s->weight;
      samp = s->state.at(blockInd);
      runCov[i] = ((totWeight-s->weight)*runCov[i-1] + s->weight*samp*samp.transpose())/totWeight;
    }
  }

  return runCov;
}

std::pair<double,double> SampleCollection::RecursiveWeightSum(std::vector<std::shared_ptr<SamplingState>>::const_iterator start,
                                                              std::vector<std::shared_ptr<SamplingState>>::const_iterator end)
{
  int numSamps = std::distance(start,end);
  const int maxSamps = 20;

  // If the number of samples is small enough, we can safely add them up directly
  if(numSamps<maxSamps){

    double weightSquareSum = (*start)->weight * (*start)->weight;
    double weightSum = (*start)->weight;

    for(auto it=start+1; it!=end; ++it){
        weightSquareSum += (*it)->weight * (*it)->weight;
        weightSum += (*it)->weight;
    }
    return std::make_pair(weightSum, weightSquareSum);

  }else{
    int halfDist = std::floor(0.5*numSamps);
    double weight1, weight2, squaredSum1, squaredSum2;
    std::tie(weight1,squaredSum1) = RecursiveWeightSum(start, start+halfDist);
    std::tie(weight2,squaredSum2) = RecursiveWeightSum(start+halfDist, end);

    return std::make_pair(weight1+weight2, squaredSum1+squaredSum2);
  }
}


Eigen::VectorXd SampleCollection::ESS(int blockInd, std::string const& method) const
{ 
  if(method=="Batch"){
    return BatchESS(blockInd);
  }else if(method=="MultiBatch"){
    return MultiBatchESS(blockInd)*Eigen::VectorXd::Ones(1);
  }else{
    std::stringstream msg;
    msg << "Invalid method (" << method << ") passed to SampleCollection::ESS.  Valid options are \"Batch\" and \"MultiBatch\".";
    throw std::runtime_error(msg.str());
    return Eigen::VectorXd();
  }
}


double SampleCollection::MultiBatchESS(int blockInd, int batchSize, int overlap) const
{
  double numSamps = size();

  // If the blocksize wasn't given explicitly, use n^{1/3}
  if(batchSize<1)
    batchSize = std::min( std::floor(numSamps/(BlockSize(blockInd)+1)), std::round(std::pow(numSamps, 1.0/2.0)));

  // If the overlap wasn't given explicitly, use 10% of the batchSize
  if(overlap<1)
    overlap = std::floor(0.75*batchSize);

  // The total number of overlapping batches in the chain
  unsigned int numBatches = std::floor( (numSamps-batchSize)/(batchSize-overlap) );

  // The number of non-overlapping batches that we could fit in this chain 
  unsigned int numNonOverlapBatches = std::floor( (numSamps / batchSize) - 1 ); 

  // If the batch size and overlap do not result in enough batches, throw an error
  if(numNonOverlapBatches < BlockSize(blockInd)){

      std::stringstream msg;
      msg << "ERROR in SampleCollection::MultiBatchError.  Batch size of " << batchSize;
      msg << " results in " << numNonOverlapBatches << " non-overlapping batches, which";
      msg << " is less than the parameter dimension " << BlockSize(blockInd);
      msg << " and will result in a singular estimator covariance matrix.";
      throw std::runtime_error(msg.str());
  }

  // Number of non-overlapping variance estimators
  unsigned int numEstimators = std::floor( (numSamps - numNonOverlapBatches*batchSize)/double(batchSize-overlap) ); 

  // Sample mean
  Eigen::VectorXd mean = Mean(blockInd);

  // The covariance of batch estimators
  Eigen::MatrixXd estCov = Eigen::MatrixXd::Zero(BlockSize(blockInd), BlockSize(blockInd));

  // Compute the estimator covariance
  for(unsigned int estInd=0; estInd<numEstimators; ++estInd){
    for(unsigned int batchInd=0; batchInd<numNonOverlapBatches; ++batchInd){
      Eigen::VectorXd diff = segment(estInd*(batchSize-overlap) + batchInd*batchSize, batchSize)->Mean(blockInd) - mean;
      estCov += diff * diff.transpose();
    }
  }
  estCov /= (numNonOverlapBatches-1.0)*numEstimators; // <- Estimate of the size b_n estimator variance
  estCov *= double(batchSize); // <-Estimate of the size n estimator variance 

  double covDet = Covariance(mean).determinant();

  return std::min(numSamps, numSamps*std::pow(covDet / estCov.determinant(), 1.0/BlockSize(blockInd)) );
}

Eigen::VectorXd SampleCollection::BatchESS(int blockInd, int batchSize, int overlap) const
{ 
  double numSamps = size();

  Eigen::VectorXd avgEstVar = BatchError(blockInd, batchSize, overlap).array().square();

  Eigen::VectorXd ess = Variance(blockInd).array() / avgEstVar.array();
  for(int i=0; i<ess.size(); ++i)
    ess(i) = std::min(ess(i), numSamps);

  return ess;
}

Eigen::VectorXd SampleCollection::StandardError(int                blockInd, 
                                                std::string const& method) const
{
  if(method=="Batch"){
    return BatchError(blockInd);
  }else if(method=="MultiBatch"){
    return MultiBatchError(blockInd);
  }else{
    std::stringstream msg;
    msg << "Invalid method (" << method << ") passed to SampleCollection::StandardError.  Valid options are \"Batch\" and \"MultiBatch\".";
    throw std::runtime_error(msg.str());
    return Eigen::VectorXd();
  }
}

Eigen::VectorXd SampleCollection::BatchError(int blockInd, int batchSize, int overlap) const
{
  double numSamps = size();

  // If the blocksize wasn't given explicitly, use n^{1/2}
  if(batchSize<1)
    batchSize = std::round(std::pow(numSamps, 1.0/2.0));

  // If the overlap wasn't given explicitly, use 10% of the batchSize
  if(overlap<1)
    overlap = std::floor(0.75*batchSize);

  // The total number of overlapping batches in the chain
  unsigned int numBatches = std::floor( (numSamps-batchSize)/(batchSize-overlap) );

  // The number of non-overlapping batches that we could fit in this chain 
  unsigned int numNonOverlapBatches = std::floor( (numSamps / batchSize) - 1 ); 

  // Number of non-overlapping variance estimators
  unsigned int numEstimators = std::floor( (numSamps - numNonOverlapBatches*batchSize)/double(batchSize-overlap) ); 

  Eigen::VectorXd mean = Mean(blockInd);

  // Vector to hold the mean (over non-overlapping estimators) estimator variance.
  Eigen::VectorXd avgEstVar = Eigen::VectorXd::Zero(BlockSize(blockInd));
  for(unsigned int estInd=0; estInd<numEstimators; ++estInd){
    for(unsigned int batchInd=0; batchInd<numNonOverlapBatches; ++batchInd){
      avgEstVar += (1.0/(numNonOverlapBatches-1.0)) * (segment(estInd*(batchSize-overlap) + batchInd*batchSize, batchSize)->Mean(blockInd) - mean).array().square().matrix();
    }
  }
  avgEstVar /= numEstimators; // <- Estimate of the size b_n estimator variance
  avgEstVar *= (double(batchSize)/double(size())); // <-Estimate of the size n estimator variance 

  return avgEstVar.cwiseSqrt();
}

Eigen::VectorXd SampleCollection::MultiBatchError(int blockInd, int batchSize, int overlap) const
{
  double ess = MultiBatchESS(blockInd, batchSize, overlap);
  return (Variance() / ess).array().sqrt();
}


std::shared_ptr<SampleCollection> SampleCollection::FromMatrix(Eigen::MatrixXd const& samps)
{
  const unsigned int numSamps = samps.cols();
  
  std::shared_ptr<SampleCollection> output = std::make_shared<SampleCollection>();
  for(unsigned int col=0; col<numSamps; ++col)
    output->Add( std::make_shared<SamplingState>(samps.col(col)));
  
  return output;
}

Eigen::MatrixXd SampleCollection::AsMatrix(int blockDim) const
{
  if(samples.size()==0)
    return Eigen::MatrixXd();

  if(blockDim<0){

    Eigen::MatrixXd output(samples.at(0)->TotalDim(), samples.size());
    for(int i=0; i<samples.size(); ++i){
      int startInd = 0;

      for(int block=0; block<samples.at(0)->state.size(); ++block){
        int blockSize = samples.at(0)->state.at(block).size();
        output.col(i).segment(startInd, blockSize) = samples.at(i)->state.at(block);
        startInd += blockSize;
      }
    }

    return output;

  }else{
    Eigen::MatrixXd output(samples.at(0)->state.at(blockDim).size(), samples.size());
    for(int i=0; i<samples.size(); ++i)
      output.col(i) = samples.at(0)->state.at(blockDim);
    return output;
  }
}

Eigen::VectorXd SampleCollection::Weights() const
{
  Eigen::VectorXd output(samples.size());
  for(int i=0; i<samples.size(); ++i)
    output(i) = samples.at(i)->weight;
  return output;
}

bool SampleCollection::CreateDataset(std::shared_ptr<muq::Utilities::HDF5File> hdf5file, std::string const& dataname, int const dataSize, int const totSamps) const {
  if( !hdf5file->IsDataSet(dataname) ) { return true; }

  Eigen::VectorXi size = hdf5file->GetDataSetSize(dataname);
  if( size(0)!=dataSize || size(1)!=totSamps ) { return true; }

  return false;
}

void SampleCollection::WriteToFile(std::string const& filename, std::string const& dataset) const {
  if( size()==0 ) { return; }

  // open the hdf5 file
  auto hdf5file = std::make_shared<HDF5File>(filename);

  // write the sample matrix and weights
  hdf5file->WriteMatrix(dataset+"/samples", AsMatrix());
  hdf5file->WriteMatrix(dataset+"/weights", (Eigen::RowVectorXd)Weights());

  // meta data
  const std::unordered_map<std::string, Eigen::MatrixXd>& meta = GetMeta();

  // write meta data to file
  for( const auto& data : meta ) { hdf5file->WriteMatrix(dataset+"/"+data.first, data.second); }

  hdf5file->Close();
}

unsigned SampleCollection::size() const { return samples.size(); }


std::set<std::string> SampleCollection::ListMeta(bool requireAll) const{

    std::set<std::string> output;

    // Add all of the metadata from the first sample
    for(auto& metaPair : at(0)->meta)
      output.insert(metaPair.first);

    // Loop through the rest of the samples
    for(unsigned int i=1; i<size(); ++i) {
      if(requireAll){

        // Make a list of any keys that we don't have already
        std::vector<std::string> eraseElements;
        for(auto& key : output){
          if( !at(i)->HasMeta(key) ){
            eraseElements.push_back(key);
          }
        }

        // Erase any key that we didn't find in this sample
        for(auto& key : eraseElements){
          output.erase(key);
        }

      }else{
        for(auto& metaPair : at(i)->meta)
          output.insert(metaPair.first);
      }
    }

    return output;
}


Eigen::MatrixXd SampleCollection::GetMeta(std::string const& name) const {

  Eigen::MatrixXd meta;

  // for each sample
  for( unsigned int i=0; i<size(); ++i ) {
    // get the meta data for that sample
    auto it = at(i)->meta.find(name);
    if( it==at(i)->meta.end() ) { continue; }

    if( it->second.type()==typeid(Eigen::Vector2d) ) {
      // create a matrix for the meta data
      if( meta.size()==0 ) {
        meta = Eigen::MatrixXd::Constant(2, size(), std::numeric_limits<double>::quiet_NaN());
      }
      meta.col(i) = boost::any_cast<Eigen::Vector2d const&>(it->second);

      continue;
    }

    if( it->second.type()==typeid(Eigen::Vector3d) ) {
      // create a matrix for the meta data
      if( meta.size()==0 ) {
        meta = Eigen::MatrixXd::Constant(3, size(), std::numeric_limits<double>::quiet_NaN());
      }
      meta.col(i) = boost::any_cast<Eigen::Vector3d const&>(it->second);

      continue;
    }

    if( it->second.type()==typeid(Eigen::Vector4d) ) {
      // create a matrix for the meta data
      if( meta.size()==0 ) {
        meta = Eigen::MatrixXd::Constant(4, size(), std::numeric_limits<double>::quiet_NaN());
      }
      meta.col(i) = boost::any_cast<Eigen::Vector4d const&>(it->second);

      continue;
    }

    if( it->second.type()==typeid(Eigen::VectorXd) ) {
      // create a matrix for the meta data
      if( meta.size()==0 ) {
        meta = Eigen::MatrixXd::Constant(boost::any_cast<Eigen::VectorXd const&>(it->second).size(), size(), std::numeric_limits<double>::quiet_NaN());
      }

      meta.col(i) = boost::any_cast<Eigen::VectorXd const&>(it->second);

      continue;
    }

    // create a matrix, assuming scalar type, for the meta data
    if( meta.size()==0 ) {
      meta = Eigen::MatrixXd::Constant(1, size(), std::numeric_limits<double>::quiet_NaN());
    }

    if( it->second.type()==typeid(double) ) { // doubles
      meta(i) = boost::any_cast<double const>(it->second);
      continue;
    }

    if( it->second.type()==typeid(float) ) { // floats
      meta(i) = boost::any_cast<float const>(it->second);
      continue;
    }

    if( it->second.type()==typeid(int) ) { // ints
      meta(i) = boost::any_cast<int const>(it->second);
      continue;
    }

    if( it->second.type()==typeid(unsigned int) ) { // unsigned ints
      meta(i) = boost::any_cast<unsigned int const>(it->second);
      continue;
    }
  }

  return meta;
}

std::unordered_map<std::string, Eigen::MatrixXd> SampleCollection::GetMeta() const {
  std::unordered_map<std::string, Eigen::MatrixXd> meta;
  for( unsigned int i=0; i<size(); ++i ) { // loop through the samples
    // loop through all meta data for this sample
    for( auto it = at(i)->meta.begin(); it!=at(i)->meta.end(); ++it ) {
      // if we have not yet gotten this meta information
      auto mat = meta.find(it->first);
      if( mat==meta.end() ) {
        meta[it->first] = GetMeta(it->first);
      }
    }
  }

  return meta;
}

Eigen::VectorXd SampleCollection::ExpectedValue(std::shared_ptr<muq::Modeling::ModPiece> const& f, std::vector<std::string> const& metains) const {
  ExpectedModPieceValue op(f, metains);

  Eigen::VectorXd val;
  double weight;

  std::tie(weight, val) = RecursiveSum(samples.begin(), samples.end(), op);
  return (val / weight).eval();
}

std::vector<Eigen::VectorXd> SampleCollection::RunningExpectedValue(std::shared_ptr<muq::Modeling::ModPiece> const& f, std::vector<std::string> const& metains) const {
  const unsigned int numSamps = size();
  std::vector<Eigen::VectorXd> runningExpected(numSamps);

  std::shared_ptr<SamplingState> s = at(0);
  double totWeight = s->weight;

  std::vector<Eigen::VectorXd> ins;
  ExpectedValueInputs(*s, metains, ins);

  runningExpected[0] = totWeight*f->Evaluate(ins) [0];
  for( unsigned int i=1; i<numSamps; ++i ) {
    s = at(i);
    ExpectedValueInputs(*s, metains, ins);
    totWeight += s->weight;

    runningExpected[i] = ((totWeight-s->weight)*runningExpected[i-1] + s->weight*f->Evaluate(ins) [0])/totWeight;
  }

  return runningExpected;
}


unsigned int SampleCollection::BlockSize(int blockInd) const
{
  if(blockInd<0){
    return samples.at(0)->TotalDim();
  }else{
    return samples.at(0)->state.at(blockInd).size();
  }
}

unsigned int SampleCollection::NumBlocks() const
{
  return samples.at(0)->state.size();
}


Eigen::VectorXd SampleCollection::Rhat(int                        blockDim, 
                                      unsigned int                numSegments, 
                                      boost::property_tree::ptree options) const
{   
    std::vector<std::shared_ptr<const SampleCollection>> chains;
    chains.push_back( shared_from_this());

    return Diagnostics::Rhat(Diagnostics::SplitChains(chains, numSegments), options);
}