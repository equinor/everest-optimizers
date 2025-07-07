#include "MUQ/SamplingAlgorithms/DILIKernel.h"

#include "MUQ/Modeling/LinearAlgebra/HessianOperator.h"
#include "MUQ/Modeling/LinearAlgebra/GaussNewtonOperator.h"
#include "MUQ/Modeling/LinearAlgebra/GaussianOperator.h"
#include "MUQ/Modeling/LinearAlgebra/StochasticEigenSolver.h"
#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/Distributions/DensityProduct.h"
#include "MUQ/Modeling/SumPiece.h"
#include "MUQ/Modeling/ModGraphPiece.h"
#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/LinearAlgebra/ZeroOperator.h"

using namespace muq::SamplingAlgorithms;
using namespace muq::Modeling;

AverageHessian::AverageHessian(unsigned int                            numOldSamps,
                               std::shared_ptr<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>> const& uQRIn,
                               std::shared_ptr<Eigen::MatrixXd> const& oldWIn,
                               std::shared_ptr<Eigen::VectorXd> const& oldValsIn,
                               std::shared_ptr<muq::Modeling::LinearOperator> const& newHessIn) : LinearOperator(newHessIn->rows(), newHessIn->cols()),
                                                                                                  numSamps(numOldSamps),
                                                                                                  uQR(uQRIn),
                                                                                                  oldW(oldWIn),
                                                                                                  oldEigVals(oldValsIn),
                                                                                                  newHess(newHessIn)
{
  assert(oldW->rows()>0);
  assert(oldW->cols()>0);
  assert(oldEigVals->size()==oldW->cols());

  thinQ = uQR->householderQ().setLength(uQR->nonzeroPivots()) * Eigen::MatrixXd::Identity(oldW->rows(), uQR->rank());
}

Eigen::MatrixXd AverageHessian::Apply(Eigen::Ref<const Eigen::MatrixXd> const& x)
{
  Eigen::MatrixXd result = newHess->Apply(x) / (numSamps+1.0);
  result += (numSamps/(numSamps+1.0)) * ((*oldW) * oldEigVals->asDiagonal()* uQR->colsPermutation() * uQR->matrixR().topLeftCorner(uQR->rank(), uQR->rank()).template triangularView<Eigen::Upper>().solve(thinQ.transpose() * x));

  return result;
}

Eigen::MatrixXd AverageHessian::ApplyTranspose(Eigen::Ref<const Eigen::MatrixXd> const& x)
{
  return Apply(x);
}


Eigen::MatrixXd CSProjector::Apply(Eigen::Ref<const Eigen::MatrixXd> const& x)
{
  Eigen::MatrixXd tmp(x - U->leftCols(lisDim) * W->leftCols(lisDim).transpose() * x);
  if (x.cols() == 1) {
    Eigen::VectorXd xVec(x.rows());
    for (size_t i(0); i < x.rows(); ++i) {
      xVec(i) = x(i,0);
    }
    Eigen::VectorXd tmpVec(xVec - U->leftCols(lisDim) * W->leftCols(lisDim).transpose() * xVec);
    for (size_t i(0); i < tmp.rows(); ++i) {
      tmp(i,0) = tmpVec(i,0);
    }
  }
  return tmp;
  //return x - U->leftCols(lisDim)*W->leftCols(lisDim).transpose() * x;
}

Eigen::MatrixXd CSProjector::ApplyTranspose(Eigen::Ref<const Eigen::MatrixXd> const& x)
{
  return x - W->leftCols(lisDim)*U->leftCols(lisDim).transpose() * x;
}

Eigen::MatrixXd LIS2Full::Apply(Eigen::Ref<const Eigen::MatrixXd> const& x)
{
  Eigen::MatrixXd tmp(U->leftCols(lisDim) * L->asDiagonal() * x);
  if (x.cols() == 1) {
    Eigen::VectorXd xVec(x.rows());
    for (size_t i(0); i < x.rows(); ++i) {
      xVec(i) = x(i,0);
    }
    Eigen::VectorXd tmpVec(U->leftCols(lisDim) * L->asDiagonal() * xVec);
    for (size_t i(0); i < tmp.rows(); ++i) {
      tmp(i,0) = tmpVec(i,0);
    }
  }
  return tmp;
  //return U->leftCols(lisDim) * L->asDiagonal() * x;
}

Eigen::MatrixXd LIS2Full::ApplyTranspose(Eigen::Ref<const Eigen::MatrixXd> const& x)
{
  return L->asDiagonal() * U->leftCols(lisDim).transpose() * x;
}



DILIKernel::DILIKernel(boost::property_tree::ptree       const& pt,
                       std::shared_ptr<AbstractSamplingProblem> problem) : DILIKernel(pt,
                                                                                      problem,
                                                                                      ExtractPrior(problem, pt.get("Prior Node", "Prior")),
                                                                                      ExtractLikelihood(problem,pt.get("Likelihood Node", "Likelihood")))
{
}

DILIKernel::DILIKernel(boost::property_tree::ptree       const& pt,
                       std::shared_ptr<AbstractSamplingProblem> problem,
                       Eigen::VectorXd                   const& genEigVals,
                       Eigen::MatrixXd                   const& genEigVecs) : DILIKernel(pt,
                                                                                         problem,
                                                                                         ExtractPrior(problem, pt.get("Prior Node", "Prior")),
                                                                                         ExtractLikelihood(problem,pt.get("Likelihood Node", "Likelihood")),
                                                                                         genEigVals,
                                                                                         genEigVecs)
{
}


DILIKernel::DILIKernel(boost::property_tree::ptree                  const& pt,
                       std::shared_ptr<AbstractSamplingProblem>            problem,
                       std::shared_ptr<muq::Modeling::GaussianBase> const& priorIn,
                       std::shared_ptr<muq::Modeling::ModPiece>     const& noiseModelIn,
                       std::shared_ptr<muq::Modeling::ModPiece>     const& forwardModelIn) : TransitionKernel(pt,problem),
                                                                                             lisKernelOpts(pt.get_child(pt.get<std::string>("LIS Block"))),
                                                                                             csKernelOpts(pt.get_child(pt.get<std::string>("CS Block"))),
                                                                                             logLikelihood(CreateLikelihood(forwardModelIn,noiseModelIn)),
                                                                                             prior(priorIn),
                                                                                             forwardModel(forwardModelIn),
                                                                                             noiseDensity(noiseModelIn),
                                                                                             hessType(pt.get("HessianType","GaussNewton")),
                                                                                             updateInterval(pt.get("Adapt Interval",-1)),
                                                                                             adaptStart(pt.get("Adapt Start",1)),
                                                                                             adaptEnd(pt.get("Adapt End", -1)),
                                                                                             initialHessSamps(pt.get("Initial Weight",100)),
                                                                                             hessValTol(pt.get("Hessian Tolerance",1e-4)),
                                                                                             lisValTol(pt.get("LIS Tolerance", 0.1))
{
  try{
    std::string blockName = pt.get<std::string>("Eigensolver Block");
    eigOpts = pt.get_child(blockName);
  }catch(const boost::property_tree::ptree_bad_path&){
    // Do nothing, just leave the solver options ptree empty
  }
}

DILIKernel::DILIKernel(boost::property_tree::ptree                  const& pt,
                       std::shared_ptr<AbstractSamplingProblem>            problem,
                       std::shared_ptr<muq::Modeling::GaussianBase> const& priorIn,
                       std::shared_ptr<muq::Modeling::ModPiece>     const& noiseModelIn,
                       std::shared_ptr<muq::Modeling::ModPiece>     const& forwardModelIn,
                       Eigen::VectorXd                              const& genEigVals,
                       Eigen::MatrixXd                              const& genEigVecs) : TransitionKernel(pt,problem),
                                                                                         lisKernelOpts(pt.get_child(pt.get<std::string>("LIS Block"))),
                                                                                         csKernelOpts(pt.get_child(pt.get<std::string>("CS Block"))),
                                                                                         logLikelihood(CreateLikelihood(forwardModelIn,noiseModelIn)),
                                                                                         prior(priorIn),
                                                                                         forwardModel(forwardModelIn),
                                                                                         noiseDensity(noiseModelIn),
                                                                                         hessType(pt.get("HessianType","GaussNewton")),
                                                                                         updateInterval(pt.get("Adapt Interval",-1)),
                                                                                         adaptStart(pt.get("Adapt Start",1)),
                                                                                         adaptEnd(pt.get("Adapt End", -1)),
                                                                                         initialHessSamps(pt.get("Initial Weight",100)),
                                                                                         hessValTol(pt.get("Hessian Tolerance",1e-4)),
                                                                                         lisValTol(pt.get("LIS Tolerance", 0.1))
{
  try{
    std::string blockName = pt.get<std::string>("Eigensolver Block");
    eigOpts = pt.get_child(blockName);
  }catch(const boost::property_tree::ptree_bad_path&){
    // Do nothing, just leave the solver options ptree empty
  }

  SetLIS(genEigVals, genEigVecs);
}

DILIKernel::DILIKernel(boost::property_tree::ptree                  const& pt,
                       std::shared_ptr<AbstractSamplingProblem>            problem,
                       std::shared_ptr<muq::Modeling::GaussianBase> const& priorIn,
                       std::shared_ptr<muq::Modeling::ModPiece>     const& likelihoodIn) : TransitionKernel(pt,problem),
                                                                                           lisKernelOpts(pt.get_child(pt.get<std::string>("LIS Block"))),
                                                                                           csKernelOpts(pt.get_child(pt.get<std::string>("CS Block"))),
                                                                                           logLikelihood(likelihoodIn),
                                                                                           prior(priorIn),
                                                                                           forwardModel(ExtractForwardModel(likelihoodIn)),
                                                                                           noiseDensity(ExtractNoiseModel(likelihoodIn)),
                                                                                           hessType(pt.get("HessianType","GaussNewton")),
                                                                                           updateInterval(pt.get("Adapt Interval",-1)),
                                                                                           adaptStart(pt.get("Adapt Start",1)),
                                                                                           adaptEnd(pt.get("Adapt End", -1)),
                                                                                           initialHessSamps(pt.get("Initial Weight",100)),
                                                                                           hessValTol(pt.get("Hessian Tolerance",1e-4)),
                                                                                           lisValTol(pt.get("LIS Tolerance", 0.1))
{
  try{
    std::string blockName = pt.get<std::string>("Eigensolver Block");
    eigOpts = pt.get_child(blockName);
  }catch(const boost::property_tree::ptree_bad_path&){
    // Do nothing, just leave the solver options ptree empty
  }
}

DILIKernel::DILIKernel(boost::property_tree::ptree                  const& pt,
                       std::shared_ptr<AbstractSamplingProblem>            problem,
                       std::shared_ptr<muq::Modeling::GaussianBase> const& priorIn,
                       std::shared_ptr<muq::Modeling::ModPiece>     const& likelihoodIn,
                       Eigen::VectorXd                              const& genEigVals,
                       Eigen::MatrixXd                              const& genEigVecs) : TransitionKernel(pt,problem),
                                                                                           lisKernelOpts(pt.get_child(pt.get<std::string>("LIS Block"))),
                                                                                           csKernelOpts(pt.get_child(pt.get<std::string>("CS Block"))),
                                                                                           logLikelihood(likelihoodIn),
                                                                                           prior(priorIn),
                                                                                           forwardModel(ExtractForwardModel(likelihoodIn)),
                                                                                           noiseDensity(ExtractNoiseModel(likelihoodIn)),
                                                                                           hessType(pt.get("HessianType","GaussNewton")),
                                                                                           updateInterval(pt.get("Adapt Interval",-1)),
                                                                                           adaptStart(pt.get("Adapt Start",1)),
                                                                                           adaptEnd(pt.get("Adapt End", -1)),
                                                                                           initialHessSamps(pt.get("Initial Weight",100)),
                                                                                           hessValTol(pt.get("Hessian Tolerance",1e-4)),
                                                                                           lisValTol(pt.get("LIS Tolerance", 0.1))
{
  try{
    std::string blockName = pt.get<std::string>("Eigensolver Block");
    eigOpts = pt.get_child(blockName);
  }catch(const boost::property_tree::ptree_bad_path&){
    // Do nothing, just leave the solver options ptree empty
  }

  SetLIS(genEigVals, genEigVecs);
}

void DILIKernel::PostStep(unsigned int const t,
                          std::vector<std::shared_ptr<SamplingState>> const& state)
{
  if((updateInterval>0)&&((t%updateInterval)<=state.size())&&(t>=adaptStart)&&((t<adaptEnd)||(adaptEnd<0))){
    numLisUpdates++;
    UpdateLIS(numLisUpdates, state.at(state.size()-1)->state);
  }
}

Eigen::VectorXd DILIKernel::ToLIS(Eigen::VectorXd const& x) const
{
  return lisL->array().inverse().matrix().asDiagonal() * hessW->leftCols(lisDim).transpose()*x;
}

/** Returns a vector in the full space given a vector \f$r\f$ in the
    low dimensional LIS.
*/
Eigen::VectorXd DILIKernel::FromLIS(Eigen::VectorXd const& r) const
{
  assert(lisDim>0);
  return hessU->leftCols(lisDim)*lisL->asDiagonal()*r;
}

Eigen::VectorXd DILIKernel::ToCS(Eigen::VectorXd const& x) const
{
  return x - hessU->leftCols(lisDim) * hessW->leftCols(lisDim).transpose() * x;
}


std::vector<std::shared_ptr<SamplingState>> DILIKernel::Step(unsigned int const t,
                                                             std::shared_ptr<SamplingState> prevState)
{
  if(hessU==nullptr){
    CreateLIS(prevState->state);
  }

  std::vector<Eigen::VectorXd> splitVec(2);
  // Special handling of case when no LIS exists
  if(lisDim==0){
    splitVec.at(1) = prevState->state.at(blockInd);
    std::shared_ptr<SamplingState> splitState = std::make_shared<SamplingState>(splitVec);
    splitState->meta = prevState->meta;

    auto nextSteps = csKernel->Step(t, splitState);

    std::vector<std::shared_ptr<SamplingState>> output(nextSteps.size());
    for(unsigned int i=0; i<nextSteps.size(); ++i){
      output.at(i) = std::make_shared<SamplingState>(prevState->state);
      output.at(i)->state.at(blockInd) = nextSteps.at(i)->state.at(1);
      output.at(i)->meta = nextSteps.at(i)->meta;
    }

    return output;
  }


  splitVec.at(0) = ToLIS(prevState->state.at(blockInd) );
  splitVec.at(1) = prevState->state.at(blockInd);

  std::shared_ptr<SamplingState> splitState = std::make_shared<SamplingState>(splitVec);
  splitState->meta = prevState->meta;

  // Take the alternating Metropolis-in-Gibbs steps for the LIS and CS
  std::vector<std::shared_ptr<SamplingState>> nextSteps = lisKernel->Step(t, splitState);
  std::vector<std::shared_ptr<SamplingState>> nextSteps2 = csKernel->Step(t+nextSteps.size(), nextSteps.back());

  // Combine the states and return the results
  std::vector<std::shared_ptr<SamplingState>> output(nextSteps.size() + nextSteps2.size());
  Eigen::VectorXd x;
  for(unsigned int i=0; i<nextSteps.size(); ++i){
    x = FromLIS( nextSteps.at(i)->state.at(0) );
    x += ToCS(nextSteps.at(i)->state.at(1));

    output.at(i) = std::make_shared<SamplingState>(prevState->state);
    output.at(i)->state.at(blockInd) = x;
    //output.at(i)->meta = nextSteps.at(i)->meta;
  }

  for(unsigned int i=0; i<nextSteps2.size(); ++i){
    x = FromLIS( nextSteps2.at(i)->state.at(0) );
    x += ToCS( nextSteps2.at(i)->state.at(1) );
    output.at(i+nextSteps.size()) = std::make_shared<SamplingState>(prevState->state);
    output.at(i+nextSteps.size())->state.at(blockInd) = x;
    //output.at(i+nextSteps.size())->meta = nextSteps2.at(i)->meta;
  }

  return output;
}

void DILIKernel::PrintStatus(std::string prefix) const
{
  std::stringstream lisStream;
  lisStream << prefix << " LIS (dim=" << lisDim << "): ";
  std::string newPrefix = lisStream.str();
  lisKernel->PrintStatus(newPrefix);
  newPrefix = prefix + " CS: ";
  csKernel->PrintStatus(newPrefix);
}

void DILIKernel::SetLIS(Eigen::VectorXd const& eigVals, Eigen::MatrixXd const& eigVecs)
{
  if ((hessU == nullptr) &&
      (hessW == nullptr) &&
      (lisL  == nullptr)) {
    // Ok
  }
  else if ((hessU != nullptr) &&
           (hessW != nullptr) &&
           (lisL  != nullptr)) {
    //std::cout << ", *hessU = " << *hessU 
    //          << ", *hessW = " << *hessW 
    //          << ", *lisL = " << *lisL 
    //          << std::endl;
  }
  else {
    throw std::runtime_error("In DILIKernel::SetLIS: hessU, hessW, and lisL shall be either (i) all nullptr or (ii) all not nullptr");
  }
  bool subDimChange = false;
  if(hessU==nullptr)
    subDimChange = true;

  assert(eigVals(1)<eigVals(0));

  if (hessU == nullptr) {
    hessU = std::make_shared<Eigen::MatrixXd>(eigVecs);
    hessUQR = std::make_shared<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>(eigVecs);
  }
  else {
    *hessU = eigVecs; 
    *hessUQR = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(eigVecs);
  }

  if (hessW == nullptr) {
    hessW = std::make_shared<Eigen::MatrixXd>(prior->ApplyPrecision(eigVecs));
    hessEigVals = std::make_shared<Eigen::VectorXd>(eigVals);
  }
  else {
    *hessW = prior->ApplyPrecision(eigVecs);
    *hessEigVals = eigVals;
  }

  // Figure out the dimension of the LIS
  int oldLisDim = lisDim;
  for(lisDim=0; lisDim<eigVals.size(); ++lisDim){
    if(eigVals(lisDim)<lisValTol)
      break;
  }

  if(oldLisDim!=lisDim)
    subDimChange = true;

  // Estimate the subspace covariance based on the posterior Hessian
  Eigen::VectorXd deltaVec = eigVals.head(lisDim).array()/(1.0+eigVals.head(lisDim).array());
  if (lisL == nullptr) {
    lisL = std::make_shared<Eigen::VectorXd>((Eigen::VectorXd::Ones(lisDim) - deltaVec).array().sqrt());
  }
  else {
    *lisL = (Eigen::VectorXd::Ones(lisDim) - deltaVec).array().sqrt();
  }

  // If the dimension of the subspace has changed, we need to recreate the transition kernels
  if(subDimChange)
    UpdateKernels();
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> DILIKernel::ComputeLocalLIS(std::vector<Eigen::VectorXd> const& currState)
{
  // First, set up the hessian operator for the log-likelihood
  std::shared_ptr<LinearOperator> hessOp;
  if(hessType=="Exact"){
    assert(logLikelihood);
    hessOp = std::make_shared<HessianOperator>(logLikelihood, currState, 0, blockInd, blockInd, Eigen::VectorXd::Ones(1), -1.0, 0.0);

  }else if(hessType=="GaussNewton"){
    assert(forwardModel);
    assert(noiseDensity);
    hessOp = std::make_shared<GaussNewtonOperator>(forwardModel, noiseDensity, currState, blockInd, -1.0, 0.0);

  }else{
    std::cerr << "\nERROR: Unrecognized Hessian type.  Options are \"Exact\" or \"GaussNewton\".\n\n";
  }

  // Set up the prior precision operator
  std::shared_ptr<LinearOperator> precOp = std::make_shared<GaussianOperator>(prior, Gaussian::Precision);
  std::shared_ptr<LinearOperator> covOp = std::make_shared<GaussianOperator>(prior, Gaussian::Covariance);

  // Solve the generalized Eigenvalue problem using StochasticEigenSolver
  eigOpts.put("AbsoluteTolerance", lisValTol); // <- We are computing the local LIS, so use the local tolerance
  if(lisDim>0){
    eigOpts.put("ExpectedRank", lisDim);
    eigOpts.put("NumEigs", 2*lisDim);
  }

  StochasticEigenSolver solver(eigOpts);
  solver.compute(hessOp, precOp, covOp);

  return std::make_pair(solver.eigenvectors(), solver.eigenvalues());
}

void DILIKernel::CreateLIS(std::vector<Eigen::VectorXd> const& currState)
{
  numLisUpdates = 0;

  Eigen::MatrixXd vecs;
  Eigen::VectorXd vals;
  std::tie(vecs,vals) = ComputeLocalLIS(currState);

  SetLIS(vals, vecs);
}

void DILIKernel::UpdateKernels()
{
  lisToFull = std::make_shared<LIS2Full>(hessU,lisL);
  fullToCS = std::make_shared<CSProjector>(hessU, hessW, lisDim);

  // Create a new graph for the split likelihood
  std::shared_ptr<WorkGraph> graph = std::make_shared<WorkGraph>();
  graph->AddNode(std::make_shared<SumPiece>(prior->Dimension(), 2), "Parameters");
  graph->AddNode(lisToFull, "Informed Parameters");
  graph->AddNode(fullToCS, "Complementary Parameters");
  graph->AddNode(logLikelihood, "Likelihood");
  graph->AddNode(prior->AsDensity(), "Prior");
  graph->AddNode(std::make_shared<DensityProduct>(2), "Posterior");
  graph->AddEdge("Informed Parameters",0, "Parameters",0);
  graph->AddEdge("Complementary Parameters",0, "Parameters",1);
  graph->AddEdge("Parameters",0,"Prior",0);
  graph->AddEdge("Parameters",0,"Likelihood",0);
  graph->AddEdge("Prior",0,"Posterior",0);
  graph->AddEdge("Likelihood",0,"Posterior",1);

  auto prob = std::make_shared<SamplingProblem>(graph->CreateModPiece("Posterior"));

  lisKernelOpts.put("BlockIndex",0);
  lisKernel = TransitionKernel::Construct(lisKernelOpts,prob);

  csKernelOpts.put("BlockIndex",1);
  csKernel = TransitionKernel::Construct(csKernelOpts,prob);

}

void DILIKernel::UpdateLIS(unsigned int                        numSamps,
                           std::vector<Eigen::VectorXd> const& currState)
{
  std::shared_ptr<LinearOperator> hessOp, newOp, precOp, covOp;
  if(hessType=="Exact"){
    newOp = std::make_shared<HessianOperator>(logLikelihood, currState, 0, blockInd, blockInd, Eigen::VectorXd::Ones(1), -1.0, 0.0);
  }else if(hessType=="GaussNewton"){
    newOp = std::make_shared<GaussNewtonOperator>(forwardModel, noiseDensity, currState, blockInd, -1.0, 0.0);
  }else{
    std::cerr << "\nERROR: Unrecognized Hessian type.  Options are \"Exact\" or \"GaussNewton\".\n\n";
  }

  hessOp = std::make_shared<AverageHessian>(numSamps+initialHessSamps, hessUQR, hessW, hessEigVals, newOp);
  precOp = std::make_shared<GaussianOperator>(prior, Gaussian::Precision);
  covOp = std::make_shared<GaussianOperator>(prior, Gaussian::Covariance);

  eigOpts.put("AbsoluteTolerance", hessValTol);
  if(hessEigVals!=nullptr){
    eigOpts.put("ExpectedRank", hessEigVals->size());
    eigOpts.put("NumEigs", 2*hessEigVals->size());
  }

  StochasticEigenSolver solver(eigOpts);
  solver.compute(hessOp, precOp, covOp);

  SetLIS(solver.eigenvalues(), solver.eigenvectors());
}

std::shared_ptr<muq::Modeling::GaussianBase> DILIKernel::ExtractPrior(std::shared_ptr<AbstractSamplingProblem> const& problem,
                                                                      std::string                              const& nodeName)
{
  // First, try to cast the abstract problem to a regular sampling problem
  auto regProb = std::dynamic_pointer_cast<SamplingProblem>(problem);
  if(!regProb){
    throw std::runtime_error("In DILIKernel::ExtractPrior: Could not cast AbstractSamplingProblem instance into SamplingProblem.");
  }

  // Try to pull out the ModGraphPiece from the sampling problem and try to cast as ModGraphPiece
  auto postPiece = regProb->GetDistribution();
  auto postGraphPiece = std::dynamic_pointer_cast<ModGraphPiece>(postPiece);
  if(!postGraphPiece){
    throw std::runtime_error("In DILIKernel::ExtractPrior: Could not cast Posterior ModPiece to ModGraphPiece.");
  }

  std::shared_ptr<muq::Modeling::WorkPiece> priorWork = postGraphPiece->GetGraph()->GetPiece(nodeName);
  std::shared_ptr<muq::Modeling::Density> priorDens = std::dynamic_pointer_cast<Density>(priorWork);
  if(!priorDens){
    throw std::runtime_error("In DILIKernel::ExtractPrior:  Could not cast prior WorkPiece to Density.");
  }

  std::shared_ptr<muq::Modeling::GaussianBase> output = std::dynamic_pointer_cast<GaussianBase>(priorDens->GetDistribution());
  if(!output){
    throw std::runtime_error("In DILIKernel::ExtractPrior:  Could not cast prior distribution to GaussianBase.");
  }

  return output;
}

std::shared_ptr<muq::Modeling::ModPiece> DILIKernel::ExtractNoiseModel(std::shared_ptr<muq::Modeling::ModPiece> const& likelihood)
{
  auto likelyGraphPiece = std::dynamic_pointer_cast<ModGraphPiece>(likelihood);
  if(!likelyGraphPiece){
    throw std::runtime_error("In DILIKernel::ExtractNoiseModel: Could not cast likelihood ModPiece to ModGraphPiece.");
  }

  return likelyGraphPiece->GetOutputPiece();
}

std::shared_ptr<muq::Modeling::ModPiece> DILIKernel::ExtractForwardModel(std::shared_ptr<muq::Modeling::ModPiece> const& likelihoodIn)
{
  if(likelihoodIn->inputSizes.size()!=1)
    throw std::runtime_error("In DILIKernel::ExtractForwardModel: Could not detect forward model because likelihood piece has more than one input.");

  auto likelyGraphPiece = std::dynamic_pointer_cast<ModGraphPiece>(likelihoodIn);
  if(!likelyGraphPiece){
    throw std::runtime_error("In DILIKernel::ExtractForwardModel: Could not cast likelihood ModPiece to ModGraphPiece.");
  }

  // Get the name of the output node
  auto graph = likelyGraphPiece->GetGraph();
  std::vector<std::string> modelNames = graph->GetParents(likelyGraphPiece->GetOutputName());

  return likelyGraphPiece->GetSubModel(modelNames.at(0));
}

std::shared_ptr<muq::Modeling::ModPiece> DILIKernel::ExtractLikelihood(std::shared_ptr<AbstractSamplingProblem> const& problem,
                                                                       std::string                              const& nodeName)
{
  // First, try to cast the abstract problem to a regular sampling problem
  auto regProb = std::dynamic_pointer_cast<SamplingProblem>(problem);
  if(!regProb){
    throw std::runtime_error("In DILIKernel::ExtractLikelihood: Could not cast AbstractSamplingProblem instance into SamplingProblem.");
  }

  // Try to pull out the ModGraphPiece from the sampling problem and try to cast as ModGraphPiece
  auto postPiece = regProb->GetDistribution();
  auto postGraphPiece = std::dynamic_pointer_cast<ModGraphPiece>(postPiece);
  if(!postGraphPiece){
    throw std::runtime_error("In DILIKernel::ExtractLikelihood: Could not cast Posterior ModPiece to ModGraphPiece.");
  }

  std::shared_ptr<muq::Modeling::ModPiece> likelihood = postGraphPiece->GetSubModel(nodeName);

  return likelihood;
}

std::shared_ptr<muq::Modeling::ModPiece> DILIKernel::CreateLikelihood(std::shared_ptr<muq::Modeling::ModPiece> const& forwardModel,
                                                                      std::shared_ptr<muq::Modeling::ModPiece> const& noiseDensity)
{
  WorkGraph graph;
  graph.AddNode(forwardModel, "Forward Model");
  graph.AddNode(noiseDensity, "Likelihood");
  graph.AddEdge("Forward Model", 0, "Likelihood",0);
  return graph.CreateModPiece("Likelihood");
}
