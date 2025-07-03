#include "MUQ/Approximation/GaussianProcesses/ConcatenateKernel.h"
#include "MUQ/Modeling/LinearAlgebra/BlockDiagonalOperator.h"
#include "MUQ/Modeling/LinearAlgebra/ZeroOperator.h"
#include "MUQ/Modeling/LinearSDE.h"

using namespace muq::Approximation;
using namespace muq::Modeling;

ConcatenateKernel::ConcatenateKernel(std::vector<std::shared_ptr<KernelBase>> const& kernelsIn) : KernelBase(kernelsIn.at(0)->inputDim,
                                                                                                             CountCoDims(kernelsIn),
                                                                                                             CountParams(kernelsIn)),
                                                                                                  kernels(kernelsIn)
{
  // Make sure all the input sizes are the same
  for(int i=1; i<kernels.size(); ++i)
    assert(kernels.at(i)->inputDim == kernels.at(0)->inputDim);

  // Set all the parameters from what was in the kernels before
  cachedParams.resize(numParams);
  int paramCnt = 0;
  for(int i=0; i<kernels.size(); ++i){
    cachedParams.segment(paramCnt, kernels.at(i)->numParams) = kernels.at(i)->GetParams();
    paramCnt += kernels.at(i)->numParams;
  }
}


void ConcatenateKernel::FillBlock(Eigen::Ref<const Eigen::VectorXd> const& x1,
                       Eigen::Ref<const Eigen::VectorXd> const& x2,
                       Eigen::Ref<const Eigen::VectorXd> const& params,
                       Eigen::Ref<Eigen::MatrixXd>              block) const
{
  block = Eigen::MatrixXd::Zero(coDim, coDim);
  int paramCnt = 0;
  int codimCnt = 0;

  for(int i=0; i<kernels.size(); ++i){

    kernels.at(i)->FillBlock(x1,
                             x2,
                             params.segment(paramCnt, kernels.at(i)->numParams),
                             block.block(codimCnt,codimCnt,kernels.at(i)->coDim, kernels.at(i)->coDim));

    codimCnt += kernels.at(i)->coDim;
    paramCnt += kernels.at(i)->numParams;
  }
}

void ConcatenateKernel::FillPosDerivBlock(Eigen::Ref<const Eigen::VectorXd> const& x1,
                               Eigen::Ref<const Eigen::VectorXd> const& x2,
                               Eigen::Ref<const Eigen::VectorXd> const& params,
                               std::vector<int>                  const& wrts,
                               Eigen::Ref<Eigen::MatrixXd>              block) const
{
  block = Eigen::MatrixXd::Zero(coDim, coDim);

  int paramCnt = 0;
  int codimCnt = 0;
  for(int i=0; i<kernels.size(); ++i){

      kernels.at(i)->FillPosDerivBlock(x1,
                                       x2,
                                       params.segment(paramCnt, kernels.at(i)->numParams),
                                       wrts,
                                       block.block(codimCnt,codimCnt,kernels.at(i)->coDim, kernels.at(i)->coDim));

      codimCnt += kernels.at(i)->coDim;
      paramCnt += kernels.at(i)->numParams;
  }
}

unsigned int ConcatenateKernel::CountCoDims(std::vector<std::shared_ptr<KernelBase>> kernelsIn)
{
  int cnt = 0;
  for(auto& kernel : kernelsIn)
    cnt += kernel->coDim;
  return cnt;
}
unsigned int ConcatenateKernel::CountParams(std::vector<std::shared_ptr<KernelBase>> kernelsIn)
{
  int cnt = 0;
  for(auto& kernel : kernelsIn)
    cnt += kernel->numParams;
  return cnt;
}


std::tuple<std::shared_ptr<muq::Modeling::LinearSDE>, std::shared_ptr<muq::Modeling::LinearOperator>, Eigen::MatrixXd> ConcatenateKernel::GetStateSpace(boost::property_tree::ptree sdeOptions) const
{   
  unsigned int numKernels = kernels.size();
  std::vector<std::shared_ptr<muq::Modeling::LinearSDE>> sdes(numKernels); // SDE definition for each kernel
  std::vector<std::shared_ptr<muq::Modeling::LinearOperator>> obsOps(numKernels); // Observation operators for each kernel
  std::vector<Eigen::MatrixXd> statCovs(numKernels); // Stationary coveriances for each kernel

  unsigned int totalStatSize = 0; // Size of concatenated stationary covariance matrix
  unsigned int totalProcSize = 0; // Size of concatenated process noise covariance matrix
  for(unsigned int i=0; i<numKernels; ++i){
    std::tie(sdes.at(i), obsOps.at(i), statCovs.at(i)) = kernels.at(i)->GetStateSpace(sdeOptions);
    totalStatSize += statCovs.at(i).rows();
    if(sdes.at(i)->GetL())
      totalProcSize += sdes.at(i)->GetL()->cols();
  }

  std::shared_ptr<muq::Modeling::LinearOperator> combinedObsOp = std::make_shared<BlockDiagonalOperator>(obsOps);

  // Create a new SDE:
  std::vector<std::shared_ptr<muq::Modeling::LinearOperator>> allFs;
  std::vector<std::shared_ptr<muq::Modeling::LinearOperator>> allLs;
  Eigen::MatrixXd newProcessCov = Eigen::MatrixXd::Zero(totalProcSize, totalProcSize);
  Eigen::MatrixXd newStatCov = Eigen::MatrixXd::Zero(totalStatSize, totalStatSize);

  unsigned int currStatRow = 0;
  unsigned int currProcRow = 0;
  for(unsigned int i=0; i<numKernels; ++i){

    // Collect stochastic parts
    if(sdes.at(i)->GetL()){
      allLs.push_back(sdes.at(i)->GetL());
    
      newProcessCov.block(currProcRow,currProcRow, allLs.back()->cols(), allLs.back()->cols()) = sdes.at(i)->GetQ();
      currProcRow += allLs.back()->cols();
    }else{
      allLs.push_back(std::make_shared<ZeroOperator>(sdes.at(i)->GetF()->rows(),0));
    }

    // Collect deterministic parts
    allFs.push_back( sdes.at(i)->GetF() );
    newStatCov.block(currStatRow,currStatRow, statCovs.at(i).rows(), statCovs.at(i).cols()) = statCovs.at(i); 
    currStatRow += statCovs.at(i).rows();
  }

  std::shared_ptr<muq::Modeling::LinearOperator> combinedF = std::make_shared<BlockDiagonalOperator>(allFs);
  std::shared_ptr<muq::Modeling::LinearOperator> combinedL = std::make_shared<BlockDiagonalOperator>(allLs);


  auto combinedSDE = std::make_shared<muq::Modeling::LinearSDE>(combinedF, combinedL, newProcessCov, sdeOptions);

  return std::make_tuple(combinedSDE, combinedObsOp, newStatCov); 
}
    