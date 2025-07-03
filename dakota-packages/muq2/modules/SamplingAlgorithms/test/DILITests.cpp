#include <gtest/gtest.h>

#include <boost/property_tree/ptree.hpp>

#include "MUQ/Modeling/WorkGraph.h"
#include "MUQ/Modeling/WorkGraphPiece.h"
#include "MUQ/Modeling/ModGraphPiece.h"
#include "MUQ/Modeling/CwiseOperators/CwiseUnaryOperator.h"

#include "MUQ/Modeling/Distributions/Gaussian.h"
#include "MUQ/Modeling/Distributions/DensityProduct.h"
#include "MUQ/Modeling/LinearAlgebra/HessianOperator.h"

#include "MUQ/Utilities/RandomGenerator.h"

#include "MUQ/SamplingAlgorithms/SingleChainMCMC.h"
#include "MUQ/SamplingAlgorithms/DILIKernel.h"
#include "MUQ/SamplingAlgorithms/MHProposal.h"
#include "MUQ/SamplingAlgorithms/MHKernel.h"

#include "MUQ/Approximation/GaussianProcesses/GaussianProcess.h"
#include "MUQ/Approximation/GaussianProcesses/MaternKernel.h"

namespace pt = boost::property_tree;
using namespace muq::Modeling;
using namespace muq::Approximation;
using namespace muq::SamplingAlgorithms;
using namespace muq::Utilities;

std::shared_ptr<Gaussian> DILITest_CreatePrior(unsigned int N)
{
  Eigen::MatrixXd xs = Eigen::VectorXd::LinSpaced(N, 0,1).transpose();

  MaternKernel kern(1, 1.0, 0.05, 5.0/2.0);
  ZeroMean mu(1,1);
  GaussianProcess gpPrior(mu,kern);

  return gpPrior.Discretize(xs);
}

TEST(MCMC, DILI_AverageHessian) {

  // Create two random SPD matrices
  unsigned int dim = 10;
  unsigned int lowDim = 5;
  double nugget = 1e-3;

  Eigen::MatrixXd random = Eigen::MatrixXd::Random(dim,lowDim);
  Eigen::MatrixXd oldHess = random*random.transpose();

  random = Eigen::MatrixXd::Random(dim,lowDim);
  Eigen::MatrixXd newHess = random*random.transpose();

  random = Eigen::MatrixXd::Random(dim,dim);
  Eigen::MatrixXd priorCov = random*random.transpose() + nugget*Eigen::MatrixXd::Identity(dim,dim);
  Eigen::MatrixXd priorPrec = priorCov.ldlt().solve(Eigen::MatrixXd::Identity(dim,dim));

  Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> es(oldHess,priorPrec);

  auto u = std::make_shared<Eigen::MatrixXd>(es.eigenvectors().rightCols(lowDim+1));
  auto w = std::make_shared<Eigen::MatrixXd>(priorPrec * (*u));
  auto uQR = std::make_shared<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>(*u);
  auto vals = std::make_shared<Eigen::VectorXd>(es.eigenvalues().tail(lowDim+1));

  // First, check that we're interpreting the QR decomposition correctly
  Eigen::MatrixXd thinQ = uQR->householderQ().setLength(uQR->nonzeroPivots()) * Eigen::MatrixXd::Identity(u->rows(), uQR->rank());
  Eigen::MatrixXd uInvT = uQR->colsPermutation() * uQR->matrixR().topLeftCorner(uQR->rank(), uQR->rank()).template triangularView<Eigen::Upper>().solve(thinQ.transpose());

  Eigen::MatrixXd QR = thinQ*uQR->matrixR().topLeftCorner(uQR->rank(), uQR->rank()).template triangularView<Eigen::Upper>() *uQR->colsPermutation().inverse();
  EXPECT_NEAR(0, (QR-(*u)).array().abs().maxCoeff(), 1e-10);

  Eigen::MatrixXd oldHessApprox = priorPrec * QR * vals->asDiagonal() * uInvT;

  auto newOp = LinearOperator::Create(newHess);

  unsigned int numSamps = 1;
  auto avgHess = std::make_shared<AverageHessian>(numSamps, uQR, w, vals, newOp);

  Eigen::MatrixXd tester = Eigen::MatrixXd::Identity(dim,dim);

  Eigen::MatrixXd truth = ((numSamps)/(numSamps+1.0))*oldHess + newHess/(numSamps+1.0);
  Eigen::MatrixXd op = avgHess->Apply(tester);
}


TEST(MCMC, DILIKernel_HessianOperator) {

  const unsigned int numNodes = 100;
  const unsigned int dataDim = 10;
  const double noiseStd = 1e-3;

  std::shared_ptr<Gaussian> prior = DILITest_CreatePrior(numNodes);

  Eigen::MatrixXd forwardMat = Eigen::MatrixXd::Zero(dataDim,numNodes);
  forwardMat.row(0) = Eigen::RowVectorXd::Ones(numNodes);
  for(unsigned int i=1; i<dataDim; ++i)
    forwardMat(i,10*i) = 1.0;

  Eigen::MatrixXd trueHess = (1.0/(noiseStd*noiseStd))*forwardMat.transpose() * forwardMat;

  auto forwardMod = LinearOperator::Create(forwardMat);
  Eigen::VectorXd data = forwardMod->Apply(prior->Sample());

  auto noiseModel = std::make_shared<Gaussian>(data, noiseStd*noiseStd*Eigen::VectorXd::Ones(dataDim));

  WorkGraph graph;
  graph.AddNode(std::make_shared<IdentityOperator>(numNodes), "Parameters");
  graph.AddNode(prior->AsDensity(), "Prior");
  graph.AddNode(forwardMod, "ForwardModel");
  graph.AddNode(noiseModel->AsDensity(), "Likelihood");
  graph.AddNode(std::make_shared<DensityProduct>(2),"Posterior");
  graph.AddEdge("Parameters",0,"Prior",0);
  graph.AddEdge("Parameters",0,"ForwardModel",0);
  graph.AddEdge("ForwardModel",0,"Likelihood",0);
  graph.AddEdge("Prior",0,"Posterior",0);
  graph.AddEdge("Likelihood",0,"Posterior",1);

  auto logLikely = graph.CreateModPiece("Likelihood");

  std::vector<Eigen::VectorXd> inputs{prior->GetMean()};
  Eigen::VectorXd sens = Eigen::VectorXd::Ones(1);
  auto hessOp = std::make_shared<HessianOperator>(logLikely, inputs, 0, 0, 0, sens, -1.0);

  Eigen::MatrixXd hess(numNodes,numNodes);
  for(unsigned int i=0; i<numNodes; ++i){
    Eigen::VectorXd unitVec = Eigen::VectorXd::Zero(numNodes);
    unitVec(i) = 1.0;
    hess.col(i) = hessOp->Apply(unitVec);
  }

  for(unsigned int j=0;j<numNodes;++j){
    for(unsigned int i=0; i<numNodes; ++i){
      EXPECT_NEAR(trueHess(i,j), hess(i,j), 1e-5*std::abs(trueHess(i,j)));
    }
  }
}


TEST(MCMC, DILIKernel_SubspaceCheck) {

  const unsigned int numNodes = 100;
  const unsigned int dataDim = 3;
  const double noiseStd = 1e-2;

  std::shared_ptr<Gaussian> prior = DILITest_CreatePrior(numNodes);

  Eigen::MatrixXd forwardMat = Eigen::MatrixXd::Zero(dataDim,numNodes);
  forwardMat.row(0) = Eigen::RowVectorXd::Ones(numNodes)/numNodes;
  for(unsigned int i=1; i<dataDim; ++i)
    forwardMat(i,10*i) = 1.0;

  auto forwardMod = LinearOperator::Create(forwardMat);
  Eigen::VectorXd trueField = Eigen::VectorXd::Zero(numNodes);
  Eigen::VectorXd data = forwardMod->Apply(trueField);

  auto noiseModel = std::make_shared<Gaussian>(data, noiseStd*noiseStd*Eigen::VectorXd::Ones(dataDim));

  WorkGraph graph;
  graph.AddNode(std::make_shared<IdentityOperator>(numNodes), "Parameters");
  graph.AddNode(prior->AsDensity(), "Prior");
  graph.AddNode(forwardMod, "ForwardModel");
  graph.AddNode(noiseModel->AsDensity(), "Likelihood");
  graph.AddNode(std::make_shared<DensityProduct>(2),"Posterior");
  graph.AddEdge("Parameters",0,"Prior",0);
  graph.AddEdge("Parameters",0,"ForwardModel",0);
  graph.AddEdge("ForwardModel",0,"Likelihood",0);
  graph.AddEdge("Prior",0,"Posterior",0);
  graph.AddEdge("Likelihood",0,"Posterior",1);

  auto logLikely = graph.CreateModPiece("Likelihood");

  pt::ptree pt;
  const unsigned int numSamps = 10000;
  pt.put("NumSamples",numSamps);
  pt.put("BurnIn", 0);
  pt.put("PrintLevel",0);
  pt.put("HessianType","Exact");
  pt.put("Adapt Interval", 1000);

  pt.put("Eigensolver Block", "EigOpts");
  pt.put("EigOpts.NumEigs",15); // Maximum number of generalized eigenvalues to compute (e.g., maximum LIS dimension)
  pt.put("EigOpts.ExpectedRank", dataDim);
  pt.put("EigOpts.OversamplingFactor", 5);
  //pt.put("EigOpts.BlockSize",20);
  //pt.put("EigOpts.Verbosity",3);

  pt.put("LIS Block", "LIS");
  pt.put("LIS.Method", "MHKernel");
  pt.put("LIS.Proposal","MyProposal");
  pt.put("LIS.MyProposal.Method","MHProposal");
  pt.put("LIS.MyProposal.ProposalVariance", 1.0);

  pt.put("CS Block", "CS");
  pt.put("CS.Method", "MHKernel");
  pt.put("CS.Proposal","MyProposal");
  pt.put("CS.MyProposal.Method", "CrankNicolsonProposal");
  pt.put("CS.MyProposal.Beta",1.0);
  pt.put("CS.MyProposal.PriorNode","Prior");


  std::shared_ptr<Gaussian> truePost = prior->Condition(forwardMat, data, noiseStd*noiseStd*Eigen::VectorXd::Ones(dataDim));


  // create a sampling problem
  auto problem = std::make_shared<SamplingProblem>(graph.CreateModPiece("Posterior"));
  auto kernel = std::make_shared<DILIKernel>(pt, problem, prior, logLikely);

  std::vector<Eigen::VectorXd> currState(1);
  currState[0] = truePost->GetMean();

  kernel->CreateLIS(currState);

  unsigned int lisDim = kernel->LISDim();
  auto& U = kernel->LISVecs().leftCols(lisDim);
  auto& W = kernel->LISW().leftCols(lisDim);
  auto& L = kernel->LISL();
  auto& vals = kernel->LISVals().head(lisDim);

  // Check to make sure round trips to/from LIS work as expected.
  Eigen::VectorXd r = Eigen::VectorXd::Random(lisDim);
  Eigen::VectorXd x = kernel->FromLIS(r);
  Eigen::VectorXd r2 = kernel->ToLIS(x);
  EXPECT_NEAR(0,(r2-r).array().abs().maxCoeff(), 1e-12);

  x = prior->Sample();
  r = kernel->ToLIS(x);
  Eigen::VectorXd x2 = kernel->FromLIS(r) + kernel->ToCS(x);
  EXPECT_NEAR(0,(x2-x).array().abs().maxCoeff(), 1e-12);

  /* A point y on the LIS is given by the Full-To-LIS transformation:
       y=(L^{-1} W^T) x
     Similarly, the LIS-TO-Full transformation is given by
       x = ULy
     Projecting onto the complementary space is given by
       x_c = (I-UW^T)x
  */

  // Compute the true covariance on the LIS and compare to the cov created by DILI
  Eigen::MatrixXd trueFullCov = truePost->GetCovariance();
  Eigen::MatrixXd trueLisCov = W.transpose() * trueFullCov * W;
  Eigen::MatrixXd diliLisCov = L.array().square().matrix().asDiagonal();

  for(int i=0; i<trueLisCov.rows(); ++i){
    for(int j=0; j<=i; ++j){
      EXPECT_NEAR(trueLisCov(i,j), diliLisCov(i,j), 1e-13);
    }
  }

  // Compute the full posterior covariance from the LIS cov
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(U.rows(), U.rows());
  Eigen::MatrixXd diliFullCov = (I-U*W.transpose())*trueFullCov*(I-W*U.transpose()) + U*diliLisCov*U.transpose();

  EXPECT_NEAR(0, (diliFullCov-trueFullCov).array().abs().maxCoeff(), 1e-12);
}



TEST(MCMC, DILIKernel_ManualConstruction) {

  const unsigned int numNodes = 100;
  const unsigned int dataDim = 5;
  const double noiseStd = 1e-2;

  std::shared_ptr<Gaussian> prior = DILITest_CreatePrior(numNodes);

  Eigen::MatrixXd forwardMat = Eigen::MatrixXd::Zero(dataDim,numNodes);
  forwardMat.row(0) = Eigen::RowVectorXd::Ones(numNodes)/numNodes;
  for(unsigned int i=1; i<dataDim; ++i)
    forwardMat(i,10*i) = 1.0;

  auto forwardMod = LinearOperator::Create(forwardMat);
  Eigen::VectorXd trueField = Eigen::VectorXd::Zero(numNodes);
  Eigen::VectorXd data = forwardMod->Apply(trueField);

  auto noiseModel = std::make_shared<Gaussian>(data, noiseStd*noiseStd*Eigen::VectorXd::Ones(dataDim));

  WorkGraph graph;
  graph.AddNode(std::make_shared<IdentityOperator>(numNodes), "Parameters");
  graph.AddNode(prior->AsDensity(), "Prior");
  graph.AddNode(forwardMod, "ForwardModel");
  graph.AddNode(noiseModel->AsDensity(), "Likelihood");
  graph.AddNode(std::make_shared<DensityProduct>(2),"Posterior");
  graph.AddEdge("Parameters",0,"Prior",0);
  graph.AddEdge("Parameters",0,"ForwardModel",0);
  graph.AddEdge("ForwardModel",0,"Likelihood",0);
  graph.AddEdge("Prior",0,"Posterior",0);
  graph.AddEdge("Likelihood",0,"Posterior",1);

  auto logLikely = graph.CreateModPiece("Likelihood");

  pt::ptree pt;
  const unsigned int numSamps = 30000;
  pt.put("NumSamples",numSamps);
  pt.put("BurnIn", 1000);
  pt.put("PrintLevel",0);
  pt.put("HessianType","Exact");
  pt.put("Adapt Interval", 1000);

  pt.put("Eigensolver Block", "EigOpts");
  pt.put("EigOpts.NumEigs",15); // Maximum number of generalized eigenvalues to compute (e.g., maximum LIS dimension)
  pt.put("EigOpts.RelativeTolerance", 1e-3); // Fraction of the largest eigenvalue used as stopping criteria on how many eigenvalues to compute
  pt.put("EigOpts.AbsoluteTolerance",0.0); // Minimum allowed eigenvalue
  pt.put("EigOpts.ExpectedRank", 5);
  pt.put("EigOpts.OversamplingFactor", 2);
  //pt.put("EigOpts.BlockSize",20);
  //pt.put("EigOpts.Verbosity",0);

  pt.put("LIS Block", "LIS");
  pt.put("LIS.Method", "MHKernel");
  pt.put("LIS.Proposal","MyProposal");
  pt.put("LIS.MyProposal.Method","MHProposal");
  pt.put("LIS.MyProposal.ProposalVariance", 1.0);

  pt.put("CS Block", "CS");
  pt.put("CS.Method", "MHKernel");
  pt.put("CS.Proposal","MyProposal");
  pt.put("CS.MyProposal.Method", "CrankNicolsonProposal");
  pt.put("CS.MyProposal.Beta",1.0);
  pt.put("CS.MyProposal.PriorNode","Prior");

  // The posterior is Gaussian in this case, so we can analytically compute the posterior mean and covariance for comparison
  std::shared_ptr<Gaussian> truePost = prior->Condition(forwardMat, data, noiseStd*noiseStd*Eigen::VectorXd::Ones(dataDim));

  // create a sampling problem
  auto problem = std::make_shared<SamplingProblem>(graph.CreateModPiece("Posterior"));

  auto extractLikely = DILIKernel::ExtractLikelihood(problem, "Likelihood");
  EXPECT_EQ(numNodes, extractLikely->inputSizes(0));
  EXPECT_EQ(1, extractLikely->outputSizes(0));

  auto kernel = std::make_shared<DILIKernel>(pt, problem, prior, logLikely);

  auto sampler = std::make_shared<SingleChainMCMC>(pt, std::vector<std::shared_ptr<TransitionKernel>>{kernel});

  auto samps = sampler->Run(truePost->Sample()); // Use a true posterior sample to avoid burnin


  // Check the posterior moments
  Eigen::VectorXd sampMean = samps->Mean();
  Eigen::MatrixXd sampCov = samps->Covariance();
  Eigen::VectorXd trueMean = truePost->GetMean();
  Eigen::MatrixXd trueCov = truePost->GetCovariance();

  Eigen::VectorXd ess = samps->ESS();
  Eigen::VectorXd estStd = samps->StandardError();

  for(int i=0; i<numNodes; ++i)
    EXPECT_NEAR(trueMean(i), sampMean(i), 4.0*estStd(i));

  // Check to make sure the CS acceptance rate is close to one
  auto csKernel = std::dynamic_pointer_cast<MHKernel>(kernel->CSKernel());
  EXPECT_LT(0.9, csKernel->AcceptanceRate());
}


TEST(MCMC, DILIKernel_AutomaticConstruction) {

  const unsigned int numNodes = 100;
  const unsigned int dataDim = 5;
  const double noiseStd = 1e-2;

  std::shared_ptr<Gaussian> prior = DILITest_CreatePrior(numNodes);

  Eigen::MatrixXd forwardMat = Eigen::MatrixXd::Zero(dataDim,numNodes);
  forwardMat.row(0) = Eigen::RowVectorXd::Ones(numNodes)/numNodes;
  for(unsigned int i=1; i<dataDim; ++i)
    forwardMat(i,10*i) = 1.0;

  auto forwardMod = LinearOperator::Create(forwardMat);
  Eigen::VectorXd trueField = Eigen::VectorXd::Zero(numNodes);
  Eigen::VectorXd data = forwardMod->Apply(trueField);

  auto noiseModel = std::make_shared<Gaussian>(data, noiseStd*noiseStd*Eigen::VectorXd::Ones(dataDim));

  WorkGraph graph;
  graph.AddNode(std::make_shared<IdentityOperator>(numNodes), "Parameters");
  graph.AddNode(prior->AsDensity(), "Prior");
  graph.AddNode(forwardMod, "ForwardModel");
  graph.AddNode(noiseModel->AsDensity(), "Likelihood");
  graph.AddNode(std::make_shared<DensityProduct>(2),"Posterior");
  graph.AddEdge("Parameters",0,"Prior",0);
  graph.AddEdge("Parameters",0,"ForwardModel",0);
  graph.AddEdge("ForwardModel",0,"Likelihood",0);
  graph.AddEdge("Prior",0,"Posterior",0);
  graph.AddEdge("Likelihood",0,"Posterior",1);

  auto logLikely = graph.CreateModPiece("Likelihood");

  pt::ptree pt;
  const unsigned int numSamps = 40000;
  pt.put("NumSamples",numSamps);
  pt.put("BurnIn", 0);
  pt.put("PrintLevel",3);
  pt.put("HessianType","Exact");
  pt.put("Adapt Interval", 0);
  pt.put("Prior Node", "Prior");
  pt.put("Likelihood Node", "Likelihood");

  pt.put("Eigensolver Block", "EigOpts");
  pt.put("EigOpts.NumEigs",15); // Maximum number of generalized eigenvalues to compute (e.g., maximum LIS dimension)
  pt.put("EigOpts.ExpectedRank", 5);
  pt.put("EigOpts.OversamplingFactor", 2);

  pt.put("LIS Block", "LIS");
  pt.put("LIS.Method", "MHKernel");
  pt.put("LIS.Proposal","MyProposal");
  pt.put("LIS.MyProposal.Method","MHProposal");
  pt.put("LIS.MyProposal.ProposalVariance", 1.0);

  pt.put("CS Block", "CS");
  pt.put("CS.Method", "MHKernel");
  pt.put("CS.Proposal","MyProposal");
  pt.put("CS.MyProposal.Method", "CrankNicolsonProposal");
  pt.put("CS.MyProposal.Beta",1.0);
  pt.put("CS.MyProposal.PriorNode","Prior");

  // The posterior is Gaussian in this case, so we can analytically compute the posterior mean and covariance for comparison
  std::shared_ptr<Gaussian> truePost = prior->Condition(forwardMat, data, noiseStd*noiseStd*Eigen::VectorXd::Ones(dataDim));

  // create a sampling problem
  auto problem = std::make_shared<SamplingProblem>(graph.CreateModPiece("Posterior"));

  auto kernel = std::make_shared<DILIKernel>(pt, problem);

  auto sampler = std::make_shared<SingleChainMCMC>(pt, std::vector<std::shared_ptr<TransitionKernel>>{kernel});

  auto samps = sampler->Run(truePost->Sample()); // Use a true posterior sample to avoid burnin


  // Check the posterior moments
  Eigen::VectorXd sampMean = samps->Mean();
  Eigen::MatrixXd sampCov = samps->Covariance();
  Eigen::VectorXd trueMean = truePost->GetMean();
  Eigen::MatrixXd trueCov = truePost->GetCovariance();

  Eigen::VectorXd mcse = samps->StandardError("Batch");

  for(int i=0; i<numNodes; ++i){

    // Estimator variance
    EXPECT_NEAR(trueMean(i), sampMean(i), 4.0*std::sqrt(mcse(i)));
  }
}

TEST(MCMC, DILIKernel_AutomaticGaussNewton) {

  const unsigned int numNodes = 100;
  const unsigned int dataDim = 5;
  const double noiseStd = 1e-2;

  std::shared_ptr<Gaussian> prior = DILITest_CreatePrior(numNodes);

  Eigen::MatrixXd forwardMat = Eigen::MatrixXd::Zero(dataDim,numNodes);
  forwardMat.row(0) = Eigen::RowVectorXd::Ones(numNodes)/numNodes;
  for(unsigned int i=1; i<dataDim; ++i)
    forwardMat(i,10*i) = 1.0;

  auto forwardMod = LinearOperator::Create(forwardMat);
  Eigen::VectorXd trueField = Eigen::VectorXd::Zero(numNodes);
  Eigen::VectorXd data = forwardMod->Apply(trueField);

  auto noiseModel = std::make_shared<Gaussian>(data, noiseStd*noiseStd*Eigen::VectorXd::Ones(dataDim));

  WorkGraph graph;
  graph.AddNode(std::make_shared<IdentityOperator>(numNodes), "Parameters");
  graph.AddNode(prior->AsDensity(), "Prior");
  graph.AddNode(forwardMod, "ForwardModel");
  graph.AddNode(noiseModel->AsDensity(), "Likelihood");
  graph.AddNode(std::make_shared<DensityProduct>(2),"Posterior");
  graph.AddEdge("Parameters",0,"Prior",0);
  graph.AddEdge("Parameters",0,"ForwardModel",0);
  graph.AddEdge("ForwardModel",0,"Likelihood",0);
  graph.AddEdge("Prior",0,"Posterior",0);
  graph.AddEdge("Likelihood",0,"Posterior",1);

  auto logLikely = graph.CreateModPiece("Likelihood");

  pt::ptree pt;
  const unsigned int numSamps = 10000;
  pt.put("NumSamples",numSamps);
  pt.put("BurnIn", 0);
  pt.put("PrintLevel",0);
  pt.put("HessianType","Exact");
  pt.put("Adapt Interval", 1000);
  pt.put("Prior Node", "Prior");
  pt.put("Likelihood Node", "Likelihood");

  pt.put("Eigensolver Block", "EigOpts");
  pt.put("EigOpts.NumEigs",15); // Maximum number of generalized eigenvalues to compute (e.g., maximum LIS dimension)
  pt.put("EigOpts.ExpectedRank", dataDim);
  pt.put("EigOpts.OversamplingFactor", 20);

  pt.put("LIS Block", "LIS");
  pt.put("LIS.Method", "MHKernel");
  pt.put("LIS.Proposal","MyProposal");
  pt.put("LIS.MyProposal.Method","MHProposal");
  pt.put("LIS.MyProposal.ProposalVariance", 1.0);

  pt.put("CS Block", "CS");
  pt.put("CS.Method", "MHKernel");
  pt.put("CS.Proposal","MyProposal");
  pt.put("CS.MyProposal.Method", "CrankNicolsonProposal");
  pt.put("CS.MyProposal.Beta",1.0);
  pt.put("CS.MyProposal.PriorNode","Prior");

  // The posterior is Gaussian in this case, so we can analytically compute the posterior mean and covariance for comparison
  std::shared_ptr<Gaussian> truePost = prior->Condition(forwardMat, data, noiseStd*noiseStd*Eigen::VectorXd::Ones(dataDim));

  // create a sampling problem
  auto problem = std::make_shared<SamplingProblem>(graph.CreateModPiece("Posterior"));

  auto kernel = std::make_shared<DILIKernel>(pt, problem);

  auto sampler = std::make_shared<SingleChainMCMC>(pt, std::vector<std::shared_ptr<TransitionKernel>>{kernel});

  auto samps = sampler->Run(truePost->Sample()); // Use a true posterior sample to avoid burnin


  // Check the posterior moments
  Eigen::VectorXd sampMean = samps->Mean();
  Eigen::MatrixXd sampCov = samps->Covariance();
  Eigen::VectorXd trueMean = truePost->GetMean();
  Eigen::MatrixXd trueCov = truePost->GetCovariance();

  Eigen::VectorXd ess = samps->ESS();

  for(int i=0; i<numNodes; ++i){

    // Estimator variance
    double estVar = trueCov(i,i)/ess(i);
    EXPECT_NEAR(trueMean(i), sampMean(i), 4.0*std::sqrt(estVar));
  }
}


TEST(MCMC, DILIKernel_LogNormal) {

  //RandomGenerator::SetSeed(2012);

  const unsigned int numNodes = 100;
  const unsigned int dataDim = 5;
  const double noiseStd = 0.5;

  std::shared_ptr<Gaussian> prior = DILITest_CreatePrior(numNodes);

  Eigen::MatrixXd forwardMat = Eigen::MatrixXd::Zero(dataDim,numNodes);
  forwardMat.row(0) = Eigen::RowVectorXd::Ones(numNodes)/numNodes;
  for(unsigned int i=1; i<dataDim; ++i)
    forwardMat(i,10*i) = 1.0;

  auto forwardMod = LinearOperator::Create(forwardMat);
  Eigen::VectorXd trueField = prior->Sample();
  Eigen::VectorXd data = forwardMod->Apply(trueField.array().exp().matrix());

  auto noiseModel = std::make_shared<Gaussian>(data, noiseStd*noiseStd*Eigen::VectorXd::Ones(dataDim));

  WorkGraph graph;
  graph.AddNode(std::make_shared<IdentityOperator>(numNodes), "Parameters");
  graph.AddNode(prior->AsDensity(), "Prior");
  graph.AddNode(forwardMod, "ForwardModel");
  graph.AddNode(std::make_shared<ExpOperator>(numNodes), "ExpOperator");
  graph.AddNode(noiseModel->AsDensity(), "Likelihood");
  graph.AddNode(std::make_shared<DensityProduct>(2),"Posterior");
  graph.AddEdge("Parameters",0,"Prior",0);
  graph.AddEdge("Parameters",0,"ExpOperator",0);
  graph.AddEdge("ExpOperator",0,"ForwardModel",0);
  graph.AddEdge("ForwardModel",0,"Likelihood",0);
  graph.AddEdge("Prior",0,"Posterior",0);
  graph.AddEdge("Likelihood",0,"Posterior",1);

  pt::ptree pt;
  const unsigned int numSamps = 20000;
  pt.put("NumSamples",numSamps);
  pt.put("BurnIn", 1000);
  pt.put("PrintLevel",0);
  pt.put("HessianType","GaussNewton");
  pt.put("Adapt Interval", 1000);
  pt.put("Initial Weight",1);


  pt.put("Eigensolver Block", "EigOpts");
  pt.put("EigOpts.NumEigs",20); // Maximum number of generalized eigenvalues to compute (e.g., maximum LIS dimension)
  // pt.put("EigOpts.ExpectedRank", dataDim);
  pt.put("EigOpts.OversamplingFactor", 2);
  // pt.put("EigOpts.Verbosity",0);
  pt.put("Hess Tolerance", 1e-4);
  pt.put("LIS Tolerance", 0.1);

  pt.put("LIS Block", "LIS");
  pt.put("LIS.Method", "MHKernel");
  pt.put("LIS.Proposal","MyProposal");
  pt.put("LIS.MyProposal.Method","MHProposal");
  pt.put("LIS.MyProposal.ProposalVariance", 0.2);

  pt.put("CS Block", "CS");
  pt.put("CS.Method", "MHKernel");
  pt.put("CS.Proposal","MyProposal");
  pt.put("CS.MyProposal.Method", "CrankNicolsonProposal");
  pt.put("CS.MyProposal.Beta",0.5);
  pt.put("CS.MyProposal.PriorNode","Prior");

  // create a sampling problem
  auto problem = std::make_shared<SamplingProblem>(graph.CreateModPiece("Posterior"));

  auto kernel = std::make_shared<DILIKernel>(pt, problem);

  auto sampler = std::make_shared<SingleChainMCMC>(pt, std::vector<std::shared_ptr<TransitionKernel>>{kernel});

  auto diliSamps = sampler->Run(trueField); // Use a true posterior sample to avoid burnin
}
