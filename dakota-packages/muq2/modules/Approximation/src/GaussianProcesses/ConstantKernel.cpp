#include "MUQ/Approximation/GaussianProcesses/ConstantKernel.h"
#include "MUQ/Modeling/LinearAlgebra/IdentityOperator.h"
#include "MUQ/Modeling/LinearAlgebra/ZeroOperator.h"
#include "MUQ/Modeling/LinearSDE.h"

using namespace muq::Approximation;

ConstantKernel::ConstantKernel(unsigned              dim,
	                          const double          sigma2In,
                              const Eigen::Vector2d sigmaBounds) : ConstantKernel(dim, sigma2In*Eigen::MatrixXd::Ones(1,1), sigmaBounds){};

ConstantKernel::ConstantKernel(unsigned              dim,
		                       std::vector<unsigned> dimInds,
	                           const double          sigma2In,
                               const Eigen::Vector2d sigmaBounds) : ConstantKernel(dim, dimInds, sigma2In*Eigen::MatrixXd::Ones(1,1), sigmaBounds){};


ConstantKernel::ConstantKernel(unsigned               dim,
	                           Eigen::MatrixXd const& sigma2In,
                               const Eigen::Vector2d  sigmaBounds) : KernelImpl<ConstantKernel>(dim, sigma2In.rows(), GetNumParams(sigma2In))
{
    paramBounds.resize(2,1);
    paramBounds(0,0) = sigmaBounds(0);
    paramBounds(1,0) = sigmaBounds(1);

    cachedParams.resize(numParams);
    int ind = 0;
    for(int i=0; i<sigma2In.rows(); ++i){
        for(int j=0; j<=i; ++j){
            cachedParams(ind) = sigma2In(i,j);
            ind++;
        }
    }
};

ConstantKernel::ConstantKernel(unsigned               dim,
		                       std::vector<unsigned>  dimInds,
	                           Eigen::MatrixXd const& sigma2In,
                               const Eigen::Vector2d  sigmaBounds) : KernelImpl<ConstantKernel>(dim, dimInds, sigma2In.rows(), GetNumParams(sigma2In))
{
    paramBounds.resize(2,1);
    paramBounds(0,0) = sigmaBounds(0);
    paramBounds(1,0) = sigmaBounds(1);

    cachedParams.resize(numParams);
    int ind = 0;
    for(int i=0; i<sigma2In.rows(); ++i){
        for(int j=0; j<=i; ++j){
            cachedParams(ind) = sigma2In(i,j);
            ind++;
        }
    }
};


std::tuple<std::shared_ptr<muq::Modeling::LinearSDE>, std::shared_ptr<muq::Modeling::LinearOperator>, Eigen::MatrixXd> ConstantKernel::GetStateSpace(boost::property_tree::ptree sdeOptions) const
{
    std::shared_ptr<muq::Modeling::LinearOperator> id =  std::make_shared<muq::Modeling::IdentityOperator>(coDim);

    Eigen::MatrixXd marginalVar(coDim, coDim);
    Eigen::VectorXd x(inputDim);
    FillBlockImpl<double,double,double>(x,x, cachedParams, marginalVar);

    boost::property_tree::ptree options;
    options.put("SDE.dt", 1e6); // large step size to ensure that we only ever take one step

    std::shared_ptr<muq::Modeling::LinearOperator> zo = std::make_shared<muq::Modeling::ZeroOperator>(coDim,coDim);
    auto sde = std::make_shared<muq::Modeling::LinearSDE>(zo, options);
    
    return std::make_tuple(sde, id, marginalVar);
}
    