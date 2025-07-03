#include "MUQ/Modeling/Distributions/MixtureDistribution.h"

#include "MUQ/Utilities/RandomGenerator.h"

using namespace muq::Utilities;
using namespace muq::Modeling;

MixtureDistribution::MixtureDistribution(std::vector<std::shared_ptr<Distribution>> const& componentsIn,
                                         Eigen::VectorXd                             const& probsIn) : Distribution(componentsIn.at(0)->varSize, componentsIn.at(0)->hyperSizes),
                                                                                                     components(componentsIn), probs(probsIn)
{

    probs = probs / probs.sum();

    // Make sure the components and probabilities have the same size
    if(components.size() != probs.size())
    {
        std::stringstream msg;
        msg << "Could not construct MixtureDistribution.  Number of components is " << components.size() << " but " << probs.size() << " probabilities were given.";
        throw std::runtime_error(msg.str());
    }

    // Make sure all the components have the same number and size of inputs 
    for(unsigned int i=1; i<components.size(); ++i)
    {
        if(components.at(i)->varSize != varSize){
            std::stringstream msg;
            msg << "Could not construct MixtureDistribution.  Component " << i << " has a dimension of " << components.at(i)->varSize << " but component 0 has a dimension of " << varSize;
            throw std::runtime_error(msg.str());
        }

        if(components.at(i)->hyperSizes.size() != hyperSizes.size()){
            std::stringstream msg;
            msg << "Could not construct MixtureDistribution.  Component " << i << " has " << components.at(i)->hyperSizes.size() << " hyperparameters but component 0 has " << hyperSizes.size() << " hyperparameters.";
            throw std::runtime_error(msg.str());
        }

        for(unsigned int j=0; j<hyperSizes.size(); ++j){
            if(components.at(i)->hyperSizes(j) != hyperSizes(j)){
                std::stringstream msg;
                msg << "Could not construct MixtureDistribution.  Hyperparameter " << j << " of component " << i << " has dimension " << components.at(i)->hyperSizes(j) << " but this hyperparameter in component 0 has dimension " << hyperSizes(j);
                throw std::runtime_error(msg.str());
            }
        }
    }
    
}


double MixtureDistribution::LogDensityImpl(ref_vector<Eigen::VectorXd> const& inputs)
{
    double pdf = 0.0;
    for(unsigned int i=0; i<components.size(); ++i)
        pdf += probs(i)*std::exp(components.at(i)->LogDensity(inputs));
    
    return std::log(pdf);
}

Eigen::VectorXd MixtureDistribution::GradLogDensityImpl(unsigned int wrt,
                                                        ref_vector<Eigen::VectorXd> const& inputs)
{   
    unsigned int gradDim = (wrt==0) ? varSize : hyperSizes(wrt-1);

    double pdf = 0.0;
    Eigen::VectorXd grad_pdf = Eigen::VectorXd::Zero(gradDim);

    for(unsigned int i=0; i<components.size(); ++i){
        double comp_pdf = std::exp(components.at(i)->LogDensity(inputs));
        pdf += probs(i)*comp_pdf;
        grad_pdf += probs(i)*comp_pdf * components.at(i)->GradLogDensity(wrt,inputs);
    }

    return (1.0/pdf) * grad_pdf; 
}


Eigen::VectorXd MixtureDistribution::SampleImpl(ref_vector<Eigen::VectorXd> const& inputs)
{
  // Pick a component at random
  int randInd = RandomGenerator::GetDiscrete(probs);

  // Return a sample of the random index
  return components.at(randInd)->Sample(inputs);
}
