#ifndef MUQ_FORWARDUQ_H
#define MUQ_FORWARDUQ_H

/** @defgroup forwarduq Forward Uncertainty Quantification

## Overview

*/

/** @defgroup polychaos Polynomial Chaos Expansions
    @ingroup forwarduq
## Quickstart
The [AdaptiveSmolyakPCE](\ref muq::Approximation::AdaptiveSmolyakPCE) class uses the algorithm described in \cite Conrad2013 to adaptively construct polynomial chaos expansions.  The [PolynomialChaosExpansion](\ref muq::Approximation::PolynomialChaosExpansion) class defines polynomial chaos expansions in MUQ.
@codeblock{cpp,C++}
// Define 1d quadrature rules for each dimension
auto quad1d = std::make_shared<GaussPattersonQuadrature>();
std::vector<std::shared_ptr<Quadrature>> quads(dim, quad1d);

// Define 1d orthogonal polynomial families for each dimension
auto polys1d = std::make_shared<Legendre>();
std::vector<std::shared_ptr<IndexedScalarBasis>> polys(dim, polys1d);

// Create the PCE solver
AdaptiveSmolyakPCE smolyPCE(model, quads, polys);

// Set solver options
boost::property_tree::ptree options;
options.put("ShouldAdapt", 1);     // After constructing an initial approximation with the terms in "multis", should we continue to adapt?
options.put("ErrorTol", 5e-4);    // Stop when the estimated L2 error is below this value
options.put("MaximumEvals", 200); // Stop adapting when more than this many model evaluations has occured

// Specify which terms we should start with
unsigned int initialOrder = 1;
auto multis = MultiIndexFactory::CreateTotalOrder(dim,initialOrder);

// Compute the polynomial chaos expansion
auto pce = smolyPCE.Compute(multis, options);
@endcodeblock
@codeblock{python,Python}
# Define a 1d quadrature rule
quad1d = ma.GaussPattersonQuadrature()

# Define a family of 1d orthogonal polynomials
polys1d = ma.Legendre()

# Create the PCE solver
smolyPCE = ma.AdaptiveSmolyakPCE(model, [quad1d]*dim, [polys1d]*dim)

# Set solver options
options = {
            'ShouldAdapt' : 1,   # After constructing an initial approximation with the terms in "multis", should we continue to adapt?
            'ErrorTol' : 5e-4,   # Stop when the estimated L2 error is below this value
            'MaximumEvals' : 200 # Stop adapting when more than this many model evaluations has occured
          }

# Start with a linear approximation
initialOrder = 1
multis = mu.MultiIndexFactory.CreateTotalOrder(dim,initialOrder)

# Compute the polynomial chaos expansion
pce = smolyPCE.Compute(multis, options)
@endcodeblock

The <code>pce</code> variable is an instance of the [PolynomialChaosExpansion](\ref muq::Approximation::PolynomialChaosExpansion) class, which is itself a child of the [ModPiece](\ref modpieces) class.  Many different functions are implemented for evaluating the PCE approximation and extracting information about the predictive distribution.   For example,
@codeblock{cpp,C++}
// Prediction mean
Eigen::VectorXd predMean = pce->Mean();

// Prediction variance
Eigen::VectorXd predVar = pce->Variance();

// Full prediction covariance
Eigen::MatrixXd predCov = pce->Covariance();

// Total sensitivity matrix of each output wrt each input
Eigen::MatrixXd totalSens = pce->TotalSensitivity();

// Main effects matrix of each output wrt each individual input
Eigen::MatrixXd mainEffects = pce->MainSensitivity();
@endcodeblock
@codeblock{python,Python}
# Prediction mean
predMean = pce.Mean()

# Prediction variance
predVar = pce.Variance()

# Full prediction covariance
predCov = pce.Covariance()

# Total sensitivity matrix of each output wrt each input
totalSens = pce.TotalSensitivity()

# Main effects matrix of each output wrt each individual input
mainEffects = pce.MainSensitivity()
@endcodeblock

## Additional Resources
See the [PCE examples](https://mituq.bitbucket.io/source/_site/examples.html) for more details on how to construct polynomial chaos expansions in MUQ.
*/

#endif
