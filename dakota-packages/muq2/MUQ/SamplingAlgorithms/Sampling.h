#ifndef MUQ_SAMPLING_H
#define MUQ_SAMPLING_H

/** @defgroup sampling Sampling

## Background and Motivation
Uncertainty quantification problems often require computing expectations of the form
\f\[
\bar{h} = \int_\Omega h(x) p(x) dx,
\f\]
where \f$h(x)\f$ is some utility function and \f$p(x)\f$ is the probability density function for the random variable \f$x\f$.  Monte Carlo approximations to \f$\bar{h}\f$ use random realizations \f$x^{(i)}\sim p(x)\f$ to approximate \f$\bar{h}\f$ with an estimator \f$\hat{h}\f$ defined by
\f\[
\hat{h} = \sum_{i=1}^N w_i h(x^{(i)}),
\f\]
where \f$w_i\f$ are appropriately defined weights.  Typically, \f$w_i = N^{-1}\f$.   MUQ's sampling module provides tools for constructing Monte Carlo estimates like \f$\hat{h}\f$.  In particular, MUQ provides a suite of [Markov chain Monte Carlo](\ref mcmc) (MCMC) algorithms for generating samples \f$x^{(i)}\f$ that can be used in Monte Carlo.


- The basics of **defining a sampling problem** in MUQ can be found [here](\ref mcmcprob).
- An introduction to **MCMC algorithms** is provided [here](\ref mcmc).
- An introduction to **dimension independent MCMC algorithms** can also be found [here](\ref disamp).
- **Multi-fidelity and Multi-index MCMC algorithms** are described [here](\ref MIMCMC).

*/

/**
@defgroup mcmc Markov chain Monte Carlo
@ingroup sampling

## Markov chain Monte Carlo (MCMC)
MCMC algorithms construct ergodic Markov chains \f$\{x^{(1)}, \ldots, x^{(N)}\}\f$ that can be used as samples in Monte Carlo approximations, like
\f\[
\int_\Omega h(x) p(x) dx \approx \frac{1}{N} h\left(x^{(i)}\right) = \hat{h}.
\f\]

The Markov chain is defined in terms of a transition kernel \f$K_n(x^{(n+1)} | x^{(n)})\f$, which is a probability distribution over the next state in the chain \f$x^{(n+1)}\f$ given the current state in the chain \f$x^{(n)}\f$.  There are many different ways of constructing the transition kernel; one of the most common approaches is to use the Metropolis-Hastings (MH) rule.   The MH rule constructs the kernel using another proposal distribution \f$q(x^\prime | x^{(n)})\f$ and evaluations of the target density \f$p(x)\f$.   In particular, the transition kernel is defined by the following process:

1. Draw a random sample \f$x^\prime \sim q(x^\prime | x^{(n)})\f$ from the proposal distribution.
2. Compute the acceptance probability
\f\[
\alpha = \min\left\{1,\quad \frac{p(x^\prime) q(x^{(n)} | x^\prime)}{p(x^{(n)}) q(x^\prime | x^{(n)})} \right\}
\f\]
3. Set \f$x^{(n+1)} = x^\prime\f$ with probability \f$\alpha\f$.  Else, \f$x^{(n+1)} = x^{(n)}\f$.

MUQ structures it's sampling module to mimic the components needed by the Metropolis-Hastings rule.   The [SamplingProblem](\ref muq::SamplingAlgorithms::SamplingProblem) interfaces with the target distribution \f$p(x)\f$ and (optionally) the quantity of interest \f$h(x)\f$.  Typically, \f$p(x)\f$ and \f$h(x)\f$ are defined as ModPieces.  (See the \ref Modeling chapter for more details on constructing ModPieces.)   Once the <code>SamplingProblem</code> is defined, an MCMC algorithm is constructed by combining an instance of the [TransitionKernel](\ref muq::SamplingAlgorithms::TransitionKernel) class with one or more instances of the [MCMCProposal](\ref muq::SamplingAlgorithms::MCMCProposal) class.

## Quickstart
Here's an example of constructing and running an MCMC sampler in MUQ.  This example
assumes that a ModPiece called <code>tgtDens</code> has been defined to compute \f$\log p(x)\f$ and
that a vector <code>x0</code> has been defined for the initial state \f$x^{(0)}\f$ of the chain.

@codeblock{cpp,C++}
// Construct the sampling problem from the target density
auto problem = std::make_shared<SamplingProblem>(tgtDens);

// Define proposal
boost::property_tree::ptree propOpts;
propOpts.put("ProposalVariance", 1.0);
auto prop = std::make_shared<MHProposal>(propOpts, problem);

// Use the proposal to construct a Metropolis-Hastings kernel
boost::property_tree::ptree kernOpts;
auto kern = std::make_shared<MHKernel>(kernOpts, problem, prop);

// Construct the MCMC sampler using this transition kernel
boost::property_tree::ptree chainOpts;
chainOpts.put("NumSamples", 2000);
chainOpts.put("BurnIn", 10);
chainOpts.put("PrintLevel", 3);
auto sampler = std::make_shared<SingleChainMCMC>(chainOpts, kern);

// Run the MCMC algorithm to generate samples. x0 is the starting location
auto samps = sampler.Run(x0);
@endcodeblock
@codeblock{python,Python}
# Construct the sampling problem from the target density
problem = ms.SamplingProblem(tgtDens)

# Define proposal
propOpts = {"ProposalVariance" : 1.0}
prop = ms.MHProposal(propOpts, problem)

# Use the proposal to construct a Metropolis-Hastings kernel
kernOpts = dict()
kern = ms.MHKernel(kernOpts, problem, prop)

# Construct the MCMC sampler using this transition kernel
chainOpts = {'NumSamples': 2000,
             'BurnIn': 10,
             'PrintLevel' : 3}

sampler = ms.SingleChainMCMC(chainOpts, [kern])

# Run the MCMC algorithm to generate samples. x0 is the starting location
samps = sampler.Run(x0)
@endcodeblock

Details on **defining MCMC algorithms** can be found [here](\ref mcmcalg).

Complete introductory examples can be found [in python](https://mituq.bitbucket.io/source/_site/example_pages_v0_3_1/webExamples/SamplingAlgorithms/MCMC/Example1_Gaussian/python/GaussianSampling.html) and [in c++](https://mituq.bitbucket.io/source/_site/example_pages_v0_3_1/webExamples/SamplingAlgorithms/MCMC/Example1_Gaussian/cpp/GaussianSampling.html).

*/


/** @defgroup mcmcalg Getting Started: Defining an MCMC Algorithm
@ingroup mcmc


## Overview
The goal of MCMC is to construct a Markov chain that is ergodic and has a given stationary distribution \f$p(x)\f$.  MUQ defines MCMC algorithms through three components: the chain, a transition kernel, and a proposal.  This idea is illustrated below for Metropolis-Hastings and Delayed-Rejection algorithms.

![mhimg]

[mhimg]: MHPuzzle.png "Components in a Metropolis-Hastings MCMC algorithm."

![drimg]

[drimg]: DRPuzzle.png "Components in a Delayed-Rejection MCMC algorithm."


In MUQ, the chain is the top level object in an MCMC definition.  In particular, the [SingleChainMCMC](\ref SingleChainMCMC) class is typically used to define a standard single-chain MCMC algorithm.   The chain, of course, depends on the transition kernel.  This corresponds to another abstract base class in the MUQ MCMC stack, the [TransitionKernel](\ref muq::SamplingAlgorithms::TransitionKernel) class.  The classic Metropolis-Hastings rule is implemented in the [MHKernel](\ref muq::SamplingAlgorithms::MHKernel) class, but this is just one of many different [transition kernels](\ref MCMCKernels) available in MUQ.  Each of these kernels can be coupled with one of the [proposal distributions](\ref MCMCProposals) implemented in MUQ.  The proposal distributions are children of the abstract [MCMCProposal](\ref muq::SamplingAlgorithms::MCMCProposal) base class.


## Step 1: Defining the Problem
The first step in using MUQ's MCMC tools is to define the posterior distribution we want to sample.  This is typically accomplished via the [SamplingProblem](\ref muq::SamplingAlgorithms::SamplingProblem) class.   The constructor of <code>SamplingProblem</code> takes a ModPiece that evaluates the log density \f$\log p(x)\f$.  It is also possible to pass another ModPiece that evaluates an additional quantity of interest that needs to be evaluated at samples of the target distribution.

Below is an example of defining a simple two dimensional Gaussian <code>SamplingProblem</code>.

The [Gaussian](\ref muq::Modeling::Gaussian) class is used to define the target density.  The Gaussian class can be used for either sampling or density evaluations.  The <code>AsDensity</code> function returns a ModPiece that evaluates the log of the Gaussian density -- exactly what the <code>SamplingProblem</code> needs.

@codeblock{cpp,C++}
Eigen::VectorXd mu(2);
mu << 1.0, 2.0;

Eigen::MatrixXd cov(2,2);
cov << 1.0, 0.8,
       0.8, 1.5;

auto targetDensity = std::make_shared<Gaussian>(mu, cov)->AsDensity();
auto problem = std::make_shared<SamplingProblem>(targetDensity);
@endcodeblock
@codeblock{python,Python}
mu = np.array([1.0,2.0])

cov = np.array([[1.0, 0.8],
                [0.8, 1.0]])

targetDensity = mm.Gaussian(mu,cov).AsDensity()
problem = ms.SamplingProblem(targetDensity)
@endcodeblock


## Step 2: Assembling an Algorithm
Once the problem is defined, it's time to combine chain, kernel, and proposal components to define an MCMC algorithm.  Below is an example combining the random walk [MHProposal](\ref muq::SamplingAlgorithms::MHProposal) proposal with the usual Metropolis-Hastings [MHKernel](\ref muq::SamplingAlgorithms::MHKernel) kernel in a single chain MCMC sampler.   Note that additional options are specified in with <code>boost::property_tree::ptree</code> variables in c++ and dictionaries in python.

@codeblock{cpp,C++}
// Define the random walk proposal proposal
boost::property_tree::ptree propOpts;
propOpts.put("ProposalVariance", 1.0);
auto prop = std::make_shared<MHProposal>(propOpts, problem);

// Define the Metropolis-Hastings transition kernel
boost::property_tree::ptree kernOpts;
auto kern = std::make_shared<MHKernel>(kernOpts, problem, prop);

// Construct the MCMC sampler using this transition kernel
boost::property_tree::ptree chainOpts;
chainOpts.put("NumSamples", 2000);
chainOpts.put("BurnIn", 10);
chainOpts.put("PrintLevel", 3);

auto sampler = std::make_shared<SingleChainMCMC>(chainOpts, kern);
@endcodeblock
@codeblock{python,Python}
# Define the random walk proposal
propOpts = {"ProposalVariance" : 1.0}
prop = ms.MHProposal(propOpts, problem)

# Define the Metropolis-Hastings transition kernel
kernOpts = dict()
kern = ms.MHKernel(kernOpts, problem, prop)

# Construct the MCMC sampler
chainOpts = {'NumSamples': 2000,
             'BurnIn': 10,
             'PrintLevel' : 3}

sampler = ms.SingleChainMCMC(chainOpts, [kern])
@endcodeblock

#### Swapping Proposals
Individual components of the MCMC sampler can be changed without changing anything else.  For example, the Metropolis-Adjusted Langevin Algorithm (MALA) uses the Metropolis-Hastings algorithm with a proposal that leverages gradient information to "shift" the random walk proposal towards higher density regions of \f$p(x)\f$.  Only the proposal distribution is different betweeen the MALA algorithm the basic standard Random Walk Metropolis (RWM) algorithm defined in the code above.  Defining a MALA sampler in MUQ therefore only requires changing the proposal to use the [MALAProposal](\ref muq::SamplingAlgorithms::MALAProposal) class.  This is shown below.
@codeblock{cpp,C++}
// Define the random walk proposal proposal
boost::property_tree::ptree propOpts;
propOpts.put("StepSize", 0.3);
auto prop = std::make_shared<MALAProposal>(propOpts, problem);
@endcodeblock
@codeblock{python,Python}
# Define the MALA proposal
propOpts = {"StepSize" : 0.3}
prop = ms.MALAProposal(propOpts, problem)
@endcodeblock

#### Changing Transition Kernels
Transition kernels are also interchangeable.  Here is an example of constructing a delayed rejection sampler \cite Mira2001, where the first stage is a random walk proposal and the second stage uses a Langevin proposal.
@codeblock{cpp,C++}
// Define a list of proposals to use with the DR kernel
std::vector<std::shared_ptr<MCMCProposal>> props(2);

// Create the random walk proposal proposal
boost::property_tree::ptree propOpts;
propOpts.put("ProposalVariance", 1.0);
props.at(0) = std::make_shared<MHProposal>(propOpts, problem);

// Create the Langevin proposal
boost::property_tree::ptree malaOpts;
malaOpts.put("StepSize", 0.3);
props.at(1) = std::make_shared<MALAProposal>(malaOpts, problem);

// Define the delayed rejection kernel
boost::property_tree::ptree kernOpts;
auto kern = std::make_shared<DRKernel>(kernOpts, problem, props);
@endcodeblock
@codeblock{python,Python}
# Create the random walk proposal
rwmOpts = {"ProposalVariance" : 1.0}
rwmProp = ms.MHProposal(rwmOpts, problem)

# Create the Langevin proposal
malaOpts = {"StepSize" : 0.3}
malaProp = ms.MALAProposal(malaOpts, problem)

# Create the delayed rejection kernel
kernOpts = dict()
kern = ms.DRKernel(kernOpts, problem, [rwmProp, malaProp])
@endcodeblock

## Step 3: Running the Sampler
We now have a sampling algorithm defined in the <code>sampler</code> variable in the code above, but no samples have been generated yet.   To run the sampler and generate samples, we need to call the <code>Run</code> method, which accepts a starting point and returns the MCMC samples in the form of a [SampleCollection](\ref muq::Modeling::SampleCollection).
@codeblock{cpp,C++}
// Define an initial state for the chain
Eigen::VectorXd x0(2);
x0 << 1.0, 2.0;

// Run the MCMC algorithm to generate samples. x0 is the starting location
std::shared_ptr<SampleCollection> samps = sampler->Run(x0);
@endcodeblock
@codeblock{python,Python}
# Define an initial state for the chain
x0 = np.array([1.0, 2.0])

# Run the MCMC algorithm to generate samples. x0 is the starting location
samps = sampler.Run([x0])
@endcodeblock


## Step 4: Inspecting the Results
The <code>Run</code> function returns a [SampleCollection](\ref muq::SamplingAlgorithms::SampleCollection) object.  SampleCollections store the state in the MCMC chain, weights for each state (in order to support importance sampling), as well as additional metadata that might have been stored for each sample by the MCMC components.

Here's an example of some basic things you might want to do with the sample collection.
@codeblock{cpp,C++}
// Return the number of samples in the collection
int numSamps = samps->size();

// Get various sample moments
Eigen::VectorXd mean = samps->Mean();
Eigen::VectorXd var = samps->Variance();
Eigen::MatrixXd cov = samps->Covariance();

// Create an Eigen matrix with all of the samples
Eigen::MatrixXd sampsAsMat = samps->AsMatrix();

// Extract metadata saved by the MCMC algorithm.
Eigen::MatrixXd logDensity = samps->GetMeta("LogTarget");
Eigen::MatrixXd gradLogDensity = samps->GetMeta("GradLogDensity_0"); // <- Only available when using gradient-based proposals (e.g., MALA)
@endcodeblock
@codeblock{python,Python}
# Return the number of samples in the collection
numSamps = samps.size()

# Get various sample moments
mean = samps.Mean()
var = samps.Variance()
cov = samps.Covariance()

# Create an Eigen matrix with all of the samples
sampsAsMat = samps.AsMatrix()

# Extract metadata saved by the MCMC algorithm.
logDensity = samps.GetMeta("LogTarget")
gradLogDensity = samps.GetMeta("GradLogDensity_0") # <- Only available when using gradient-based proposals (e.g., MALA)
@endcodeblock

Just like ModPieces can have multiple vector-valued inputs, the log density we're trying to sample might have multiple inputs, say \f$p(x,y)\f$.  In <code>GetMeta("GradLogDensity_0")</code>, the "_0" part of the string refers to index of the input that the gradient was taken with respect to, e.g., "_0" matches \f$\nabla_x \log p(x,y)\f$ and "_1" would match \f$\nabla_y \log p(x,y)\f$.  More information on sampling problems with more than one input can be found in the [Blocks and Kernel Composition](#blockmcmc) section.

### Assessing Convergence {#convergence}
The transition kernels used in MCMC methods guarantee that the target distribution \f$p(x)\f$ is the unique stationary distribution of the chain.  If the distribution of the initial state \f$x^{(0)}\f$ is \f$p(x)\f$, this stationarity property guarantees that all subsequent steps in the chain will also be distributed according to the target distribution \f$p(x)\f$.  In practice however, the reason we are using MCMC is because we can't generate samples of the target distribution and therefore can't guarantee \f$x^{(0)}\sim p(x)\f$.

For an arbitrary starting point \f$x^{(0)}\f$, we therefore need to assess whether the chain is long enough to ensure that the states have converged to the target distribution.   Often the first step in assessing convergence is to look a trace plot of the chain.   In python, we can convert extract a matrix of MCMC states from the <code>SampleCollection</code> class and then use matplotlib to create a trace plot (see below).

@codeblock{python,Python}
sampMat = samps.AsMatrix()

plt.plot(sampMat[0,:])
plt.xlabel('MCMC Step',fontsize=14)
plt.ylabel('Parameter $x_0$',fontsize=14)
plt.title('Converged Trace Plot',fontsize=16)
plt.show()
@endcodeblock

For a sufficient long chain, this code might result in a trace plot like:

![traceimg1]

[traceimg1]: ConvergedTracePlot.png "Trace plot for a sufficiently long MCMC chain."

For a short chain that has not "forgotten" it's starting point, you might see a trace plot that looks like:

![shorttraceplot]

[shorttraceplot]: ShortTracePlot.png "Trace plot for a short MCMC chain."


Trace plots are incredibly useful, but MUQ also has diagnostic tools for **quantitatively assessing convergence** using parallel chains and the\f$\hat{R}\f$ diagnostic described in \cite Gelman2013, \cite Vehtari2021, and many other publications.  The idea is to run multiple chains from different starting points and to assess whether the chains have reached the same distribution, presumably the target distribution \f$p(x)\f$.   Inter-chain means and variances are used to compute \f$\hat{R}\f$ for each dimension of the chain.  Values close to \f$1\f$ indicate convergence.    Here's an example running four chains and computing \f$\hat{R}\f$ with MUQ:

@codeblock{cpp,C++}
unsigned int dim = 2;
unsigned int numChains = 4;
std::vector<std::shared_ptr<SampleCollection>> collections(numChains);

for(int i=0; i<numChains; ++i){
  Eigen::VectorXd startPt = RandomGenerator::GetNormal(dim);
  auto sampler = std::make_shared<SingleChainMCMC>(chainOpts, kern);
  collections.at(i) = sampler->Run(startPt);
}

Eigen::VectorXd rhat = Diagnostics::Rhat(collections);
@endcodeblock
@codeblock{python,Python}
dim = 2
numChains = 4
collections = [None]*numChains

for i in range(numChains):
    startPt = np.random.randn(dim);
    sampler = ms.SingleChainMCMC(chainOpts, [kern])
    collections[i] = sampler.Run([startPt])

rhat = ms.Diagnostics.Rhat(collections)
@endcodeblock

The [Rhat](\ref muq::SamplingAlgorithms::Diagnostics::Rhat) function returns a vector containing the \f$\hat{R}\f$ statistic for each component of the chain.  Historically, values of \f$\hat{R}<1.1\f$ have been used to indicate convergence.  More recently however, it has been argued that this can sometimes lead to erroneous conclusions and a threshold of \f$\hat{R}<1.01\f$ may be more appropriate.  See \cite Vehtari2021 for more discussion on this point.

### Measuring Efficiency {#efficiency}
The \f$\hat{R}\f$ statistic helps quantify if the MCMC sampler is indeed producing samples of the target distribution \f$p(x)\f$.   However, it does not indicate how efficiently the sampler is exploring \f$p(x)\f$.  This is commonly accomplished by looking at the **effective sample size.**   MCMC produces a chain of *correlated* samples.  Monte Carlo estimators using these correlated samples will therefore have a larger variance than an estimator using the same number of independent samples.  The effective sample size (ESS) of a chain is the number of independent samples that would be needed to obtain the same estimator variance as the MCMC chain.

MUQ estimates the ESS of a single chain using the approach in \cite Wolff2004, which is based on the chain's autocorrelation function.   The <SampleCollection> class has a [ESS](\ref muq::Modeling::MarkovChain::ESS) function that returns a vector of estimate ESS values for each component of the chain.  Here's an example:
@codeblock{cpp,C++}
// Estimate the effective sample size
Eigen::VectorXd ess = samps->ESS();

// Estimate the Monte Carlo standard error (MCSE)
Eigen::VectorXd variance = samps->Variance();
Eigen::VectorXd mcse = (variance.array() / ess.array()).sqrt();
@endcodeblock
@codeblock{python,Python}
# Estimate the effective sample size
ess = samps.ESS()

# Estimate the Monte Carlo standard error (MCSE)
variance = samps.Variance()
mcse = np.sqrt(variance/ess)
@endcodeblock

## Using Options Lists
The MCMC samplers defined above were created by programmtically creating a proposal, then a kernel, and then an instance of the <code>SingleChainMCMC</code> class.   It is also possible to define MCMC algorithms entirely through string-valued options stored in either a <code>boost::property_tree::ptree</code> in c++ or a dictionary in python.   MUQ internally registers each MCMC proposal and kernel class with a string.  This allows scripts and executables to easily define difference algorithms entirely from text-based input files (e.g., JSON, YAML, XML).   Below is code to recreate a random walk sampler using only the list of options.
@codeblock{cpp,C++}
boost::property_tree::ptree options;

options.put("NumSamples", 10000); // number of MCMC steps
options.put("KernelList", "Kernel1"); // name of block that defines the transition kernel
options.put("Kernel1.Method", "MHKernel"); // name of the transition kernel class
options.put("Kernel1.Proposal", "MyProposal"); // name of block defining the proposal distribution
options.put("Kernel1.MyProposal.Method", "MHProposal"); // name of proposal class
options.put("Kernel1.MyProposal.ProposalVariance", 0.5); // variance of the isotropic MH proposal

auto mcmc = std::make_shared<SingleChainMCMC>(options,problem);
@endcodeblock
@codeblock{python,Python}
options = {
           "NumSamples": 10000, # number of MCMC steps
           "KernelList": "Kernel1",  # name of block that defines the transition kernel
           "Kernel1.Method": "MHKernel",  # name of the transition kernel class
           "Kernel1.Proposal": "MyProposal", # name of block defining the proposal distribution
           "Kernel1.MyProposal.Method": "MHProposal", # name of proposal class
           "Kernel1.MyProposal.ProposalVariance": 0.5 # variance of the isotropic MH proposal
         }

mcmc = ms.SingleChainMCMC(options,problem)
@endcodeblock
@codeblock{json,JSON}
{
  "NumSamples" : 10000,
  "KernelList" : "Kernel1",
  "Kernel1" :
  {
    "Method" : "MHKernel",
    "Proposal" : "MyProposal",
    "MyProposal" :
    {
      "Method" : "MHProposal",
      "ProposalVariance" : 0.5
    }
  }
}
@endcodeblock
@codeblock{cpp,C++ w/ JSON}
// Make Sure to include this at top of file
// #include <boost/property_tree/json_parser.hpp>

// Create an empty ptree
boost::property_tree::ptree options;

// Load the json contents into the ptree
boost::property_tree::read_json("mcmc-options.json", options);

auto mcmc = std::make_shared<SingleChainMCMC>(options,problem);
@endcodeblock
@codeblock{python,Python w/ JSON}
# Make sure to import json at start of script:
# import json

with open('mcmc-options.json','r') as fin:
  options = json.load(fin)

mcmc = ms.SingleChainMCMC(options,problem)
@endcodeblock

## Blockwise Sampling and Kernel Composition {#blockmcmc}
In the examples above, we considered sampling a density \f$p(x)\f$ that had one vector-valued input \f$x\f$.  However, SamplingProblems can have more than one input.  This situation commonly arises in hierarchical models, where we are interested in sampling a density \f$p(x, \alpha)\f$ with additional vector-valued hyperparameters \f$\alpha\f$ (e.g., the prior variance, or likelihood precision).   Often it is more convenient, and sometimes more efficient, to sample \f$x\f$ and \f$\alpha\f$ one at a time.  In the two-input case, the idea is to alternate through MCMC steps that first target the conditional \f$p(x | \alpha^{(k)})\f$ for a fixed value \f$\alpha^{(k)}\f$ and then target the conditional \f$p(\alpha | x^{(k)})\f$.  When variants of the Metropolis-Hastings rule are used for the MCMC transition kernel, this blockwise approach is known as Metropolis-in-Gibbs sampling.

MUQ defines Metrpolis-in-Gibbs samplers by passing in multiple transition kernels to the <code>SingleChainMCMC</code> class.  For illustration, consider a simple target density \f$p(x,y)=p(x)p(y)\f$, where \f$x\f$ takes values in \f$\mathbb{R}\f$, \f$y\f$ takes values in \f$\mathbb{R}^2\f$, \f$p(x)=N(\mu_x,\sigma_x)\f$, and \f$p(y)=N(\mu_y,\sigma_y)\f$.   In MUQ, we could define this density using a [WorkGraph](\ref modgraphs) as shown below
@codeblock{cpp,C++}
Eigen::VectorXd mux(1), varx(1);
mux << 1.0;
varx << 0.5;

auto px = std::make_shared<Gaussian>(mux,varx)->AsDensity();

Eigen::VectorXd muy(2), vary(2);
muy << -1.0, -1.0;
vary << 1.0, 2.0;

auto py = std::make_shared<Gaussian>(muy,vary)->AsDensity();

auto graph = std::make_shared<WorkGraph>();
graph->AddNode(px, "p(x)");
graph->AddNode(py, "p(y)");
graph->AddNode(std::make_shared<DensityProduct>(2), "p(x,y)");
graph->AddEdge("p(x)",0,"p(x,y)",0);
graph->AddEdge("p(y)",0,"p(x,y)",1);

auto pxy = graph->CreateModPiece("p(x,y)");
@endcodeblock
@codeblock{python,Python}
mux = np.array([1.0])
varx = np.array([0.5])

# Create the joint log density, which will have two inputs with sizes [1,2]
px = mm.Gaussian(mux,varx).AsDensity()

muy = np.array([-1.0,-1.0])
vary = np.array([1.0,2.0])

py = mm.Gaussian(muy,vary).AsDensity()

graph = mm.WorkGraph>()
graph.AddNode(px, "p(x)")
graph.AddNode(py, "p(y)")
graph.AddNode(mm.DensityProduct(2), "p(x,y)")
graph.AddEdge("p(x)",0,"p(x,y)",0)
graph.AddEdge("p(y)",0,"p(x,y)",1)

# Create the joint log density, which will have two inputs with sizes [1,2]
pxy = graph.CreateModPiece("p(x,y)")
@endcodeblock

In this example, the <code>pxy</code> variable is a two-input ModPiece that evaluates the log density \f$\log p(x,y)\f$.  The index of each input to this ModPiece can be specified through the <code>BlockIndex</code> option in the MCMCProposal class, which tells the proposal which input of the target density to target.   The snippet below demonstrates this for two random walk proposals.
@codeblock{cpp,C++}
// Define the sampling problem as usual
auto problem = std::make_shared<SamplingProblem>(pxy);

// Construct two kernels: one for x and one for y
boost::property_tree::ptree opts;

// A vector to holding the two transition kernels
std::vector<std::shared_ptr<TransitionKernel>> kernels(2);

// Construct the kernel on x
opts.put("ProposalVariance", 3.0);
opts.put("BlockIndex", 0); // Specify that this proposal should target x
auto propx = std::make_shared<MHProposal>(opts, problem);
kernels.at(0) = std::make_shared<MHKernel>(opts, problem, propx);

// Construct the kernel on y
opts.put("ProposalVariance", 5.0);
opts.put("BlockIndex", 1); // Specify that this proposal should target y
auto propy = std::make_shared<MHProposal>(opts, problem);
kernels.at(1) = std::make_shared<MHKernel>(opts, problem, propy);

// Construct the MCMC sampler using this transition kernel
opts.put("NumSamples", 2000);
opts.put("BurnIn", 10);
opts.put("PrintLevel", 3);

auto sampler = std::make_shared<SingleChainMCMC>(chainOpts, kernels);
@endcodeblock
@codeblock{python,Python}
# Define the sampling problem as usual
problem = ms.SamplingProblem(pxy)

# Construct the proposal on x
opts = {
        "ProposalVariance": 3.0,
        "BlockIndex" : 0 # Specify that this proposal should target x
       }

propx = ms.MHProposal(opts, problem)

# Construct the proposal on y
opts = {
        "ProposalVariance": 5.0,
        "BlockIndex": 1
       }

propy = ms.MHProposal(opts, problem)

# Construct the MCMC sampler using this transition kernel
kernels = [ms.MHKernel({}, problem, propx),
           ms.MHKernel({}, problem, propy)]

sampler = SingleChainMCMC(opts, kernels)
@endcodeblock

See the [pump test example](https://mituq.bitbucket.io/source/_site/example_pages_v0_3_1/webExamples/OtherApplications/PumpTest_Theis/TheisInference.html) for a more comprehensive example showing the use of Metropolis-in-Gibbs sampling for hyperparameters in a Gaussian likelihood function.

*/


/**
@defgroup disamp Dimension-Independent MCMC
@ingroup mcmc

## Dimension-Independent MCMC
Documentation coming soon...
*/

/** @defgroup mcmcdiag Markov Chain Diagnostics
@ingroup mcmc
*/

/**
@defgroup MIMCMC Multi-Index MCMC
@ingroup mcmc
@brief Tools for defininig and running multilevel and multiindex MCMC algorithms.
@details Multiindex MCMC methods are built on an entire arbitriry-dimensional
grid of sampling problems, in contrast to classical MCMC only sampling from
a single distribution.

In order to be effective, MIMCMC methods require distributions closely related to each
other, where evaluating the coarser ones should be significantly computationally cheaper.
A typical example are models based on the Finite Element method, where strongly varying mesh
resolutions lead to a hierarchy of models. Then, coarser models are a
good approximation of finer ones while being far cheaper to compute.

In analogy to the mathematical definition, running a MIMCMC method requires defining
a grid of models (as well as other needed components) via MIComponentFactory.
Most importantly, how proposals will be drawn from coarser chains has to be defined
as well as how to combine them (via MIInterpolation) with a proposal for the current
fine chain.

Refer to the MCMC examples for complete code examples.
*/

/**
@defgroup MCMCProposals MCMC Proposal Distributions
@ingroup mcmc
*/

#endif // #ifndef MUQ_SAMPLING_H
