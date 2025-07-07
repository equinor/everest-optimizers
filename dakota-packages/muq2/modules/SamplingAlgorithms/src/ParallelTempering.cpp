#include "MUQ/SamplingAlgorithms/ParallelTempering.h"

#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <ios>

#include "MUQ/Utilities/AnyHelpers.h"
#include "MUQ/Utilities/StringUtilities.h"

#include "MUQ/Utilities/RandomGenerator.h"
#include "MUQ/SamplingAlgorithms/MarkovChain.h"

namespace pt = boost::property_tree;
using namespace muq::Utilities;
using namespace muq::SamplingAlgorithms;


ParallelTempering::ParallelTempering(boost::property_tree::ptree              opts,
                                     std::shared_ptr<InferenceProblem> const& problem) : ParallelTempering(opts, ExtractTemps(opts), ExtractKernels(opts, problem))
{}

ParallelTempering::ParallelTempering(boost::property_tree::ptree                    opts,
                                     Eigen::VectorXd                                inverseTemps,
                                     std::vector<std::shared_ptr<TransitionKernel>> kernels) : ParallelTempering(opts, inverseTemps, StackObjects(kernels))
{}



ParallelTempering::ParallelTempering(boost::property_tree::ptree                                        opts,
                                     Eigen::VectorXd                                             inverseTemps,
                                     std::vector<std::vector<std::shared_ptr<TransitionKernel>>> kernelsIn) : numTemps(inverseTemps.size()), 
                                                                                                                     scheduler(std::make_shared<ThinScheduler>(opts)),
                                                                                                                     kernels(kernelsIn),
                                                                                                                     numSamps(opts.get<double>("NumSamples")),
                                                                                                                     burnIn(opts.get("BurnIn",0)),
                                                                                                                     printLevel(opts.get("PrintLevel",3)),
                                                                                                                     swapIncr(opts.get("Swap Increment", 2)),
                                                                                                                     seoSwaps(opts.get("Swap Type","DEO")=="SEO"),
                                                                                                                     cumulativeSwapProb(Eigen::VectorXd::Zero(numTemps-1)),
                                                                                                                     successfulSwaps(Eigen::VectorXd::Zero(numTemps-1)),
                                                                                                                     attemptedSwaps(Eigen::VectorXd::Zero(numTemps-1)),
                                                                                                                     nextAdaptInd(opts.get("Adapt Start", 100))
{   
    if(std::abs(inverseTemps(0))>std::numeric_limits<double>::epsilon()){
        std::stringstream msg;
        msg << "In ParallelTempering constructor.  First inverse temperature in schedule must be 0.0.";
        throw std::invalid_argument(msg.str());
    }

    if(std::abs(inverseTemps(inverseTemps.size()-1)-1.0)>std::numeric_limits<double>::epsilon()){
        std::stringstream msg;
        msg << "In ParallelTempering constructor.  Last inverse temperature in schedule must be 1.0.";
        throw std::invalid_argument(msg.str());
    }


    if(inverseTemps.minCoeff()<-std::numeric_limits<double>::epsilon()){
        std::stringstream msg;
        msg << "In ParallelTempering constructor.  Inverse temperatures must be in [0,1], but found a minimum temperature of " << inverseTemps.minCoeff();
        throw std::invalid_argument(msg.str());
    }

    if(inverseTemps.maxCoeff()>1+std::numeric_limits<double>::epsilon()){
        std::stringstream msg;
        msg << "In ParallelTempering constructor.  Inverse temperatures must be in [0,1], but found a maximum temperature of " << inverseTemps.maxCoeff();
        throw std::invalid_argument(msg.str());
    }

    problems.resize(kernels.size());
    for(unsigned int i=0; i<kernels.size(); ++i){
        problems.at(i) = std::dynamic_pointer_cast<InferenceProblem>( kernels.at(i).at(0)->Problem() );
        
        if(problems.at(i)==nullptr){
            std::stringstream msg;
            msg << "In ParallelTempering constructor.  Could not cast sampling problem for dimension " << i << " into an InfereceProblem.";
            throw std::invalid_argument(msg.str());
        }
    }

    // Check to make sure the problems are not pointing at the same thing.  Otherwise we won't be able to set the temperatures
    for(unsigned int i=0; i<kernels.size()-1; ++i){
        for(unsigned j=i+1; j<kernels.size(); ++j){
            if(problems.at(i)==problems.at(j)){
                std::stringstream msg;
                msg << "In ParallelTempering constructor.  Found pointers to the same sampling problem, which prevents setting the temperature at different levels.";
                throw std::invalid_argument(msg.str());
            }
        }
    }
    
    // Set the temperatures 
    chains.resize(kernels.size());
    sampNums.resize(kernels.size(), 0);
    for(unsigned int i=0; i<kernels.size(); ++i){
        problems.at(i)->SetInverseTemp(inverseTemps(i));
        chains.at(i) = std::make_shared<MarkovChain>();
    }
}


void ParallelTempering::SetState(std::vector<std::shared_ptr<SamplingState>> const& x0)
{   
    if(x0.size() != numTemps){
        std::stringstream msg;
        msg << " In ParallelTempering::SetState, the size of the argument x0 is " << x0.size() << ", but the temperature schedule has " << numTemps << " levels.";
        throw std::invalid_argument(msg.str());
    }

    prevStates = x0;
}


void ParallelTempering::SetState(std::vector<Eigen::VectorXd> const& x0)
{   
    std::vector<std::shared_ptr<SamplingState>> states(numTemps);
    for(unsigned int i=0; i<numTemps; ++i)
        states.at(i) = std::make_shared<SamplingState>(x0);

    SetState(states);
}
      
void ParallelTempering::SetState(std::vector<std::vector<Eigen::VectorXd>> const& x0)
{   
    if(x0.size() != numTemps){
        std::stringstream msg;
        msg << " In ParallelTempering::SetState, the size of the argument x0 is " << x0.size() << ", but the temperature schedule has " << numTemps << " levels.";
        throw std::invalid_argument(msg.str());
    }

    std::vector<std::shared_ptr<SamplingState>> states(numTemps);
    for(unsigned int i=0; i<numTemps; ++i)
        states.at(i) = std::make_shared<SamplingState>(x0.at(i));

    SetState(states);
}


double ParallelTempering::GetInverseTemp(unsigned int chainInd) const
{
    return problems.at(chainInd)->GetInverseTemp();
}

std::vector<std::shared_ptr<TransitionKernel>> const& ParallelTempering::Kernels(unsigned int chainInd) const
{
    return kernels.at(chainInd);
}


std::shared_ptr<MarkovChain> ParallelTempering::Run(){return Run(std::vector<std::vector<Eigen::VectorXd>>());}

std::shared_ptr<MarkovChain> ParallelTempering::Run(std::vector<Eigen::VectorXd> const& x0){
    return Run(StackObjects(x0));
}

std::shared_ptr<MarkovChain> ParallelTempering::Run(std::vector<std::vector<Eigen::VectorXd>> const& x0) {
 
  if( !x0.empty() ) { SetState(x0); }

  // What is the next iteration that we want to print at
  const unsigned int printIncr = std::floor(numSamps / double(10));
  unsigned int nextPrintInd = printIncr;
  unsigned int nextSwapInd = swapIncr;

  // Run until we've received the desired number of samples
  if(printLevel>0)
    std::cout << "Starting parallel tempering sampler..." << std::endl;

  while(sampNums.at(numTemps-1) < numSamps)
  { 
    // Should we print
    if(sampNums.at(numTemps-1) > nextPrintInd){
      if(printLevel>0){
        PrintStatus("  ", sampNums.at(numTemps-1));
      }
      nextPrintInd += printIncr;
    }

    // Swap every swapIncr steps
    if(sampNums.at(numTemps-1)>nextSwapInd){
        SwapStates();
        nextSwapInd += swapIncr;
    }

    // Exploratory steps (independent MCMC kernels for each MCMC chain)
    Sample();
    
    // Adapt the temperatures 
    if(sampNums.at(numTemps-1) > nextAdaptInd){
        AdaptTemperatures();
        adaptIncr *= 2;
        nextAdaptInd += adaptIncr;
    }
  }


  if(printLevel>0){
    PrintStatus("  ", numSamps+1);
    std::cout << "Completed in " << totalTime << " seconds." << std::endl;
  }

  return chains.at(numTemps-1);
}


void ParallelTempering::AdaptTemperatures(){

    // We need at least one swap before we can adapt
    if(attemptedSwaps.minCoeff()==0)
        return;

    // Current inverse temperatures
    Eigen::VectorXd currBetas = CollectInverseTemps();//(numTemps);
    for(unsigned int i=0; i<numTemps; ++i)
        currBetas(i) = problems.at(i)->GetInverseTemp();

    // First, compute a cumulative sum of the average acceptance probabilities 
    Eigen::VectorXd cumProbs(numTemps);
    cumProbs(0) = 0.0;//cumulativeSwapProb(0) / attemptedSwaps(0);
    for(unsigned int i=1; i<numTemps; ++i)
        cumProbs(i) = cumProbs(i-1) + 1.0- (cumulativeSwapProb(i-1) / attemptedSwaps(i-1));

    if((cumProbs.maxCoeff()-cumProbs.minCoeff())<1e-8)
        return;

    for(unsigned int i=1; i<numTemps-1; ++i){
        double desiredVal = cumProbs(numTemps-1) * double(i)/(numTemps-1);
        
        // Use linear interpolation to get the temperature that would achieve this value
        int j;
        for(j=0; j<numTemps; ++j){
            if(cumProbs(j) >= desiredVal)
                break;
        }
        if((j==0)||(j==numTemps)){
            std::cout << "Cumulative probs: " << cumProbs.transpose() << std::endl;
            std::cout << "desiredVal: " << desiredVal << std::endl;
        }
        assert(j!=numTemps);
        assert(j>0);

        double w = (desiredVal - cumProbs(j-1)) / (cumProbs(j) - cumProbs(j-1));
        double newTemp = w*currBetas(j) + (1.0-w)*currBetas(j-1);

        // Update the temperature of the problem
        problems.at(i)->SetInverseTemp(newTemp);
    }   

    // Reset the recorded swap probabilities 
    successfulSwaps.setZero();
    cumulativeSwapProb.setZero();
    attemptedSwaps.setZero();
}

Eigen::VectorXd ParallelTempering::CollectInverseTemps() const
{
    Eigen::VectorXd output(numTemps);
    for(unsigned int i=0; i<numTemps; ++i)
        output(i) = problems.at(i)->GetInverseTemp();
    return output;
}

void ParallelTempering::PrintStatus(std::string prefix, unsigned int currInd) const
{
  std::cout << prefix << int(std::floor(double((currInd - 1) * 100) / double(numSamps))) << "% Complete" << std::endl;
  
  if(printLevel>1){
      std::streamsize ss = std::cout.precision();
      std::cout.precision(2);
      std::cout << prefix << "  Avg. Swap Probs: " << (cumulativeSwapProb.array() / attemptedSwaps.array()).matrix().transpose() << std::endl;
      std::cout << prefix << "  Inverse Temps:   " << CollectInverseTemps().transpose() << std::endl;
      std::cout.precision(ss);
  }

  if(printLevel==2){
    std::cout << prefix << "  Kernel 0:\n";
    for(int blockInd=0; blockInd<kernels.at(0).size(); ++blockInd){
      std::cout << prefix << "    Block " << blockInd << ":\n";
      kernels.at(0).at(blockInd)->PrintStatus(prefix + "    ");
    }
  }else if(printLevel>2){
      for(int chainInd=0; chainInd<numTemps; ++chainInd){
          std::cout << prefix << "  Kernel " << chainInd << ":\n";
          for(int blockInd=0; blockInd<kernels.at(chainInd).size(); ++blockInd){
            std::cout << prefix << "    Block " << blockInd << ":\n";
            kernels.at(chainInd).at(blockInd)->PrintStatus(prefix + "      ");
          }
      }
  }
}

void ParallelTempering::CheckForMeta(std::shared_ptr<SamplingState> const& state)
{
    if(!state->HasMeta("InverseTemp")){
        std::stringstream msg;
        msg << "Error in ParallelTempering::SwapStates. Tried swapping states with a state that does not have temperature metadata.  The state must have the \"InverseTemp\" metadata, which is typically set in InferenceProblem::LogDensity.";
        throw std::runtime_error(msg.str());
    }

    if(!state->HasMeta("LogLikelihood")){
        std::stringstream msg;
        msg << "Error in ParallelTempering::SwapStates. Tried swapping states with a state that does not have likelihood metadata.  The state must have the \"LogLikelihood\" metadata, which is typically set in InferenceProblem::LogDensity.";
        throw std::runtime_error(msg.str());
    }

    if(!state->HasMeta("LogPrior")){
        std::stringstream msg;
        msg << "Error in ParallelTempering::SwapStates. Tried swapping states with a state that does not have prior metadata.  The state must have the \"LogPrior\" metadata, which is typically set in InferenceProblem::LogDensity.";
        throw std::runtime_error(msg.str());
    }
}

void ParallelTempering::SwapStates() {
  
    // Figure out if this is an even swap or an odd swap
    unsigned int startInd;
    if(seoSwaps){
        startInd = RandomGenerator::GetUniformInt(0,1);
    }else{
        startInd = evenSwap ? 0 : 1;
    }

    double beta1, beta2, logLikely1, logLikely2, alpha;

    for(unsigned int i=startInd; i<numTemps-1; i+=2){
        CheckForMeta(prevStates.at(i));
        CheckForMeta(prevStates.at(i+1));

        beta1 = AnyCast( prevStates.at(i)->meta["InverseTemp"] );
        beta2 = AnyCast( prevStates.at(i+1)->meta["InverseTemp"] );        

        logLikely1 = AnyCast( prevStates.at(i)->meta["LogLikelihood"] );
        logLikely2 = AnyCast( prevStates.at(i+1)->meta["LogLikelihood"] );
        
        alpha = std::exp( (beta1 - beta2)*(logLikely2 - logLikely1));
        
        attemptedSwaps(i)++;
        cumulativeSwapProb(i) += std::min(alpha, 1.0);

        if(RandomGenerator::GetUniform() < alpha){ 
            std::swap(prevStates[i], prevStates[i+1]);
            prevStates.at(i)->meta["InverseTemp"] = problems.at(i)->GetInverseTemp();
            prevStates.at(i+1)->meta["InverseTemp"] = problems.at(i+1)->GetInverseTemp();
            successfulSwaps(i)++;
        }
    }


    // alternate between true and false.  If evenSwap is true, this will set it false.  If evenSwap is false, this will set it to true.
    evenSwap = evenSwap ^ true;

}

void ParallelTempering::Sample() {

  assert(kernels.size()==numTemps);
  
  bool initError = false;
  if(prevStates.size()==0){
      initError = true;
  }else if(prevStates.at(0)==nullptr){
      initError = true;
  }
  
  if(initError){
    std::stringstream msg;
    msg << "\nERROR in ParallelTempering::Sample.  Trying to sample chain but previous (or initial) state has not been set.\n";
    throw std::runtime_error(msg.str());
  }

  auto startTime = std::chrono::high_resolution_clock::now();

  std::vector<std::vector<std::shared_ptr<SamplingState>>> newStates(numTemps);
  newStates.resize(numTemps);
  
  // Loop over all the different chains 
  for(unsigned int chainInd=0; chainInd<numTemps; ++chainInd){
    newStates.at(chainInd).resize(kernels.at(chainInd).size());
    
    // Loop through each parameter block
    for(int kernInd=0; kernInd<kernels.at(chainInd).size(); ++kernInd){
  
        // Set some metadata that might be needed by the expensive sampling problem
        prevStates.at(chainInd)->meta["iteration"] = sampNums.at(chainInd);
        prevStates.at(chainInd)->meta["IsProposal"] = false;
  
        // kernel prestep
        kernels.at(chainInd).at(kernInd)->PreStep(sampNums.at(chainInd), prevStates.at(chainInd));
  
        // use the kernel to get the next state(s)
        newStates.at(chainInd) = kernels.at(chainInd).at(kernInd)->Step(sampNums.at(chainInd), prevStates.at(chainInd));
        // save when these samples where created
        double now = std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-startTime).count();
        for(auto& state : newStates.at(chainInd))
            state->meta["time"] = now;
       
        // kernel post-processing
        kernels.at(chainInd).at(kernInd)->PostStep(sampNums.at(chainInd), newStates.at(chainInd));
    }
  }

  // add the new states to the SampleCollection (this also increments sampNum and updates prevStates)
  SaveSamples(newStates);

  auto endTime = std::chrono::high_resolution_clock::now();
  totalTime += std::chrono::duration<double>(endTime - startTime).count();
}



void ParallelTempering::SaveSamples(std::vector<std::vector<std::shared_ptr<SamplingState>>> const& newStates)
{   
    assert(newStates.size() == numTemps);
    
    for(unsigned int chainInd=0; chainInd<numTemps; ++chainInd){
        for(unsigned int stateInd=0; stateInd<newStates.at(chainInd).size(); ++stateInd){
            
            // Should we save this sample?
            if( ShouldSave(chainInd, sampNums.at(chainInd)) ) {
                chains.at(chainInd)->Add(newStates.at(chainInd).at(stateInd));

                // Save the QOI if the state has one
                if(newStates.at(chainInd).at(stateInd)->HasMeta("QOI")) {
                    std::shared_ptr<SamplingState> qoi = AnyCast(newStates.at(chainInd).at(stateInd)->meta["QOI"]);
                    QOIs.at(chainInd)->Add(qoi);
                }
            }

            ++sampNums.at(chainInd);
            // Increment the number of samples and break if we're the posterior chain and we hit the max. number
            if(chainInd==numTemps-1){
                if( sampNums.at(numTemps-1)>=numSamps ) { return; }
            }
        }
    }

    for(unsigned int chainInd=0; chainInd<numTemps; ++chainInd){  
        prevStates.at(chainInd) = newStates.at(chainInd).back();
    }
}


std::vector<std::vector<std::shared_ptr<TransitionKernel>>> ParallelTempering::StackKernels(std::vector<std::shared_ptr<TransitionKernel>> const& kerns)
{
    std::vector<std::vector<std::shared_ptr<TransitionKernel>>> newKernels(kerns.size());
    for(unsigned int i=0; i<kerns.size(); ++i){
        newKernels.resize(1);
        newKernels.at(i).at(0) = kerns.at(i);
    }
    return newKernels;
}

Eigen::VectorXd ParallelTempering::ExtractTemps(boost::property_tree::ptree opts)
{
  std::string allTemps = opts.get<std::string>("Inverse Temperatures");
  std::vector<std::string> tempStrings = StringUtilities::Split(allTemps, ',');

  Eigen::VectorXd inverseTemps(tempStrings.size());
  for(unsigned int i=0; i<tempStrings.size(); ++i)
    inverseTemps(i) = std::stod(tempStrings.at(i));

  return inverseTemps;
}
      
std::vector<std::vector<std::shared_ptr<TransitionKernel>>> ParallelTempering::ExtractKernels(boost::property_tree::ptree              opts, 
                                                                                              std::shared_ptr<InferenceProblem> const& problem)
{
  std::string allKernelString = opts.get<std::string>("Kernel Lists");
  std::vector<std::string> chainStrings = StringUtilities::Split(allKernelString, ';');

  std::vector<std::vector<std::shared_ptr<TransitionKernel>>> kernels(chainStrings.size());

  for(unsigned int chainInd=0; chainInd<chainStrings.size(); ++chainInd){
    std::vector<std::string> kernelNames = StringUtilities::Split(chainStrings.at(chainInd), ',');

    unsigned int numBlocks = kernelNames.size();
    assert(numBlocks>0);
    kernels.at(chainInd).resize(numBlocks);

    // Add the block id to a child tree and construct a kernel for each block
    for(int i=0; i<numBlocks; ++i) {
        boost::property_tree::ptree subTree = opts.get_child(kernelNames.at(i));
        subTree.put("BlockIndex",i);

        auto prob = problem->Clone();
        prob->AddOptions(subTree);
        kernels.at(chainInd).at(i) = TransitionKernel::Construct(subTree, prob);
    }
  }

  return kernels;
}

bool ParallelTempering::ShouldSave(unsigned int chainInd, unsigned int const sampNum) const 
{ 
    return ((sampNum>=burnIn) && (scheduler->ShouldSave(sampNum))); 
}
