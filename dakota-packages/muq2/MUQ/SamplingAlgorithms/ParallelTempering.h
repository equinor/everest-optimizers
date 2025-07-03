#ifndef PARALLELTEMPERING_H
#define PARALLELTEMPERING_H

#include "MUQ/SamplingAlgorithms/InferenceProblem.h"
#include "MUQ/SamplingAlgorithms/TransitionKernel.h"
#include "MUQ/SamplingAlgorithms/ThinScheduler.h"
#include "MUQ/SamplingAlgorithms/MarkovChain.h"

#include <vector>

#include <boost/property_tree/ptree.hpp>

namespace muq{
  namespace SamplingAlgorithms{

    /** @ingroup MCMC
        @class ParallelTempering
        @brief Defines an MCMC sampler with multiple chains running on problems with different temperatues
        @details
        <B>Configuration Parameters:</B>
        Parameter Key | Type | Default Value | Description |
        ------------- | ------------- | ------------- | ------------- |
        "NumSamples"  | Int | - | The total number of steps (including burnin) to take, i.e., the length of the Markov chain with temperature 1. |
        "BurnIn"      | Int | 0 | The number of steps at the beginning of the chain to ignore. |
        "PrintLevel"  | Int | 3 | The amount of information to print to std::cout. Valid values are in [0,1,2,3] with  0 = Nothing, 3 = The most |
        "Kernel Lists"  | String | - | A semi-colon separated list of comma separated lists containing the name of other parameter blocks that define the transition kernels for each Metropolis-in-Gibbs block. |
        "Inverse Temperatures" | String | - | A comma-separated list of floats containing the inverse temperature for each chain.  Values of 0 correspond to the prior, values of 1 correspond to the posterior.  The last value must be 1. |
        "Swap Increment" | Int | 2 | How often to apply the swap kernel, which might exchange states between chains. The first swap occurs after "Swap Increment" steps have been taken.  It is therefore possible to prevent any swaps from occuring by setting the swap increment to a value larger than the number of samples. For large number of chains, the DEO option will typically perform better than the SEO method.  Similar efficiency is often observed for a small number of chains.|
        "Swap Type" | String | "DEO" | The type of swapping mechanism to use.  Current options are "DEO" and "SEO" corresponding to the variants described in \cite Syed2019 |
        "Adapt Start" | Int | 100 | The number of steps to take before starting to adapt temperatures.  Setting this value larger than NumSamples will disable adaptation. |
    */
    class ParallelTempering
    {

    public:


        ParallelTempering(boost::property_tree::ptree              opts,
                          std::shared_ptr<InferenceProblem> const& problem);

        ParallelTempering(boost::property_tree::ptree                    opts,
                          Eigen::VectorXd                                inverseTemps,
                          std::vector<std::shared_ptr<TransitionKernel>> kernels);

        ParallelTempering(boost::property_tree::ptree                                 opts,
                          Eigen::VectorXd                                             inverseTemps,
                          std::vector<std::vector<std::shared_ptr<TransitionKernel>>> kernels);

        
      /// Set the state of the MCMC chain
      /**
        If no steps have been taken, this function sets the starting point.
      */
      void SetState(std::vector<std::shared_ptr<SamplingState>> const& x0);
      void SetState(std::vector<Eigen::VectorXd> const& x0);
      void SetState(std::vector<std::vector<Eigen::VectorXd>> const& x0);

      template<typename... Args>
      inline void SetState(Args const&... args) {
        std::vector<Eigen::VectorXd> vec;
        SetStateRecurse(vec, args...);
      }
    
      
      /** Returns the inverse temperature of one of the chains. 
        @param[in] chainInd The index of the chain.
        @return The inverse temperature that multiplies the log-likelihood in this chain
      */
      double GetInverseTemp(unsigned int chainInd) const;

      /** Returns the transition kernel used by one of the parallel chains 
      @params[in] chainInd The index of the chain.
      */
      std::vector<std::shared_ptr<TransitionKernel>> const& Kernels(unsigned int chainInd) const;


      /** Runs each parallel chain, starting from the current state.  Before this method is called,
          the state should be set using either the SetState function or another version of the `Run` method.
      */
      std::shared_ptr<MarkovChain> Run();
      
      std::shared_ptr<MarkovChain> Run(Eigen::VectorXd const& x0){return Run(std::vector<Eigen::VectorXd>(numTemps,x0));}

      /** Runs each parallel chain.  Assumes each chain operates on a density with a single input.
       */
      std::shared_ptr<MarkovChain> Run(std::vector<Eigen::VectorXd> const& x0);
      
      /** Runs each parallel chain.  Starts each chain at a different initial point. */
      std::shared_ptr<MarkovChain> Run(std::vector<std::vector<Eigen::VectorXd>> const& initialPoints);


      //double TotalTime() { return totalTime; }

      /** When the Run method is called, the Sample() method is called until the
          total number of samples generated by this class is equal to a private
          member variable `numSamps`, which is generally set in the
          options passed to the constructor.  In order to generate more samples
          after an initial call to `Run`, the numSamps variables needs to be
          increasd.  This function essentially sets numSamps=numSamps+numNewSamps.

          Typical usage will be something like:
          @code{.cpp}
          std::shared_ptr<SampleCollection> samps = mcmc->Run(startPt);
          for(unsigned int batchInd=1; batchInd<numBatches; ++batchInd){
            mcmc->AddNumSamps(batchSizes);
            mcmc->Run();
          }
          @endcode


          @param[in] numNewSamps The number of new samples we want to add to numSamps.
                                 After calling this function, the next call to Run
                                 will add an additional numNewSamps to the SampleCollection.


      */
      void AddNumSamps(unsigned int numNewSamps){numSamps+=numNewSamps;};

      /** Returns the current value of the private numSamps variable.  When `Run`
          is called, it calls the `Sample` function until this number of samples
          has been generated.  Note  that unless numSamps is updated by calling
          AddNumSamps, subsequent calls to Run will not produce any new samples.
      */
      unsigned int NumSamps() const{return numSamps;};

      /** Returns the samples generated by the algorithm so far. */
      std::shared_ptr<MarkovChain> GetSamples() const{return chains.at(numTemps-1);}

      /** Returns the Quantities of Interest (if any) computed so far. */
      std::shared_ptr<MarkovChain> GetQOIs() const{return QOIs.at(numTemps-1);}

      /// Number of temperatures in the temperature schedule
      const unsigned int numTemps;
      
    protected:
      
      Eigen::VectorXd cumulativeSwapProb; // A running sum of the swap probabilities
      Eigen::VectorXd successfulSwaps; // The number of succesfull swaps between chains i and i+1
      Eigen::VectorXd attemptedSwaps; // The number of attempted swaps between chains i and i+1

      
      void PrintStatus(unsigned int currInd) const{PrintStatus("",currInd);};
      void PrintStatus(std::string prefix, unsigned int currInd) const;

      /// Adapts the temperatures according to the procedure outlined in Section 5 of \cite Syed2019
      void AdaptTemperatures();

      /** Takes one "step" with each kernel. Updates the prevStates member variable and saves states to the chains variable. */
      void Sample();

      /** Swap states between chains.  Updates the prevStates private member variable. */
      void SwapStates();
      
      void SaveSamples(std::vector<std::vector<std::shared_ptr<SamplingState>>> const& newStates);

      /**
       * @brief Returns true if a sample of a particular chain should be saved
       */
      bool ShouldSave(unsigned int chainInd, unsigned int sampNum) const;
      
      Eigen::VectorXd CollectInverseTemps() const;

      // Samples and quantities of interest will be stored in the MarkovChain class
      std::vector<std::shared_ptr<InferenceProblem>> problems;
      std::vector<std::shared_ptr<MarkovChain>> chains;
      std::vector<std::shared_ptr<MarkovChain>> QOIs;

      std::shared_ptr<SaveSchedulerBase> scheduler;
      std::shared_ptr<SaveSchedulerBase> schedulerQOI;

      unsigned int numSamps;
      unsigned int burnIn;
      unsigned int printLevel;
      unsigned int swapIncr;
      bool seoSwaps = false; // True if stochastic "SEO" swaps of Syed et al. 2019 should be used.  Otherwise the deterministic DEO swaps are employed.
      unsigned int adaptIncr = 2;
      unsigned int nextAdaptInd;

      /** A vector of transition kernels: One for each block
      kernels[chainInd][blockInd]
      */
      std::vector<std::vector<std::shared_ptr<TransitionKernel>>> kernels;
    
      std::vector<std::shared_ptr<SamplingState>> prevStates;

    private:

      bool evenSwap = true;
      
      std::vector<unsigned int> sampNums;
      double totalTime = 0.0;

      static std::vector<std::vector<std::shared_ptr<TransitionKernel>>> StackKernels(std::vector<std::shared_ptr<TransitionKernel>> const& kerns);
      static Eigen::VectorXd ExtractTemps(boost::property_tree::ptree opts);
      static std::vector<std::vector<std::shared_ptr<TransitionKernel>>> ExtractKernels(boost::property_tree::ptree opts, std::shared_ptr<InferenceProblem> const& prob);

      template<typename T>
      static std::vector<std::vector<T>> StackObjects(std::vector<T> const& kerns)
      {
            std::vector<std::vector<T>> newKernels(kerns.size(), std::vector<T>(1));
            for(unsigned int i=0; i<kerns.size(); ++i)
                newKernels.at(i).at(0) = kerns.at(i);
            return newKernels;
      }

      /// Checks a sampling state to make sure it has the metadata necessary to swap states.
      static void CheckForMeta(std::shared_ptr<SamplingState> const& state);



    //   void Setup(boost::property_tree::ptree pt,
    //              std::vector<std::shared_ptr<TransitionKernel>> const& kernelsIn);

    //   void Setup(boost::property_tree::ptree pt, std::shared_ptr<AbstractSamplingProblem> const& problem);

    }; // class ParallelTempering 

  } // namespace SamplingAlgorithms
} // namespace muq

#endif // #ifndef SINGLECHAINMCMC_H
