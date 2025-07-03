/*  _______________________________________________________________________

    DAKOTA: Design Analysis Kit for Optimization and Terascale Applications
    Copyright (c) 2006, Sandia National Laboratories.
    This software is distributed under the GNU Lesser General Public License.
    For more information, see the README file in the top Dakota directory.
    _______________________________________________________________________ */

//- Class:       DakotaPsuade
//- Description: This class provides necessary interface for DAKOTA with PSUADE
//- Owner:       Brian M. Adams, Sandia National Laboratories

#ifndef DAKOTA_PSUADE_H
#define DAKOTA_PSUADE_H

#include <boost/version.hpp>
#if (BOOST_VERSION < 107000) && !defined(BOOST_ALLOW_DEPRECATED_HEADERS)
//could alternately use: #define BOOST_PENDING_INTEGER_LOG2_HPP 1
#define BOOST_ALLOW_DEPRECATED_HEADERS 1
#include <boost/random/mersenne_twister.hpp>
#undef BOOST_ALLOW_DEPRECATED_HEADERS
#else
#include <boost/random/mersenne_twister.hpp>
#endif
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#define PSUADE_UNDEFINED -DBL_MAX

/// glue class providing RNG services to PSUADE
class DakotaPsuade
{

public:

  DakotaPsuade();

  /// constructor that accepts a random seed
  DakotaPsuade(int seed);

  ~DakotaPsuade();

  // for safety for now, only allowing element-wise access to sampleMatrix
  /// return the i,j entry of the sample matrix
  double sample_matrix(int i, int j);

  /// Random shuffle with C++17 shuffle API, but using Boost for portability
  template<class RandomIt, class URBG>
  void rand_shuffle(RandomIt first, RandomIt last, URBG&& g);

protected:

  /// disallow copy/assignment due to pointer
  DakotaPsuade(const DakotaPsuade&);
  DakotaPsuade& operator=(const DakotaPsuade&);

  //-- Member functions

  /// could contain initializer for RNG -- not used (as named in MOAT*)
  void PSUADE_randInit() { /* no-op */ }

  /// generate a random number in [0,1) (half open) (as named in MOAT*) 
  double PSUADE_drand();

  /// create a vector with integers [0:num_inputs-1] and permute
  /// (as named in MOAT*)
  void generateRandomIvector(int num_inputs, int *permute);

  //-- Data

  /// seed for shared generator
  unsigned int rngSeed;

  /// shared Mersenne Twister generator
  boost::mt19937 rnumGenerator;

  /// distribution of U[0,1) random numbers
  boost::random::uniform_real_distribution<> uniRealDist;

  /// matrix of samples to evaluate (name as given in client MOAT* files)
  double **sampleMatrix_;

};

inline double DakotaPsuade::sample_matrix(int i, int j)
{ return sampleMatrix_[i][j]; }


/*
   Should be portable for a given version of Boost, when passing either a std
   or boost URBG, such as mt19937.

   Taken from reference implementation example at
   https://en.cppreference.com/w/cpp/algorithm/random_shuffle, which is similar
   to the libc++ implementation (and perhaps less optimized than libstdc++).

   RATIONALE: While the Mersenne Twister and other RNGs are cross-platform
   deterministic, shuffle and uniform_int_distribution themselves have
   implementation details that vary. Using the boost uniform_int_distribution
   with a custom shuffle stabilizes this for a given Boost version.
*/
template<class RandomIt, class URBG>
void DakotaPsuade::rand_shuffle(RandomIt first, RandomIt last, URBG&& g)
{
  typedef typename std::iterator_traits<RandomIt>::difference_type diff_t;
  // uses the Boost distribution from cross-platform portability (though may
  // change between Boost versions)
  typedef boost::random::uniform_int_distribution<diff_t> distr_t;
  typedef typename distr_t::param_type param_t;

  distr_t D;
  diff_t n = last - first;
  for (diff_t i = n-1; i > 0; --i)
      std::swap(first[i], first[D(g, param_t(0, i))]);
}


#endif
