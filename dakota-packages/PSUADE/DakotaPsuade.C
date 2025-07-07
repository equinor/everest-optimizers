/*  _______________________________________________________________________

    DAKOTA: Design Analysis Kit for Optimization and Terascale Applications
    Copyright (c) 2006, Sandia National Laboratories.
    This software is distributed under the GNU Lesser General Public License.
    For more information, see the README file in the top Dakota directory.
    _______________________________________________________________________ */

//- Class:       DakotaPsuade
//- Description: Implementation code for the DakotaPsuade class
//- Owner:       Brian M. Adams, Sandia National Laboratories

#include "DakotaPsuade.H"
#include <cstddef>
#include <algorithm>
#include <vector>
#include <cstdlib>

static const char rcsId[]="@(#) $Id$";

/** Constructor using default seed */
DakotaPsuade::DakotaPsuade():
  rngSeed(41u), rnumGenerator(rngSeed), uniRealDist(0.0, 1.0)
{ /* empty constructor */ }

/** Constructor using DAKOTA-specified seed */
DakotaPsuade::DakotaPsuade(int seed):
  rngSeed(seed), rnumGenerator(rngSeed), uniRealDist(0.0, 1.0)
{ /* empty constructor */ }

DakotaPsuade::~DakotaPsuade()
{ /* no-op */ }

double DakotaPsuade::PSUADE_drand()
{ return uniRealDist(rnumGenerator); }

/** emulation of PSUADE's integer vector shuffler generateRandomIvector
    presumes permute has been allocated
    populates with [0:num_inputs-1] and permutes */
void DakotaPsuade::generateRandomIvector(int num_inputs, int *permute)
{
  // TODO: make more efficient by using original data in permute instead of 
  // copying
  std::vector<int> p;
  for (int i=0; i<num_inputs; i++) p.push_back(i);
  rand_shuffle(p.begin(), p.end(), rnumGenerator);
  for (int i=0; i<num_inputs; i++) permute[i] = p[i];
}
