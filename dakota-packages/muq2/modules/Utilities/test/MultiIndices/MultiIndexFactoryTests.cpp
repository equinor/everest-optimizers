#include <iostream>

#include "gtest/gtest.h"

#include "MUQ/Utilities/MultiIndices/MultiIndexFactory.h"

using namespace muq::Utilities;

/*
  Utilities_MultiIndices.CreateAnisotropic
  ----------------------------------------------------

  Purpose:
  Verify for an exemplary call of the CreateAnisotropic factory method that
  the anisotropic multi-index set is constructed as desired.

  Test:
  The set of 2-dimensional multi-indices that are feasible given the
  weight vector [.5, .25] and the cutoff threshold epsilon = .1 is:
      [0,0], [1,0], [0,1], [1,1], [2,0], [3,0]
  For simplicity, the multi-indices in the created set are printed out
  and their total number is expected to be 6.
*/
TEST(Utilities_MultiIndices, CreateAnisotropic)
{
  Eigen::RowVectorXf weights(2);
  weights << .5, .25;
  auto indexSet = MultiIndexFactory::CreateAnisotropic(weights, .1);
  auto indices = indexSet->GetAllMultiIndices();
  for (auto index = indices.begin(); index != indices.end(); ++index)
      std::cout << (*index)->ToString() << '\n';
  EXPECT_EQ(6, indexSet->GetAllMultiIndices().size());
 }