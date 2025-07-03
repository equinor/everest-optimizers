
# Compile flags options
option(MUQ_USE_MPI "Use OpenMPI" OFF)
option(MUQ_USE_OPENMP "Use OpenMP (non OS X)" ON)
option(MUQ_USE_MKL "Use the Eigen wrapper around the Intel MKL" OFF)

# Bindings 
option(MUQ_USE_PYTHON "Compile Python bindings using pybind11" ON)

# External dependencies
option(MUQ_USE_EIGEN3 "Allow the use of Eigen. (Most of MUQ requires this.)" ON)
option(MUQ_USE_BOOST "Allow the use of Boost.  (Most of MUQ requires this.)" ON)
option(MUQ_USE_NANOFLANN "Allow the use of Nanoflann." ON)
option(MUQ_USE_NLOPT "Include NLOPT in the optimization module" ON)
option(MUQ_USE_SUNDIALS "Compile the Sundials wrapper." ON)
option(MUQ_USE_HDF5 "Support the use of HDF5." ON)
option(MUQ_USE_SPDLOG "Enable the use of SPDLOG." ON)
option(MUQ_USE_OTF2 "Enable the use of OTF2." ON)
option(MUQ_USE_STANMATH "Enable the use of stan math functionality." ON)
option(MUQ_USE_PARCER "Use ParCer (PARallel CEReal) MPI wrapper." ON)

# Tests 
option(MUQ_USE_GTEST "Enable unit testing using GTEST" OFF)
