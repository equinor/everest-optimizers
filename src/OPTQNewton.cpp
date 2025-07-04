//------------------------------------------------------------------------
// Copyright (C) 1993:
// J.C. Meza
// Sandia National Laboratories
// meza@california.sandia.gov
//------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#ifdef HAVE_CONFIG_H
#include "OPT++_config.h"
#endif

#ifdef HAVE_STD
#include <cstring>
#include <cmath>
#else
#include <string.h>
#include <math.h>
#endif
#include <float.h>
#include "OptQNewton.h"
/* #include "optpp_blas_wrap.h" */
#include "ioformat.h"

#include "Teuchos_SerialSymDenseMatrix.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"

using Teuchos::SerialDenseMatrix;
using Teuchos::SerialDenseVector;
using Teuchos::SerialSymDenseMatrix;

namespace py = pybind11;

// Forward declare your optimization function that uses OptQNewton
std::vector<double> optimize_with_optqnewton(py::function func, const std::vector<double> &x0);

namespace OPTPP
{

  //------------------------------------------------------------------------
  //
  //   Quasi-Newton Method member functions
  //   checkDeriv()
  //   updateH()
  //------------------------------------------------------------------------

  // static char* class_name = "OptQNewton";

  int OptQNewton::checkDeriv() // check the analytic gradient with FD gradient
  {
    return checkAnalyticFDGrad();
  }

  //----------------------------------------------------------------------------
  //
  // Update Hessian using a Quasi-Newton update
  //
  //----------------------------------------------------------------------------
  SerialSymDenseMatrix<int, double> OptQNewton::updateH(SerialSymDenseMatrix<int, double> &Hk, int k)
  {

    double mcheps = DBL_EPSILON;
    double sqrteps = sqrt(mcheps);

    int i;

    NLP1 *nlp = nlprob();
    int nr = nlp->getDim();
    SerialDenseVector<int, double> grad(nr), xc(nlp->getXc().length());
    xc = nlp->getXc();
    grad = nlp->getGrad();

    SerialDenseVector<int, double> D(nr);
    // BFGS formula

    if (k == 0)
    {
      Hessian = 0.0;
      //    D = sx.AsDiagonal()*sx.AsDiagonal();
      //    D = sfx.AsDiagonal();
      double typx, xmax, gnorm;
      //    double gamma;
      // gnorm = Norm2(grad);
      gnorm = sqrt(grad.dot(grad));
      // Initialize xmax, typx and D to default values
      xmax = -1.e30;
      typx = 1.0;
      D = 1.0;

      for (i = 0; i < nr; i++)
        xmax = max(xmax, fabs(xc(i)));
      if (xmax != 0.0)
        typx = xmax;
      if (gnorm != 0.0)
        D = gnorm / typx;
      if (debug_)
      {
        *optout << "UpdateH: gnorm0 = " << gnorm
                << "typx = " << typx << "\n";
      }
      for (i = 0; i < nr; i++)
        Hessian(i, i) = D(i);

      return Hessian;
    }

    SerialDenseVector<int, double> yk(nr), sk(nr), Bsk(nr);
    SerialSymDenseMatrix<int, double> Htmp(nr), Htmp2(nr);

    // yk = grad - gprev;
    // sk = xc   - xprev;
    yk = grad;
    yk -= gprev;
    sk = xc;
    sk -= xprev;

    if (debug_)
    {
      Print(yk);
      Print(sk);
    }

    double gts = gprev.dot(sk);
    double yts = yk.dot(sk);

    // double snorm = Norm2(sk);
    // double ynorm = Norm2(yk);
    double snorm = sqrt(sk.dot(sk));
    double ynorm = sqrt(yk.dot(yk));

    if (debug_)
    {
      *optout << "UpdateH: gts   = " << gts
              << "  yts = " << yts << "\n";
      *optout << "UpdateH: snorm = " << snorm
              << "  ynorm = " << ynorm << "\n";
    }

    if (yts <= sqrteps * snorm * ynorm)
    {
      if (debug_)
      {
        *optout << "UpdateH: <y,s> = " << e(yts, 12, 4) << " is too small\n";
        *optout << "UpdateH: The BFGS update is skipped\n";
      }
      Hessian = Hk;

      return Hk;
    }

    SerialDenseVector<int, double> res(nr), tmp(nr);
    tmp.multiply(Teuchos::LEFT_SIDE, 1.0, Hk, sk, 0.0);
    res = yk;
    res -= tmp;
    // res = yk - Hk*sk;
    if (res.normInf() <= sqrteps)
    {
      if (debug_)
      {
        *optout << "UpdateH: <y,s> = " << e(yts, 12, 4) << " is too small\n";
        *optout << "UpdateH: The BFGS update is skipped\n";
      }
      Hessian = Hk;

      return Hk;
    }

    // Bsk = Hk*sk;
    Bsk.multiply(Teuchos::LEFT_SIDE, 1.0, Hk, sk, 0.0);
    double sBs = sk.dot(Bsk);
    double etol = 1.e-8;

    if (sBs <= etol * snorm * snorm)
    {
      if (debug_)
      {
        *optout << "UpdateH: <y,s> = " << e(yts, 12, 4) << " is too small\n";
        *optout << "UpdateH: The BFGS update is skipped\n";
      }
      // D = sx.AsDiagonal()*sx.AsDiagonal();
      Hk = 0;
      for (i = 0; i < nr; i++)
        Hk(i, i) = sx(i) * sx(i);
      Hessian = Hk;

      return Hk;
    }

    // Otherwise update the Hessian approximation
    if (debug_)
    {
      //    *optout << "\nUpdateH: before update, k = " << k << "\n";
      //    FPrint(optout, Hk);
    }

    SerialDenseMatrix<int, double> Htmp3(Htmp.numRows(), Htmp.numCols());
    Htmp3.multiply(Teuchos::NO_TRANS, Teuchos::TRANS, -1 / sBs, Bsk, Bsk, 0.0);

    for (int i = 0; i < Htmp3.numRows(); i++)
      for (int j = 0; j <= i; j++)
      {
        Htmp(i, j) = Htmp3(i, j);
      }

    // Htmp = Htmp + (yk * yk.t()) / yts;
    Htmp3.multiply(Teuchos::NO_TRANS, Teuchos::TRANS, 1 / yts, yk, yk, 0.0);
    for (int i = 0; i < nr; i++)
      for (int j = 0; j <= i; j++)
      {
        Htmp2(i, j) = Htmp3(i, j);
      }

    Htmp += Htmp2;

    Htmp += Hk;

    Hk = Htmp;

    // Htmp.Release();
    SerialDenseVector<int, double> Bgk(nr);
    Bgk.multiply(Teuchos::LEFT_SIDE, 1.0, Hk, grad, 0.0);
    double gBg = grad.dot(Bgk);
    double gg = grad.dot(grad);
    double ckp1 = gBg / gg;
    if (debug_)
    {
      //    *optout << "\nUpdateH: after update, k = " << k << "\n";
      //    FPrint(optout, Hk);
      *optout << "UpdateH: sBs  = " << sBs << "\n";
      *optout << "UpdateH: ckp1 = " << ckp1 << "\n";
    }
    Hessian = Hk;

    return Hk;
  }

} // namespace OPTPP

PYBIND11_MODULE(everest_optimizers_test, m)
{
  m.doc() = "Everest Optimizers using OptQNewton algorithm";

  m.def("test_optpp", []()
        { return "Everest Optimizers OptQNewton binding compilation successful!"; });

  // Main optimization function using actual OptQNewton algorithm
  m.def("optimize_python_func", [](py::function func, std::vector<double> x0)
        { return optimize_with_optqnewton(func, x0); }, "Optimize a Python function using OPTPP's OptQNewton algorithm", py::arg("func"), py::arg("x0"));

  // Backward compatibility: simple optimization function
  m.def("optimize_simple", []()
        {
        // Test quadratic function: f(x,y) = (x-1)^2 + (y-2)^2
        auto func = [](const std::vector<double>& x) -> double {
            return (x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 2.0) * (x[1] - 2.0);
        };

        // Wrap C++ lambda as py::function
        py::function py_func = py::cpp_function([func](const std::vector<double>& x) {
            return func(x);
        });

        std::vector<double> x0 = {0.0, 0.0};
        return optimize_with_optqnewton(py_func, x0); }, "Test optimization with quadratic function using OptQNewton");

  // Additional utility functions
  m.def("get_everest_version", []()
        { return "Everest Optimizers with OptQNewton BFGS implementation"; }, "Get version information about the Everest Optimizers implementation");
}