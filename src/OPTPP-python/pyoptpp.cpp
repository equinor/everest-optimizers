#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BoundConstraint.h"
#include "CompoundConstraint.h"
#include "Constraint.h"
#include "ConstraintBase.h"
#include "LinearEquation.h"
#include "LinearInequality.h"
#include "NLF.h"
#include "NLP.h"
#include "NLPBase.h"
#include "NonLinearConstraint.h"
#include "NonLinearEquation.h"
#include "NonLinearInequality.h"
#include "Opt.h"
#include "OptConstrQNewton.h"
#include "OptQNIPS.h"
#include "OptQNewton.h"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_SerialSymDenseMatrix.hpp"

namespace py = pybind11;
using namespace OPTPP;

using T_SerialDenseVector = Teuchos::SerialDenseVector<int, double>;
using T_SerialSymDenseMatrix = Teuchos::SerialSymDenseMatrix<int, double>;
using T_SerialDenseMatrix = Teuchos::SerialDenseMatrix<int, double>;

// Dummy C++ function for the default case
void default_update_model(int, int, T_SerialDenseVector) {}

// C++ NLF1 class that holds Python callbacks - fully C++-managed
// This avoids Python/C++ ownership conflicts
class CallbackNLF1 : public NLF1 {
private:
  py::function py_eval_f;
  py::function py_eval_g;
  py::function py_eval_cf;
  py::function py_eval_cg;
  bool has_cf;
  bool has_cg;

public:
  // Constructor for objective function (no constraint callbacks)
  CallbackNLF1(int ndim, py::function eval_f, py::function eval_g)
      : NLF1(ndim), py_eval_f(eval_f), py_eval_g(eval_g), py_eval_cf(), py_eval_cg(), has_cf(false),
        has_cg(false) {}

  // Constructor for constraint function (with constraint callbacks)
  CallbackNLF1(
      int ndim, py::function eval_f, py::function eval_g, py::function eval_cf, py::function eval_cg
  )
      : NLF1(ndim), py_eval_f(eval_f), py_eval_g(eval_g), py_eval_cf(eval_cf), py_eval_cg(eval_cg),
        has_cf(true), has_cg(true) {}

  void initFcn() override {}

  void eval() override {
    this->fvalue = this->evalF(this->mem_xc);
    this->mem_grad = this->evalG(this->mem_xc);
    this->nfevals++;
    this->ngevals++;
  }

  real evalF(const T_SerialDenseVector& x) override {
    py::gil_scoped_acquire gil;
    try {
      return py_eval_f(x).cast<real>();
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("Error in evalF callback: ") + e.what());
    }
  }

  T_SerialDenseVector evalG(const T_SerialDenseVector& x) override {
    py::gil_scoped_acquire gil;
    try {
      py::object result = py_eval_g(x);
      py::array_t<double, py::array::c_style | py::array::forcecast> result_array =
          result.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
      py::buffer_info buf = result_array.request();

      if (buf.ndim != 1 || buf.shape[0] != this->getDim()) {
        throw std::runtime_error("evalG: Returned numpy array has wrong dimensions!");
      }
      // The optimizer depends on the internal state of mem_grad being updated.
      // We must copy the data from the Python override into mem_grad.
      if (mem_grad.length() != this->getDim()) {
        mem_grad.resize(this->getDim());
      }
      std::memcpy(mem_grad.values(), buf.ptr, this->getDim() * sizeof(double));
      return mem_grad;
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("Error in evalG callback: ") + e.what());
    }
  }

  T_SerialDenseVector evalG() override { return this->evalG(this->getXc()); }

  T_SerialSymDenseMatrix evalH(T_SerialDenseVector& x) override {
    // Return identity matrix - Hessian not implemented for callbacks
    int dim = this->getDim();
    T_SerialSymDenseMatrix H(dim);
    for (int i = 0; i < dim; ++i) {
      H(i, i) = 1.0;
    }
    return H;
  }

  T_SerialDenseVector evalCF(const T_SerialDenseVector& x) override {
    if (!has_cf) {
      return T_SerialDenseVector(1);
    }

    py::gil_scoped_acquire gil;
    try {
      py::object result = py_eval_cf(x);
      py::array_t<double, py::array::c_style | py::array::forcecast> result_array =
          result.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
      py::buffer_info buf = result_array.request();

      if (buf.ndim != 1) {
        throw std::runtime_error("evalCF: Returned numpy array must be 1-D!");
      }

      T_SerialDenseVector constraints(buf.shape[0]);
      std::memcpy(constraints.values(), buf.ptr, buf.shape[0] * sizeof(double));
      return constraints;
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("Error in evalCF callback: ") + e.what());
    }
  }

  T_SerialDenseMatrix evalCG(const T_SerialDenseVector& x) override {
    if (!has_cg) {
      return T_SerialDenseMatrix(1, this->getDim());
    }

    py::gil_scoped_acquire gil;
    try {
      py::object result = py_eval_cg(x);
      py::array_t<double, py::array::c_style | py::array::forcecast> result_array =
          result.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
      py::buffer_info buf = result_array.request();

      if (buf.ndim != 2) {
        throw std::runtime_error("evalCG: Returned numpy array must be 2-D!");
      }

      T_SerialDenseMatrix grad(buf.shape[0], buf.shape[1]);
      std::memcpy(grad.values(), buf.ptr, buf.shape[0] * buf.shape[1] * sizeof(double));
      return grad;
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("Error in evalCG callback: ") + e.what());
    }
  }
};

PYBIND11_MODULE(pyoptpp, m) {
  m.doc() = "Python bindings for OPTPP library";

  // Bind Teuchos::SerialDenseVector
  py::class_<T_SerialDenseVector>(m, "SerialDenseVector", py::buffer_protocol())
      .def(py::init<int>())
      .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
        auto vec = new T_SerialDenseVector(arr.size());
        std::memcpy(vec->values(), arr.data(), arr.size() * sizeof(double));
        return vec;
      }))
      .def("size", &T_SerialDenseVector::length)
      .def("__len__", &T_SerialDenseVector::length)
      .def(
          "__getitem__",
          [](const T_SerialDenseVector& v, int i) {
            if (i < 0)
              i += v.length();
            if (i < 0 || i >= v.length())
              throw py::index_error();
            return v(i);
          }
      )
      .def(
          "__setitem__",
          [](T_SerialDenseVector& v, int i, double val) {
            if (i < 0)
              i += v.length();
            if (i < 0 || i >= v.length())
              throw py::index_error();
            v(i) = val;
          }
      )
      .def(
          "to_numpy",
          [](T_SerialDenseVector& v) {
            return py::array_t<double>(
                {(size_t)v.length()}, {sizeof(double)}, v.values(), py::cast(v)
            );
          }
      )
      .def_buffer([](T_SerialDenseVector& v) -> py::buffer_info {
        return py::buffer_info(
            v.values(), sizeof(double), py::format_descriptor<double>::format(), 1,
            {(size_t)v.length()}, {sizeof(double)}
        );
      });

  // Bind Teuchos::SerialSymDenseMatrix
  py::class_<Teuchos::SerialSymDenseMatrix<int, double>>(m, "SerialSymDenseMatrix")
      .def(py::init<int>());

  // Bind Teuchos::SerialDenseMatrix
  py::class_<T_SerialDenseMatrix>(m, "SerialDenseMatrix", py::buffer_protocol())
      .def(py::init<int, int>())
      .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
        if (arr.ndim() != 2) {
          throw std::runtime_error("Numpy array must be 2-D");
        }
        auto mat = new T_SerialDenseMatrix(arr.shape(0), arr.shape(1));
        std::memcpy(mat->values(), arr.data(), arr.size() * sizeof(double));
        return mat;
      }))
      .def_buffer([](T_SerialDenseMatrix& m) -> py::buffer_info {
        return py::buffer_info(
            m.values(), sizeof(double), py::format_descriptor<double>::format(), 2,
            {(size_t)m.numRows(), (size_t)m.numCols()},
            {sizeof(double) * m.numCols(), sizeof(double)}
        );
      });

  // Bind Constraint so it can be used in lists
  py::class_<Constraint>(m, "Constraint")
      .def("getNumOfCons", &Constraint::getNumOfCons)
      .def("getNumOfVars", &Constraint::getNumOfVars)
      .def("getLower", &Constraint::getLower)
      .def("getUpper", &Constraint::getUpper)
      .def("evalResidual", &Constraint::evalResidual, py::arg("x"))
      .def("evalGradient", &Constraint::evalGradient, py::arg("x"))
      .def(
          "amIFeasible",
          [](Constraint& self, const T_SerialDenseVector& x, double epsilon = 1e-6) {
            return self.amIFeasible(x, epsilon);
          },
          py::arg("x"), py::arg("epsilon") = 1e-6
      );

  // Bind CompoundConstraint
  py::class_<CompoundConstraint>(m, "CompoundConstraint")
      .def("getNumOfCons", &CompoundConstraint::getNumOfCons)
      .def("getNumOfVars", &CompoundConstraint::getNumOfVars)
      .def("evalResidual", &CompoundConstraint::evalResidual, py::arg("x"))
      .def(
          "amIFeasible",
          [](CompoundConstraint& self, const T_SerialDenseVector& x, double epsilon = 1e-6) {
            return self.amIFeasible(x, epsilon);
          },
          py::arg("x"), py::arg("epsilon") = 1e-6
      );

  // Bind NLPBase first (abstract base class)
  py::class_<NLPBase>(m, "NLPBase");

  // Bind NLP class (handle for NLPBase)
  py::class_<NLP>(m, "NLP")
      .def_static(
          "create", [](NLPBase* nlp) { return new NLP(nlp); }, py::return_value_policy::reference,
          py::arg("nlp")
      )
      .def("initFcn", &NLP::initFcn)
      .def("evalF", static_cast<double (NLP::*)()>(&NLP::evalF))
      .def("evalCF", &NLP::evalCF, py::arg("x"))
      .def("evalCG", &NLP::evalCG, py::arg("x"))
      .def("getXc", &NLP::getXc)
      .def("getF", &NLP::getF)
      .def("getDim", &NLP::getDim)
      .def("setX", static_cast<void (NLP::*)(const T_SerialDenseVector&)>(&NLP::setX));

  // Bind NLF1 - only expose methods needed by optimizers
  py::class_<NLF1, NLPBase>(m, "NLF1")
      .def_static(
          "create",
          [](int ndim, py::function eval_f, py::function eval_g,
             const T_SerialDenseVector& x0) -> NLF1* {
            // Create the C++ object - fully managed by C++
            CallbackNLF1* nlf1 = new CallbackNLF1(ndim, eval_f, eval_g);
            nlf1->setX(x0);
            nlf1->setIsExpensive(true);

            return static_cast<NLF1*>(nlf1);
          },
          py::return_value_policy::reference, py::arg("ndim"), py::arg("eval_f"), py::arg("eval_g"),
          py::arg("x0"), "Create an objective NLF1 object with Python callbacks (C++-managed)"
      )
      .def_static(
          "create_constrained",
          [](int ndim, py::function eval_cf, py::function eval_cg, const T_SerialDenseVector& x0) {
            // Create a CallbackNLF1 with dummy objective (not used for constraints)
            auto dummy_f = py::cpp_function([](const T_SerialDenseVector& x) { return 0.0; });
            auto dummy_g = py::cpp_function([ndim](const T_SerialDenseVector& x) {
              return py::array_t<double>(ndim);
            });

            CallbackNLF1* nlf1 = new CallbackNLF1(ndim, dummy_f, dummy_g, eval_cf, eval_cg);
            nlf1->setX(x0);
            nlf1->setIsExpensive(true);

            return static_cast<NLPBase*>(nlf1);
          },
          py::return_value_policy::reference, py::arg("ndim"), py::arg("eval_cf"),
          py::arg("eval_cg"), py::arg("x0"),
          "Create a constraint NLF1 object with Python callbacks (C++-managed)"
      )
      .def("getXc", &NLF1::getXc)
      .def("getF", &NLF1::getF)
      .def("getDim", &NLF1::getDim)
      .def(
          "setConstraints", [](NLF1& self, CompoundConstraint* cc) { self.setConstraints(cc); },
          py::arg("compound_constraint")
      );

  // Bind SearchStrategy enum
  py::enum_<SearchStrategy>(m, "SearchStrategy")
      .value("TrustRegion", SearchStrategy::TrustRegion)
      .value("LineSearch", SearchStrategy::LineSearch)
      .value("TrustPDS", SearchStrategy::TrustPDS)
      .export_values();

  // Bind MeritFcn enum for interior-point methods
  py::enum_<MeritFcn>(m, "MeritFcn")
      .value("NormFmu", MeritFcn::NormFmu)
      .value("ArgaezTapia", MeritFcn::ArgaezTapia)
      .value("VanShanno", MeritFcn::VanShanno)
      .export_values();

  // Bind OptQNewton
  py::class_<OptQNewton>(m, "OptQNewton")
      .def(
          py::init([](NLF1* p) {
            NLP1* p_base = static_cast<NLP1*>(p);
            return new OptQNewton(p_base, &default_update_model);
          }),
          py::arg("p"), py::keep_alive<0, 1>()
      )
      .def("setDebug", &OPTPP::OptimizeClass::setDebug, "Set debug flag to true")
      .def("optimize", &OptQNewton::optimize)
      .def("printStatus", &OptQNewton::printStatus, py::arg("s"))
      .def("cleanup", &OptQNewton::cleanup)
      .def("setSearchStrategy", &OptQNewton::setSearchStrategy, py::arg("s"))
      .def(
          "setOutputFile",
          static_cast<int (OptQNewton::*)(const char*, int)>(&OptQNewton::setOutputFile),
          py::arg("filename"), py::arg("mode") = 0
      )
      .def("setTRSize", &OptQNewton::setTRSize, py::arg("size"));

  // Bind ConstraintBase so we can establish the inheritance hierarchy
  py::class_<ConstraintBase>(m, "ConstraintBase")
      .def_static("delete", [](ConstraintBase* constraint) { delete constraint; });

  // Bind LinearConstraint as it's a base for Equation and Inequality
  py::class_<LinearConstraint, ConstraintBase>(m, "LinearConstraint");

  // Bind LinearEquation
  py::class_<LinearEquation, LinearConstraint>(m, "LinearEquation")
      .def_static(
          "create",
          [](const T_SerialDenseMatrix& A, const T_SerialDenseVector& rhs) {
            return new LinearEquation(A, rhs);
          },
          py::return_value_policy::reference, py::arg("A"), py::arg("rhs")
      );

  // Bind LinearInequality
  py::class_<LinearInequality, LinearConstraint>(m, "LinearInequality")
      .def_static(
          "create",
          [](const T_SerialDenseMatrix& A, const T_SerialDenseVector& rhs) {
            return new LinearInequality(A, rhs);
          },
          py::return_value_policy::reference, py::arg("A"), py::arg("rhs")
      );

  // Bind BoundConstraint
  py::class_<BoundConstraint, ConstraintBase>(m, "BoundConstraint")
      .def_static(
          "create",
          [](int nc, const T_SerialDenseVector& A, const T_SerialDenseVector& rhs) {
            return new BoundConstraint(nc, A, rhs);
          },
          py::return_value_policy::reference, py::arg("nc"), py::arg("A"), py::arg("rhs")
      );

  // Helper to create CompoundConstraint from bound constraints only
  m.def(
      "create_compound_constraint",
      [](py::array_t<double, py::array::c_style | py::array::forcecast> lower,
         py::array_t<double, py::array::c_style | py::array::forcecast> upper) {
        if (lower.ndim() != 1 || upper.ndim() != 1)
          throw std::runtime_error("Lower and upper arrays must be 1-D");
        if (lower.size() != upper.size())
          throw std::runtime_error("Lower and upper arrays must have same length");
        int n = lower.size();
        T_SerialDenseVector lb(n);
        std::memcpy(lb.values(), lower.data(), n * sizeof(double));
        T_SerialDenseVector ub(n);
        std::memcpy(ub.values(), upper.data(), n * sizeof(double));
        auto bc = new BoundConstraint(n, lb, ub);
        auto cc = new CompoundConstraint(Constraint(bc));
        return cc;
      },
      py::return_value_policy::reference, py::arg("lower"), py::arg("upper")
  );

  // Helper to create CompoundConstraint from a list of constraints
  m.def(
      "create_compound_constraint",
      [](std::vector<ConstraintBase*> constraints) {
        OptppArray<Constraint> constraint_array{};
        constraint_array.reserve(constraints.size());
        for (auto& c : constraints) {
          constraint_array.append(Constraint(c));
        }
        return new CompoundConstraint(constraint_array);
      },
      py::return_value_policy::reference, py::arg("constraints")
  );

  // Helper to create a Constraint object from a variety of constraint types
  m.def(
      "create_constraint",
      [](ConstraintBase* constraint_obj) { return new Constraint(constraint_obj); },
      py::return_value_policy::reference
  );

  py::class_<NonLinearConstraint, ConstraintBase>(m, "NonLinearConstraint");

  // Bind NonLinearInequality
  py::class_<NonLinearInequality, NonLinearConstraint>(m, "NonLinearInequality")
      .def_static(
          "create",
          [](NLP* nlprob, const T_SerialDenseVector& rhs, int numconstraints) {
            return new NonLinearInequality(nlprob, rhs, numconstraints);
          },
          py::return_value_policy::reference, py::arg("nlprob"), py::arg("rhs"),
          py::arg("numconstraints") = 1
      )
      .def_static(
          "create",
          [](NLP* nlprob, const T_SerialDenseVector& lower, const T_SerialDenseVector& upper,
             int numconstraints) {
            return new NonLinearInequality(nlprob, lower, upper, numconstraints);
          },
          py::return_value_policy::reference, py::arg("nlprob"), py::arg("lower"), py::arg("upper"),
          py::arg("numconstraints") = 1
      )
      .def_static(
          "create",
          [](NLP* nlprob, int numconstraints) {
            return new NonLinearInequality(nlprob, numconstraints);
          },
          py::return_value_policy::reference, py::arg("nlprob"), py::arg("numconstraints") = 1
      );

  // Bind NonLinearEquation
  py::class_<NonLinearEquation, NonLinearConstraint>(m, "NonLinearEquation")
      .def_static(
          "create",
          [](NLP* nlprob, const T_SerialDenseVector& rhs, int numconstraints) {
            return new NonLinearEquation(nlprob, rhs, numconstraints);
          },
          py::return_value_policy::reference, py::arg("nlprob"), py::arg("rhs"),
          py::arg("numconstraints") = 1
      )
      .def_static(
          "create",
          [](NLP* nlprob, int numconstraints) {
            return new NonLinearEquation(nlprob, numconstraints);
          },
          py::return_value_policy::reference, py::arg("nlprob"), py::arg("numconstraints") = 1
      );

  // Bind OptConstrQNewton (constrained Quasi-Newton)
  py::class_<OptConstrQNewton>(m, "OptConstrQNewton")
      .def(
          py::init([](NLF1* p) {
            NLP1* p_base = static_cast<NLP1*>(p);
            return new OptConstrQNewton(p_base, &default_update_model);
          }),
          py::arg("p"), py::keep_alive<0, 1>()
      )
      .def("setDebug", &OPTPP::OptimizeClass::setDebug)
      .def("optimize", &OptConstrQNewton::optimize)
      .def("printStatus", &OptConstrQNewton::printStatus, py::arg("s"))
      .def("cleanup", &OptConstrQNewton::cleanup)
      .def("setSearchStrategy", &OptConstrQNewton::setSearchStrategy, py::arg("s"))
      .def(
          "setOutputFile",
          static_cast<int (OptConstrQNewton::*)(const char*, int)>(
              &OptConstrQNewton::setOutputFile
          ),
          py::arg("filename"), py::arg("mode") = 0
      )
      .def("setTRSize", &OptConstrQNewton::setTRSize, py::arg("size"));

  // Bind OptQNIPS (Quasi-Newton Interior-Point Solver)
  py::class_<OptQNIPS>(m, "OptQNIPS")
      .def(
          py::init([](NLF1* p) {
            NLP1* p_base = static_cast<NLP1*>(p);
            return new OptQNIPS(p_base, &default_update_model);
          }),
          py::arg("p"), py::keep_alive<0, 1>()
      )
      .def("setDebug", &OPTPP::OptimizeClass::setDebug)
      .def("optimize", &OptQNIPS::optimize)
      .def("printStatus", &OptQNIPS::printStatus, py::arg("s"))
      .def("cleanup", &OptQNIPS::cleanup)
      .def("setSearchStrategy", &OptQNIPS::setSearchStrategy, py::arg("s"))
      .def(
          "setOutputFile",
          static_cast<int (OptQNIPS::*)(const char*, int)>(&OptQNIPS::setOutputFile),
          py::arg("filename"), py::arg("mode") = 0
      )
      .def("setTRSize", &OptQNIPS::setTRSize, py::arg("size"))
      .def("getTRSize", &OptQNIPS::getTRSize)
      .def("setMu", &OptQNIPS::setMu, py::arg("mu"))
      .def("getMu", &OptQNIPS::getMu)
      .def("setCenteringParameter", &OptQNIPS::setCenteringParameter, py::arg("sigma"))
      .def("setStepLengthToBdry", &OptQNIPS::setStepLengthToBdry, py::arg("tau"))
      .def("setMeritFcn", &OptQNIPS::setMeritFcn, py::arg("merit_function"))
      .def("getMeritFcn", &OptQNIPS::getMeritFcn)
      .def("setMaxIter", &OptQNIPS::setMaxIter, py::arg("max_iterations"))
      .def("setMaxFeval", &OptQNIPS::setMaxFeval, py::arg("max_function_evaluations"))
      .def("setFcnTol", &OptQNIPS::setFcnTol, py::arg("convergence_tolerance"))
      .def("setGradTol", &OptQNIPS::setGradTol, py::arg("gradient_tolerance"))
      .def("setConTol", &OptQNIPS::setConTol, py::arg("constraint_tolerance"))
      .def("setGradMult", &OptQNIPS::setGradMult, py::arg("gradient_multiplier"))
      .def("getGradMult", &OptQNIPS::getGradMult)
      .def("setSearchSize", &OptQNIPS::setSearchSize, py::arg("search_pattern_size"))
      .def("getSearchSize", &OptQNIPS::getSearchSize);
}
