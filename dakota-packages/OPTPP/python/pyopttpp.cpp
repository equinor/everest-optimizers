#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "NLF.h"
#include "Opt.h"
#include "OptQNewton.h"
#include "OptConstrQNewton.h"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_SerialSymDenseMatrix.hpp"
#include "BoundConstraint.h"
#include "Constraint.h"
#include "CompoundConstraint.h"
#include "LinearEquation.h"
#include "LinearInequality.h"
#include "NonLinearConstraint.h"
#include "NonLinearInequality.h"
#include "NonLinearEquation.h"

#include <functional>

namespace py = pybind11;
using namespace OPTPP;

using T_SerialDenseVector = Teuchos::SerialDenseVector<int, double>;
using T_SerialSymDenseMatrix = Teuchos::SerialSymDenseMatrix<int, double>;

// Dummy C++ function for the default case
void default_update_model(int, int, T_SerialDenseVector) {}

// Trampoline class for NonLinearInequality - enables Python-defined constraint functions
class PyNonLinearInequality : public NonLinearInequality {
public:
  using NonLinearInequality::NonLinearInequality;
};

// Trampoline class for NLF1
class PyNLF1 : public NLF1 {
public:
  // Inherit constructors
  using NLF1::NLF1;

  // Override virtual functions
  void initFcn() override {
    py::gil_scoped_acquire gil;
    py::function override = py::get_override(static_cast<const NLF1*>(this), "initFcn");
    if (override) {
      override();
    }
    // If no override is provided in Python, we do nothing.
    // The base NLF1::initFcn() is flawed for our use case as it calls a null function pointer.
  }

  // This eval() hides the non-virtual NLF1::eval() in the base class.
  // The optimizer calls this method. We then call the virtual methods
  // which are correctly routed to the Python overrides.
  void eval() {
    this->fvalue = this->evalF(this->mem_xc);
    this->mem_grad = this->evalG(this->mem_xc);
    this->nfevals++;
    this->ngevals++;
  }

  real evalF(const T_SerialDenseVector& x) override {
    py::gil_scoped_acquire gil;
    py::function override = py::get_override(static_cast<const NLF1*>(this), "evalF");
    if (override) {
      return override(x).cast<real>();
    }
    // If no override, return 0.0. The base class method is flawed.
    return 0.0;
  }

  T_SerialDenseVector evalG(const T_SerialDenseVector& x) override {
    py::gil_scoped_acquire gil;
    py::function override = py::get_override(static_cast<const NLF1*>(this), "evalG");
    if (override) {
      py::object result = override(x);
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
    }
    // If no override, return an empty vector of the correct size.
    // The base class method is flawed.
    return T_SerialDenseVector(this->getDim());
  }

  T_SerialDenseVector evalG() override {
    // This override is crucial. The base NLF1::evalG() calls evalG(mem_xc),
    // but if the virtual method override has the wrong signature (e.g. const),
    // the base implementation of evalG(x) gets called, which crashes.
    // By overriding this, we ensure our safe evalG(x) is always called.
    return this->evalG(this->getXc());
  }

  T_SerialSymDenseMatrix evalH(T_SerialDenseVector& x) override {
    py::gil_scoped_acquire gil;
    py::function override = py::get_override(static_cast<const NLF1*>(this), "evalH");
    if (override) {
      py::print("Warning: evalH Python override is not fully implemented in the wrapper.");
    }
    // If no override, return an empty matrix of the correct size.
    // If no Python override, return the identity matrix
    int dim = this->getDim();
    T_SerialSymDenseMatrix H(dim);
    for (int i = 0; i < dim; ++i) {
      H(i, i) = 1.0;
    }
    return H;
  }
};

PYBIND11_MODULE(pyopttpp, m) {
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
  py::class_<Teuchos::SerialDenseMatrix<int, double>>(m, "SerialDenseMatrix", py::buffer_protocol())
      .def(py::init<int, int>())
      .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
        if (arr.ndim() != 2) {
            throw std::runtime_error("Numpy array must be 2-D");
        }
        auto mat = new Teuchos::SerialDenseMatrix<int, double>(arr.shape(0), arr.shape(1));
        std::memcpy(mat->values(), arr.data(), arr.size() * sizeof(double));
        return mat;
      }))
      .def_buffer([](Teuchos::SerialDenseMatrix<int, double>& m) -> py::buffer_info {
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
      .def("amIFeasible", [](Constraint& self, const T_SerialDenseVector& x, double epsilon = 1e-6) {
          return self.amIFeasible(x, epsilon);
      }, py::arg("x"), py::arg("epsilon") = 1e-6);

  // Bind CompoundConstraint
  py::class_<CompoundConstraint>(m, "CompoundConstraint")
      .def(py::init<const OptppArray<Constraint>&>())
      .def("getNumOfCons", &CompoundConstraint::getNumOfCons)
      .def("getNumOfVars", &CompoundConstraint::getNumOfVars)
      .def("evalResidual", &CompoundConstraint::evalResidual, py::arg("x"))
      .def("amIFeasible", [](CompoundConstraint& self, const T_SerialDenseVector& x, double epsilon = 1e-6) {
          return self.amIFeasible(x, epsilon);
      }, py::arg("x"), py::arg("epsilon") = 1e-6);

  // Bind NLF1 using the trampoline class
  py::class_<NLF1, PyNLF1>(m, "NLF1")
      .def(py::init<int>(), py::arg("ndim"))
      .def("initFcn", &NLF1::initFcn)
      .def(
          "evalF", static_cast<real (NLF1::*)(const T_SerialDenseVector&)>(&NLF1::evalF),
          py::arg("x")
      )
      .def(
          "evalG",
          static_cast<T_SerialDenseVector (NLF1::*)(const T_SerialDenseVector&)>(&NLF1::evalG),
          py::arg("x")
      )
      .def(
          "evalH",
          static_cast<T_SerialSymDenseMatrix (NLF1::*)(T_SerialDenseVector&)>(&NLF1::evalH),
          py::arg("x")
      )
      .def("getXc", &NLF1::getXc)
      .def("getF", &NLF1::getF)
      .def("setX", static_cast<void (NLF1::*)(const T_SerialDenseVector&)>(&NLF1::setX))
      .def("setIsExpensive", &NLF1::setIsExpensive, py::arg("is_expensive"))
       .def("setConstraints", [](NLF1& self, CompoundConstraint* cc){ self.setConstraints(cc); }, py::arg("compound_constraint"));

  // Bind SearchStrategy enum
  py::enum_<SearchStrategy>(m, "SearchStrategy")
      .value("TrustRegion", SearchStrategy::TrustRegion)
      .value("LineSearch", SearchStrategy::LineSearch)
      .value("TrustPDS", SearchStrategy::TrustPDS)
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
  py::class_<ConstraintBase>(m, "ConstraintBase");

  // Bind LinearConstraint as it's a base for Equation and Inequality
  py::class_<LinearConstraint, ConstraintBase>(m, "LinearConstraint");

  // Bind LinearEquation
  py::class_<LinearEquation, LinearConstraint>(m, "LinearEquation")
      .def(py::init<const Teuchos::SerialDenseMatrix<int,double>&, const T_SerialDenseVector&>(),
           py::arg("A"), py::arg("rhs"))
      .def("getB", &LinearEquation::getB)
      .def("evalAx", &LinearEquation::evalAx, py::arg("x"));

  // Bind LinearInequality
  py::class_<LinearInequality, LinearConstraint>(m, "LinearInequality")
      .def(py::init<const Teuchos::SerialDenseMatrix<int,double>&, const T_SerialDenseVector&>(),
           py::arg("A"), py::arg("rhs"));

  // Bind BoundConstraint
  py::class_<BoundConstraint, ConstraintBase>(m, "BoundConstraint")
      .def(py::init<int, const T_SerialDenseVector&, const T_SerialDenseVector&>(),
           py::arg("nvar"), py::arg("lower"), py::arg("upper"));

  // Helper to create CompoundConstraint from bound constraints only
  m.def("create_compound_constraint", [](py::array_t<double, py::array::c_style | py::array::forcecast> lower,
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
   }, py::return_value_policy::reference, py::arg("lower"), py::arg("upper"));

  // Helper to create a Constraint object from a variety of constraint types
  m.def("create_constraint", [](py::object constraint_obj) {
      if (py::isinstance<LinearEquation>(constraint_obj)) {
          auto* le = constraint_obj.cast<LinearEquation*>();
          return new Constraint(le);
      } else if (py::isinstance<LinearInequality>(constraint_obj)) {
          auto* li = constraint_obj.cast<LinearInequality*>();
          return new Constraint(li);
      } else if (py::isinstance<BoundConstraint>(constraint_obj)) {
          auto* bc = constraint_obj.cast<BoundConstraint*>();
          return new Constraint(bc);
      } else if (py::isinstance<NonLinearInequality>(constraint_obj)) {
          auto* nli = constraint_obj.cast<NonLinearInequality*>();
          return new Constraint(nli);
      } else if (py::isinstance<NonLinearEquation>(constraint_obj)) {
          auto* nle = constraint_obj.cast<NonLinearEquation*>();
          return new Constraint(nle);
      }
      throw std::runtime_error("Unknown constraint type");
  }, py::return_value_policy::reference);

  // Helper to create CompoundConstraint from a list of constraints
  m.def("create_compound_constraint", [](py::list constraints) {
      OptppArray<Constraint> constraint_array(constraints.size());
      int i = 0;
      for (auto& c : constraints) {
          if (py::isinstance<LinearEquation>(c)) {
              constraint_array[i] = Constraint(c.cast<LinearEquation*>());
          } else if (py::isinstance<LinearInequality>(c)) {
              constraint_array[i] = Constraint(c.cast<LinearInequality*>());
          } else if (py::isinstance<BoundConstraint>(c)) {
              constraint_array[i] = Constraint(c.cast<BoundConstraint*>());
          } else if (py::isinstance<NonLinearInequality>(c)) {
              constraint_array[i] = Constraint(c.cast<NonLinearInequality*>());
          } else if (py::isinstance<NonLinearEquation>(c)) {
              constraint_array[i] = Constraint(c.cast<NonLinearEquation*>());
          } else {
              throw std::runtime_error("Unknown constraint type in list");
          }
          i++;
      }
      return new CompoundConstraint(constraint_array);
  }, py::return_value_policy::reference, py::arg("constraints"));

  // Bind NonLinearConstraint as base class
  py::class_<NonLinearConstraint, ConstraintBase>(m, "NonLinearConstraint");

  // Bind NonLinearInequality
  py::class_<NonLinearInequality, NonLinearConstraint>(m, "NonLinearInequality")
      .def(py::init<NLP*, int>(), py::arg("nlp"), py::arg("numconstraints") = 1)
      .def(py::init<NLP*, const T_SerialDenseVector&, int>(), 
           py::arg("nlp"), py::arg("rhs"), py::arg("numconstraints") = 1)
      .def(py::init<NLP*, const bool, int>(), 
           py::arg("nlp"), py::arg("flag"), py::arg("numconstraints") = 1)
      .def(py::init<NLP*, const T_SerialDenseVector&, const T_SerialDenseVector&, int>(),
           py::arg("nlp"), py::arg("lower"), py::arg("upper"), py::arg("numconstraints") = 1);

  // Bind NonLinearEquation
  py::class_<NonLinearEquation, NonLinearConstraint>(m, "NonLinearEquation")
      .def(py::init<NLP*, int>(), py::arg("nlp"), py::arg("numconstraints") = 1)
      .def(py::init<NLP*, const T_SerialDenseVector&, int>(), 
           py::arg("nlp"), py::arg("rhs"), py::arg("numconstraints") = 1);

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
          static_cast<int (OptConstrQNewton::*)(const char*, int)>(&OptConstrQNewton::setOutputFile),
          py::arg("filename"), py::arg("mode") = 0
      )
      .def("setTRSize", &OptConstrQNewton::setTRSize, py::arg("size"));
}
