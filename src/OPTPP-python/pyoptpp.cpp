#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fmt/format.h>

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
#include "OptBCQNewton.h"
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

// Non-linear function class that holds Python callbacks
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
    } catch (py::error_already_set &e) {
      throw;
    } catch (const std::exception& e) {
      throw std::runtime_error(fmt::format("Error in evalF callback: {}", e.what()));
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
    } catch (py::error_already_set &e) {
      throw;
    } catch (const std::exception& e) {
      throw std::runtime_error(fmt::format("Error in evalG callback: {}", e.what()));
    }
  }

  T_SerialDenseVector evalG() override { return this->evalG(this->getXc()); }

  T_SerialSymDenseMatrix evalH(T_SerialDenseVector& x) override {
    // Return identity matrix - Hessian is not used in these algorithms
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
        throw std::runtime_error("evalCF: Returned numpy array must be 1D!");
      }

      T_SerialDenseVector constraints(buf.shape[0]);
      std::memcpy(constraints.values(), buf.ptr, buf.shape[0] * sizeof(double));
      return constraints;
    } catch (py::error_already_set &e) {
      throw;
    } catch (const std::exception& e) {
      throw std::runtime_error(fmt::format("Error in evalCF callback: {}", e.what()));
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
        throw std::runtime_error("evalCG: Returned numpy array must be 2D!");
      }

      T_SerialDenseMatrix grad(buf.shape[0], buf.shape[1]);
      std::memcpy(grad.values(), buf.ptr, buf.shape[0] * buf.shape[1] * sizeof(double));
      return grad;
    } catch (py::error_already_set &e) {
      throw;
    } catch (const std::exception& e) {
      throw std::runtime_error(fmt::format("Error in evalCG callback: {}", e.what()));
    }
  }
};

PYBIND11_MODULE(_pyoptpp, m) {
  m.doc() = "Python bindings for OPTPP library";

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
      .def("to_numpy", [](T_SerialDenseVector& v) {
        return py::array_t<double>({(size_t)v.length()}, {sizeof(double)}, v.values(), py::cast(v));
      });

  py::class_<Teuchos::SerialSymDenseMatrix<int, double>>(m, "SerialSymDenseMatrix");
  py::class_<T_SerialDenseMatrix>(m, "SerialDenseMatrix", py::buffer_protocol())
      .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
        if (arr.ndim() != 2) {
          throw std::runtime_error("Numpy array must be 2D");
        }
        auto mat = new T_SerialDenseMatrix(arr.shape(0), arr.shape(1));
        std::memcpy(mat->values(), arr.data(), arr.size() * sizeof(double));
        return mat;
      }));

  py::class_<NLPBase>(m, "NLPBase");
  py::class_<NLP>(m, "NLP").def_static(
      "create", [](NLPBase* nlp) { return new NLP(nlp); }, py::return_value_policy::reference,
      py::arg("nlp")
  );

  py::class_<NLF1, NLPBase>(m, "NLF1")
      .def_static(
          "create",
          [](int ndim, py::function eval_f, py::function eval_g,
             const T_SerialDenseVector& x0) -> NLF1* {
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
          [](int ndim, py::function eval_cf, py::function eval_cg,
             const T_SerialDenseVector& x0) -> NLF1* {
            // Create a CallbackNLF1 with dummy objective (not used for constraints)
            auto dummy_f = py::cpp_function([](const T_SerialDenseVector& x) { return 0.0; });
            auto dummy_g = py::cpp_function([ndim](const T_SerialDenseVector& x) {
              return py::array_t<double>(ndim);
            });

            CallbackNLF1* nlf1 = new CallbackNLF1(ndim, dummy_f, dummy_g, eval_cf, eval_cg);
            nlf1->setX(x0);
            nlf1->setIsExpensive(true);

            return static_cast<NLF1*>(nlf1);
          },
          py::return_value_policy::reference, py::arg("ndim"), py::arg("eval_cf"),
          py::arg("eval_cg"), py::arg("x0"),
          "Create a constraint NLF1 object with Python callbacks (C++-managed)"
      )
      .def("getXc", &NLF1::getXc)
      .def("getF", &NLF1::getF)
      .def(
          "setConstraints", [](NLF1& self, CompoundConstraint* cc) { self.setConstraints(cc); },
          py::arg("compound_constraint")
      );

  py::class_<Constraint>(m, "Constraint");
  py::class_<CompoundConstraint>(m, "CompoundConstraint");
  py::class_<ConstraintBase>(m, "ConstraintBase")
      .def_static("delete", [](ConstraintBase* constraint) { delete constraint; });

  py::class_<LinearConstraint, ConstraintBase>(m, "LinearConstraint");

  py::class_<LinearEquation, LinearConstraint>(m, "LinearEquation")
      .def_static(
          "create",
          [](const T_SerialDenseMatrix& A, const T_SerialDenseVector& rhs) {
            return new LinearEquation(A, rhs);
          },
          py::return_value_policy::reference, py::arg("A"), py::arg("rhs")
      );

  py::class_<LinearInequality, LinearConstraint>(m, "LinearInequality")
      .def_static(
          "create",
          [](const T_SerialDenseMatrix& A, const T_SerialDenseVector& lower,
             const T_SerialDenseVector& upper) { return new LinearInequality(A, lower, upper); },
          py::return_value_policy::reference, py::arg("A"), py::arg("lower"), py::arg("upper")
      );

  py::class_<BoundConstraint, ConstraintBase>(m, "BoundConstraint")
      .def_static(
          "create",
          [](int nc, const T_SerialDenseVector& A, const T_SerialDenseVector& rhs) {
            return new BoundConstraint(nc, A, rhs);
          },
          py::return_value_policy::reference, py::arg("nc"), py::arg("A"), py::arg("rhs")
      );

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

  py::class_<NonLinearConstraint, ConstraintBase>(m, "NonLinearConstraint");

  py::class_<NonLinearInequality, NonLinearConstraint>(m, "NonLinearInequality")
      .def_static(
          "create",
          [](NLP* nlprob, const T_SerialDenseVector& lower, const T_SerialDenseVector& upper,
             int numconstraints) {
            return new NonLinearInequality(nlprob, lower, upper, numconstraints);
          },
          py::return_value_policy::reference, py::arg("nlprob"), py::arg("lower"), py::arg("upper"),
          py::arg("numconstraints") = 1
      );

  py::class_<NonLinearEquation, NonLinearConstraint>(m, "NonLinearEquation")
      .def_static(
          "create",
          [](NLP* nlprob, const T_SerialDenseVector& rhs, int numconstraints) {
            return new NonLinearEquation(nlprob, rhs, numconstraints);
          },
          py::return_value_policy::reference, py::arg("nlprob"), py::arg("rhs"),
          py::arg("numconstraints") = 1
      );

  py::enum_<SearchStrategy>(m, "SearchStrategy")
      .value("TrustRegion", SearchStrategy::TrustRegion)
      .value("LineSearch", SearchStrategy::LineSearch)
      .value("TrustPDS", SearchStrategy::TrustPDS)
      .export_values();

  py::enum_<MeritFcn>(m, "MeritFcn")
      .value("NormFmu", MeritFcn::NormFmu)
      .value("ArgaezTapia", MeritFcn::ArgaezTapia)
      .value("VanShanno", MeritFcn::VanShanno)
      .export_values();

  py::class_<OptQNewton>(m, "OptQNewton")
      .def(
          py::init([](NLF1* p) {
            NLP1* p_base = static_cast<NLP1*>(p);
            return new OptQNewton(p_base, &default_update_model);
          }),
          py::arg("p"), py::keep_alive<0, 1>()
      )
      .def("cleanup", &OptQNewton::cleanup)
      .def("optimize", &OptQNewton::optimize)
      .def("setDebug", &OPTPP::OptimizeClass::setDebug, "Set debug flag to true")
      .def(
          "setOutputFile",
          static_cast<int (OptQNewton::*)(const char*, int)>(&OptQNewton::setOutputFile),
          py::arg("filename"), py::arg("mode") = 0
      )
      .def("setSearchStrategy", &OptQNewton::setSearchStrategy, py::arg("s"))
      .def("setSearchSize", &OptQNewton::setSearchSize, py::arg("search_pattern_size"))
      .def("setTRSize", &OptQNewton::setTRSize, py::arg("size"))
      .def("setGradMult", &OptQNewton::setGradMult, py::arg("gradient_multiplier"))
      .def("setMaxIter", &OptQNewton::setMaxIter, py::arg("max_iterations"))
      .def("setMaxFeval", &OptQNewton::setMaxFeval, py::arg("max_function_evaluations"))
      .def("setFcnTol", &OptQNewton::setFcnTol, py::arg("convergence_tolerance"))
      .def("setGradTol", &OptQNewton::setGradTol, py::arg("gradient_tolerance"));

      py::class_<OptBCQNewton>(m, "OptBCQNewton")
      .def(
        py::init([](NLF1* p) {
            NLP1* p_base = static_cast<NLP1*>(p);
            return new OptBCQNewton(p_base, &default_update_model);
          }),
          py::arg("p"), py::keep_alive<0, 1>()
      )
      .def("cleanup", &OptBCQNewton::cleanup)
      .def("optimize", &OptBCQNewton::optimize)
      .def("setDebug", &OPTPP::OptimizeClass::setDebug, "Set debug flag to true")
      .def(
        "setOutputFile",
        static_cast<int (OptBCQNewton::*)(const char*, int)>(&OptBCQNewton::setOutputFile),
        py::arg("filename"), py::arg("mode") = 0
      )
      .def("setSearchStrategy", &OptBCQNewton::setSearchStrategy, py::arg("s"))
      .def("setSearchSize", &OptBCQNewton::setSearchSize, py::arg("search_pattern_size"))
      .def("setTRSize", &OptBCQNewton::setTRSize, py::arg("size"))
      .def("setGradMult", &OptBCQNewton::setGradMult, py::arg("gradient_multiplier"))
      .def("setMaxIter", &OptBCQNewton::setMaxIter, py::arg("max_iterations"))
      .def("setMaxFeval", &OptBCQNewton::setMaxFeval, py::arg("max_function_evaluations"))
      .def("setFcnTol", &OptBCQNewton::setFcnTol, py::arg("convergence_tolerance"))
      .def("setGradTol", &OptBCQNewton::setGradTol, py::arg("gradient_tolerance"));

  py::class_<OptQNIPS>(m, "OptQNIPS")
      .def(
          py::init([](NLF1* p) {
            NLP1* p_base = static_cast<NLP1*>(p);
            return new OptQNIPS(p_base, &default_update_model);
          }),
          py::arg("p"), py::keep_alive<0, 1>()
      )
      .def("cleanup", &OptQNIPS::cleanup)
      .def("optimize", &OptQNIPS::optimize)
      .def("setDebug", &OPTPP::OptimizeClass::setDebug)
      .def(
        "setOutputFile",
        static_cast<int (OptQNIPS::*)(const char*, int)>(&OptQNIPS::setOutputFile),
        py::arg("filename"), py::arg("mode") = 0
      )
      .def("setSearchStrategy", &OptQNIPS::setSearchStrategy, py::arg("s"))
      .def("setSearchSize", &OptQNIPS::setSearchSize, py::arg("search_pattern_size"))
      .def("setTRSize", &OptQNIPS::setTRSize, py::arg("size"))
      .def("setGradMult", &OptQNIPS::setGradMult, py::arg("gradient_multiplier"))
      .def("setMaxIter", &OptQNIPS::setMaxIter, py::arg("max_iterations"))
      .def("setMaxFeval", &OptQNIPS::setMaxFeval, py::arg("max_function_evaluations"))
      .def("setFcnTol", &OptQNIPS::setFcnTol, py::arg("convergence_tolerance"))
      .def("setGradTol", &OptQNIPS::setGradTol, py::arg("gradient_tolerance"))
      .def("setConTol", &OptQNIPS::setConTol, py::arg("constraint_tolerance"))
      .def("setMeritFcn", &OptQNIPS::setMeritFcn, py::arg("merit_function"))
      .def("setCenteringParameter", &OptQNIPS::setCenteringParameter, py::arg("sigma"))
      .def("setStepLengthToBdry", &OptQNIPS::setStepLengthToBdry, py::arg("tau"))
      .def("setMu", &OptQNIPS::setMu, py::arg("mu"));
}
