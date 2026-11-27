#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "BoundConstraint.h"
#include "CompoundConstraint.h"
#include "Constraint.h"
#include "ConstraintBase.h"
#include "LinearEquation.h"
#include "LinearInequality.h"
#include "NLF.h"
#include "NLP.h"
#include "NonLinearConstraint.h"
#include "Opt.h"
#include "OptBCQNewton.h"
#include "OptQNIPS.h"
#include "OptQNewton.h"
#include "Teuchos_SerialDenseMatrix.hpp"
#include "Teuchos_SerialDenseVector.hpp"

namespace py = pybind11;
using namespace OPTPP;

namespace {

using T_SerialDenseVector = Teuchos::SerialDenseVector<int, double>;
using T_SerialDenseMatrix = Teuchos::SerialDenseMatrix<int, double>;

void initialize(int, T_SerialDenseVector&) {}

struct Evaluation {
  std::vector<double> values;
  std::vector<double> gradients;
};

std::vector<double> to_vector(py::handle data, py::ssize_t size) {
  auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(data);
  if (!array || array.size() != size) {
    throw std::runtime_error("The evaluate callback returned an array of the wrong size");
  }
  return {array.data(), array.data() + size};
}

Evaluation call_evaluate_callback(
    const py::function& evaluate_callback, const T_SerialDenseVector& x, int mode,
    int num_constraints
) {
  py::gil_scoped_acquire gil;
  py::array_t<double> point(x.length());
  std::memcpy(point.mutable_data(), x.values(), x.length() * sizeof(double));
  py::tuple result = evaluate_callback(point, mode).cast<py::tuple>();
  Evaluation evaluation;
  if (mode & NLPFunction) {
    evaluation.values = to_vector(result[0], num_constraints + 1);
  }
  if (mode & NLPGradient) {
    evaluation.gradients = to_vector(result[1], (num_constraints + 1) * x.length());
  }
  return evaluation;
}

// OPT++ passes a context pointer to objective callbacks, so the stock NLF1
// evaluation methods, including their caching and counting, are used as is.
class ObjectiveNLF1 : public NLF1 {
private:
  py::function evaluate_callback;
  int num_constraints;

  static void evaluate_objective(
      int mode, int num_variables, const T_SerialDenseVector& x, double& value,
      T_SerialDenseVector& gradient, int& result, void* context
  ) {
    auto* self = static_cast<ObjectiveNLF1*>(static_cast<NLF1*>(context));
    const Evaluation evaluation =
        call_evaluate_callback(self->evaluate_callback, x, mode, self->num_constraints);
    if (mode & NLPFunction) {
      value = evaluation.values[0];
    }
    if (mode & NLPGradient) {
      std::memcpy(gradient.values(), evaluation.gradients.data(), num_variables * sizeof(double));
    }
    result = mode;
  }

public:
  ObjectiveNLF1(
      int num_variables, int num_constraints, py::function evaluate, const T_SerialDenseVector& x0
  )
      : NLF1(num_variables, &ObjectiveNLF1::evaluate_objective, &initialize),
        evaluate_callback(std::move(evaluate)), num_constraints(num_constraints) {
    setX(x0);
  }
};

// OPT++ passes no context pointer to constraint callbacks, so the constraint
// evaluation methods are overridden, following the stock NLF1 methods.
class ConstraintNLF1 : public NLF1 {
private:
  py::function evaluate_callback;

  void extract_constraints(
      const Evaluation& evaluation, int mode, T_SerialDenseVector& values,
      T_SerialDenseMatrix& gradients
  ) const {
    for (int j = 0; j < ncnln; ++j) {
      if (mode & NLPFunction) {
        values(j) = evaluation.values[j + 1];
      }
      if (mode & NLPGradient) {
        for (int i = 0; i < dim; ++i) {
          gradients(i, j) = evaluation.gradients[(j + 1) * dim + i];
        }
      }
    }
  }

public:
  ConstraintNLF1(int num_variables, int num_constraints, py::function evaluate)
      : NLF1(num_variables, num_constraints, nullptr, &initialize),
        evaluate_callback(std::move(evaluate)) {}

  T_SerialDenseVector evalCF(const T_SerialDenseVector& x) override {
    T_SerialDenseVector values(ncnln);
    T_SerialDenseMatrix gradients(dim, ncnln);
    // Reuse constraint values cached for this x when available.
    if (!application.getCF(x, values)) {
      extract_constraints(
          call_evaluate_callback(evaluate_callback, x, NLPFunction, ncnln), NLPFunction, values,
          gradients
      );
      // Cache the computed values for later OPT++ evaluations at this x.
      application.constraint_update(NLPFunction, dim, ncnln, x, values, gradients);
    }
    return values;
  }

  T_SerialDenseMatrix evalCG(const T_SerialDenseVector& x) override {
    T_SerialDenseVector values(ncnln);
    T_SerialDenseMatrix gradients(dim, ncnln);
    // Reuse constraint gradients cached for this x when available.
    if (!application.getCGrad(x, gradients)) {
      extract_constraints(
          call_evaluate_callback(evaluate_callback, x, NLPGradient, ncnln), NLPGradient, values,
          gradients
      );
      // Cache the computed gradients for later OPT++ evaluations at this x.
      application.constraint_update(NLPGradient, dim, ncnln, x, values, gradients);
    }
    return gradients;
  }

  void evalC(const T_SerialDenseVector& x) override {
    const int mode = NLPFunction | NLPGradient;
    T_SerialDenseVector values(ncnln);
    T_SerialDenseMatrix gradients(dim, ncnln);
    // If either result is missing, evaluate and cache both for this x.
    if (!application.getCF(x, values) || !application.getCGrad(x, gradients)) {
      extract_constraints(
          call_evaluate_callback(evaluate_callback, x, mode, ncnln), mode, values, gradients
      );
      application.constraint_update(mode, dim, ncnln, x, values, gradients);
    }
  }
};

// Workaround for OPT++: NonLinearConstraint stores a raw NLP* and never deletes it.
// This subclass owns the NLP, so the NLP and its callback live as long as the constraint.
// The NLP exists before the base constructor runs, because that constructor calls getDim() on it.
class NonLinearConstraintWithOwnedNLP : public NonLinearConstraint {
private:
  std::unique_ptr<NLP> owned_nlp;

public:
  NonLinearConstraintWithOwnedNLP(
      std::unique_ptr<NLP> nlp, const T_SerialDenseVector& lower, const T_SerialDenseVector& upper,
      int num_equalities, int num_inequalities
  )
      : NonLinearConstraint(nlp.get(), lower, upper, num_equalities, num_inequalities),
        owned_nlp(std::move(nlp)) {}
};

} // namespace

PYBIND11_MODULE(_pyoptpp, m) {
  py::class_<T_SerialDenseVector>(m, "SerialDenseVector")
      .def(py::init([](py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
        auto vec = new T_SerialDenseVector(arr.size());
        std::memcpy(vec->values(), arr.data(), arr.size() * sizeof(double));
        return vec;
      }))
      .def("to_numpy", [](T_SerialDenseVector& v) {
        return py::array_t<double>(
            {static_cast<size_t>(v.length())}, {sizeof(double)}, v.values(), py::cast(v)
        );
      });
  py::class_<T_SerialDenseMatrix>(m, "SerialDenseMatrix")
      .def(py::init([](py::array_t<double, py::array::f_style | py::array::forcecast> arr) {
        if (arr.ndim() != 2) {
          throw std::runtime_error("Numpy array must be 2D");
        }
        auto mat = new T_SerialDenseMatrix(arr.shape(0), arr.shape(1));
        std::memcpy(mat->values(), arr.data(), arr.size() * sizeof(double));
        return mat;
      }));

  py::classh<NLF1>(m, "NLF1")
      .def("getXc", &NLF1::getXc)
      .def("setIsExpensive", &NLF1::setIsExpensive, py::arg("is_expensive"))
      .def("setModeOverride", &NLF1::setModeOverride, py::arg("override_mode"))
      .def(
          "setConstraints", [](NLF1& self, CompoundConstraint* cc) { self.setConstraints(cc); },
          py::keep_alive<1, 2>(), py::arg("compound_constraint")
      );
  py::classh<ObjectiveNLF1, NLF1>(m, "ObjectiveNLF1")
      .def(
          py::init<int, int, py::function, const T_SerialDenseVector&>(), py::arg("num_variables"),
          py::arg("num_constraints"), py::arg("evaluate"), py::arg("x0")
      );
  py::classh<ConstraintNLF1, NLF1>(m, "ConstraintNLF1")
      .def(
          py::init<int, int, py::function>(), py::arg("num_variables"), py::arg("num_constraints"),
          py::arg("evaluate")
      );

  py::classh<ConstraintBase>(m, "ConstraintBase");
  py::classh<CompoundConstraint>(m, "CompoundConstraint");
  py::classh<BoundConstraint, ConstraintBase>(m, "BoundConstraint")
      .def(
          py::init<int, const T_SerialDenseVector&, const T_SerialDenseVector&>(),
          py::arg("num_variables"), py::arg("lower"), py::arg("upper")
      );
  py::classh<LinearEquation, ConstraintBase>(m, "LinearEquation")
      .def(
          py::init<const T_SerialDenseMatrix&, const T_SerialDenseVector&>(), py::arg("A"),
          py::arg("rhs")
      );
  py::classh<LinearInequality, ConstraintBase>(m, "LinearInequality")
      .def(
          py::init<
              const T_SerialDenseMatrix&, const T_SerialDenseVector&, const T_SerialDenseVector&>(),
          py::arg("A"), py::arg("lower"), py::arg("upper")
      );
  py::classh<NonLinearConstraint, ConstraintBase>(m, "NonLinearConstraint")
      .def(
          py::init([](std::unique_ptr<ConstraintNLF1> nlf1, const T_SerialDenseVector& lower,
                      const T_SerialDenseVector& upper, int num_equalities, int num_inequalities) {
            return new NonLinearConstraintWithOwnedNLP(
                std::make_unique<NLP>(nlf1.release()), lower, upper, num_equalities,
                num_inequalities
            );
          }),
          py::arg("nlf1"), py::arg("lower"), py::arg("upper"), py::arg("num_equalities"),
          py::arg("num_inequalities")
      );
  m.def(
      "create_compound_constraint",
      [](std::vector<std::unique_ptr<ConstraintBase>> constraints) {
        OptppArray<Constraint> constraint_array{};
        constraint_array.reserve(constraints.size());
        for (auto& c : constraints) {
          // OPT++ crashes sorting a constraint without active rows (all bounds >= BIG_BND).
          if (c->getNumOfCons() == 0) {
            continue;
          }
          constraint_array.append(Constraint(c.release()));
        }
        return new CompoundConstraint(constraint_array);
      },
      py::arg("constraints")
  );

  py::enum_<SearchStrategy>(m, "SearchStrategy")
      .value("TrustRegion", SearchStrategy::TrustRegion)
      .value("LineSearch", SearchStrategy::LineSearch);
  py::enum_<MeritFcn>(m, "MeritFcn")
      .value("NormFmu", MeritFcn::NormFmu)
      .value("ArgaezTapia", MeritFcn::ArgaezTapia)
      .value("VanShanno", MeritFcn::VanShanno);

  // The optimizers keep a non-owning pointer to the NLF1, which the caller must keep alive.
  py::class_<OptQNewton>(m, "OptQNewton")
      .def(py::init([](NLF1* p) { return new OptQNewton(p); }), py::arg("p"))
      .def("cleanup", &OptQNewton::cleanup)
      .def("optimize", &OptQNewton::optimize)
      .def("getReturnCode", &OptQNewton::getReturnCode)
      .def("setDebug", &OptimizeClass::setDebug)
      .def(
          "setOutputFile",
          static_cast<int (OptQNewton::*)(const char*, int)>(&OptQNewton::setOutputFile),
          py::arg("filename"), py::arg("mode")
      )
      .def("setSearchStrategy", &OptQNewton::setSearchStrategy, py::arg("s"))
      .def("setTRSize", &OptQNewton::setTRSize, py::arg("size"))
      .def("setMaxStep", &OptQNewton::setMaxStep, py::arg("max_step"))
      .def("setMaxIter", &OptQNewton::setMaxIter, py::arg("max_iterations"))
      .def("setMaxFeval", &OptQNewton::setMaxFeval, py::arg("max_function_evaluations"))
      .def("setFcnTol", &OptQNewton::setFcnTol, py::arg("convergence_tolerance"))
      .def("setGradTol", &OptQNewton::setGradTol, py::arg("gradient_tolerance"));

  py::class_<OptBCQNewton>(m, "OptBCQNewton")
      .def(py::init([](NLF1* p) { return new OptBCQNewton(p); }), py::arg("p"))
      .def("cleanup", &OptBCQNewton::cleanup)
      .def("optimize", &OptBCQNewton::optimize)
      .def("getReturnCode", &OptBCQNewton::getReturnCode)
      .def("setDebug", &OptimizeClass::setDebug)
      .def(
          "setOutputFile",
          static_cast<int (OptBCQNewton::*)(const char*, int)>(&OptBCQNewton::setOutputFile),
          py::arg("filename"), py::arg("mode")
      )
      .def("setSearchStrategy", &OptBCQNewton::setSearchStrategy, py::arg("s"))
      .def("setTRSize", &OptBCQNewton::setTRSize, py::arg("size"))
      .def("setMaxStep", &OptBCQNewton::setMaxStep, py::arg("max_step"))
      .def("setMaxIter", &OptBCQNewton::setMaxIter, py::arg("max_iterations"))
      .def("setMaxFeval", &OptBCQNewton::setMaxFeval, py::arg("max_function_evaluations"))
      .def("setFcnTol", &OptBCQNewton::setFcnTol, py::arg("convergence_tolerance"))
      .def("setGradTol", &OptBCQNewton::setGradTol, py::arg("gradient_tolerance"));

  py::class_<OptQNIPS>(m, "OptQNIPS")
      .def(py::init([](NLF1* p) { return new OptQNIPS(p); }), py::arg("p"))
      .def("cleanup", &OptQNIPS::cleanup)
      .def("optimize", &OptQNIPS::optimize)
      .def("getReturnCode", &OptQNIPS::getReturnCode)
      .def("setDebug", &OptimizeClass::setDebug)
      .def(
          "setOutputFile",
          static_cast<int (OptQNIPS::*)(const char*, int)>(&OptQNIPS::setOutputFile),
          py::arg("filename"), py::arg("mode")
      )
      .def("setMaxStep", &OptQNIPS::setMaxStep, py::arg("max_step"))
      .def("setMaxIter", &OptQNIPS::setMaxIter, py::arg("max_iterations"))
      .def("setMaxFeval", &OptQNIPS::setMaxFeval, py::arg("max_function_evaluations"))
      .def("setFcnTol", &OptQNIPS::setFcnTol, py::arg("convergence_tolerance"))
      .def("setMeritFcn", &OptQNIPS::setMeritFcn, py::arg("merit_function"))
      .def("setCenteringParameter", &OptQNIPS::setCenteringParameter, py::arg("sigma"))
      .def("setStepLengthToBdry", &OptQNIPS::setStepLengthToBdry, py::arg("tau"));
}
