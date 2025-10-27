#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "constraint_tests.h"

#include <functional>

PYBIND11_MODULE(pyopttpp, m) {

  // ============================================================================
  // CONSTRAINT TESTING FRAMEWORK
  // ============================================================================

  // Bind TestResult structure
  py::class_<ConstraintTests::TestResult>(m, "TestResult")
      .def(py::init<>())
      .def_readwrite("success", &ConstraintTests::TestResult::success)
      .def_readwrite("final_objective", &ConstraintTests::TestResult::final_objective)
      .def_readwrite("final_point", &ConstraintTests::TestResult::final_point)
      .def_readwrite("constraint_violation", &ConstraintTests::TestResult::constraint_violation)
      .def_readwrite("iterations", &ConstraintTests::TestResult::iterations)
      .def_readwrite("message", &ConstraintTests::TestResult::message);

  // Bind AllTestResults structure
  py::class_<ConstraintTests::AllTestResults>(m, "AllTestResults")
      .def(py::init<>())
      .def_readwrite("linear_eq_test1", &ConstraintTests::AllTestResults::linear_eq_test1)
      .def_readwrite("linear_eq_test2", &ConstraintTests::AllTestResults::linear_eq_test2)
      .def_readwrite("linear_ineq_test1", &ConstraintTests::AllTestResults::linear_ineq_test1)
      .def_readwrite("linear_ineq_test2", &ConstraintTests::AllTestResults::linear_ineq_test2)
      .def_readwrite("bounds_test1", &ConstraintTests::AllTestResults::bounds_test1)
      .def_readwrite("bounds_test2", &ConstraintTests::AllTestResults::bounds_test2)
      .def_readwrite("mixed_linear_test1", &ConstraintTests::AllTestResults::mixed_linear_test1)
      .def_readwrite("nonlinear_eq_test1", &ConstraintTests::AllTestResults::nonlinear_eq_test1)
      .def_readwrite("nonlinear_ineq_test1", &ConstraintTests::AllTestResults::nonlinear_ineq_test1)
      .def_readwrite(
          "mixed_nonlinear_test1", &ConstraintTests::AllTestResults::mixed_nonlinear_test1
      )
      .def_readwrite(
          "diagnostic_single_matrix", &ConstraintTests::AllTestResults::diagnostic_single_matrix
      )
      .def_readwrite(
          "diagnostic_separate_compounds",
          &ConstraintTests::AllTestResults::diagnostic_separate_compounds
      )
      .def_readwrite(
          "diagnostic_two_constraints", &ConstraintTests::AllTestResults::diagnostic_two_constraints
      )
      .def_readwrite(
          "diagnostic_one_constraint", &ConstraintTests::AllTestResults::diagnostic_one_constraint
      )
      .def_readwrite(
          "diagnostic_different_start", &ConstraintTests::AllTestResults::diagnostic_different_start
      )
      .def_readwrite(
          "diagnostic_skip_first", &ConstraintTests::AllTestResults::diagnostic_skip_first
      )
      .def_readwrite(
          "diagnostic_simple_bounds", &ConstraintTests::AllTestResults::diagnostic_simple_bounds
      )
      .def_readwrite(
          "diagnostic_detailed_eval", &ConstraintTests::AllTestResults::diagnostic_detailed_eval
      )
      .def_readwrite(
          "diagnostic_manual_evaluation",
          &ConstraintTests::AllTestResults::diagnostic_manual_evaluation
      )
      .def("total_tests", &ConstraintTests::AllTestResults::total_tests)
      .def("passed_tests", &ConstraintTests::AllTestResults::passed_tests);

  m.doc() = "Python bindings for OPTPP library";
  // ============================================================================
  // INDIVIDUAL CONSTRAINT TEST FUNCTIONS
  // ============================================================================

  // Linear Equality Constraint Tests
  m.def(
      "run_linear_eq_test1", &ConstraintTests::run_linear_eq_test1,
      "Test 1: Simple 2D quadratic with linear equality constraint x + y = 3"
  );
  m.def(
      "run_linear_eq_test2", &ConstraintTests::run_linear_eq_test2,
      "Test 2: 3D quadratic with multiple linear equality constraints"
  );

  // Linear Inequality Constraint Tests
  m.def(
      "run_linear_ineq_test1", &ConstraintTests::run_linear_ineq_test1,
      "Test 3: 2D quadratic with linear inequality constraint x + y <= 2"
  );
  m.def(
      "run_linear_ineq_test2", &ConstraintTests::run_linear_ineq_test2,
      "Test 4: 3D quadratic with multiple linear inequality constraints"
  );

  // Bound Constraint Tests
  m.def(
      "run_bounds_test1", &ConstraintTests::run_bounds_test1,
      "Test 5: Simple bounds test with constrained optimum"
  );
  m.def(
      "run_bounds_test2", &ConstraintTests::run_bounds_test2,
      "Test 6: Asymmetric bounds test with unconstrained optimum within "
      "bounds"
  );

  // Mixed Linear Constraint Tests
  m.def(
      "run_mixed_linear_test1", &ConstraintTests::run_mixed_linear_test1,
      "Test 7: Mixed linear constraints (bounds + equality + inequality)"
  );

  // Nonlinear Constraint Tests
  m.def(
      "run_nonlinear_eq_test1", &ConstraintTests::run_nonlinear_eq_test1,
      "Test 8: Nonlinear equality constraint x^2 + y^2 = 4"
  );
  m.def(
      "run_nonlinear_ineq_test1", &ConstraintTests::run_nonlinear_ineq_test1,
      "Test 9: Nonlinear inequality constraint x^2 + y^2 <= 1"
  );
  m.def(
      "run_mixed_nonlinear_test1", &ConstraintTests::run_mixed_nonlinear_test1,
      "Test 10: Mixed nonlinear constraints (bounds + nonlinear equality + "
      "inequality)"
  );

  // ============================================================================
  // DIAGNOSTIC TESTS FOR MULTIPLE LINEAR INEQUALITY CONSTRAINTS
  // ============================================================================
  m.def(
      "run_diagnostic_single_matrix", &ConstraintTests::run_diagnostic_single_matrix,
      "Diagnostic A: Same as failing test but with single constraint matrix"
  );
  m.def(
      "run_diagnostic_separate_compounds", &ConstraintTests::run_diagnostic_separate_compounds,
      "Diagnostic B: Same as failing test but with separate "
      "CompoundConstraints"
  );
  m.def(
      "run_diagnostic_two_constraints", &ConstraintTests::run_diagnostic_two_constraints,
      "Diagnostic C: Only first two constraints from failing test"
  );
  m.def(
      "run_diagnostic_one_constraint", &ConstraintTests::run_diagnostic_one_constraint,
      "Diagnostic D: Only first constraint from failing test"
  );
  m.def(
      "run_diagnostic_different_start", &ConstraintTests::run_diagnostic_different_start,
      "Diagnostic E: Different starting point for failing test"
  );
  m.def(
      "run_diagnostic_skip_first", &ConstraintTests::run_diagnostic_skip_first,
      "Diagnostic F: Skip first constraint from failing test"
  );
  m.def(
      "run_diagnostic_simple_bounds", &ConstraintTests::run_diagnostic_simple_bounds,
      "Diagnostic G: Simple individual bounds as inequalities"
  );
  m.def(
      "run_diagnostic_detailed_eval", &ConstraintTests::run_diagnostic_detailed_eval,
      "Diagnostic H: Detailed constraint evaluation behavior"
  );
  m.def(
      "run_diagnostic_manual_evaluation", &ConstraintTests::run_diagnostic_manual_evaluation,
      "Diagnostic I: Manual constraint evaluation at specific points"
  );

  // ============================================================================
  // COMPREHENSIVE TEST RUNNER
  // ============================================================================

  m.def(
      "run_all_constraint_tests", &ConstraintTests::run_all_constraint_tests,
      "Run all constraint tests and return comprehensive results"
  );

  // ============================================================================
  // UTILITY FUNCTIONS
  // ============================================================================

  m.def(
      "check_constraint_violation", &ConstraintTests::check_constraint_violation,
      "Check constraint violation for a given point and constraint set", py::arg("x"),
      py::arg("constraints")
  );

  m.def(
      "print_test_result", &ConstraintTests::print_test_result, "Pretty print a single test result",
      py::arg("test_name"), py::arg("result")
  );

  m.def(
      "print_all_test_results", &ConstraintTests::print_all_test_results,
      "Pretty print all test results with summary", py::arg("results")
  );
}
