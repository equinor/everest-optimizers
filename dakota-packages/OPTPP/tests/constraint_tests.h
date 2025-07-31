#ifndef constraint_tests_h
#define constraint_tests_h

#include "CompoundConstraint.h"
#include "LinearEquation.h"
#include "LinearInequality.h"
#include "BoundConstraint.h"
#include "NonLinearEquation.h"
#include "NonLinearInequality.h"
#include "NLF.h"
#include "OptQNIPS.h"
#include "OptConstrQNewton.h"
#include "Teuchos_SerialDenseVector.hpp"
#include "Teuchos_SerialDenseMatrix.hpp"

using namespace OPTPP;
using Teuchos::SerialDenseVector;
using Teuchos::SerialDenseMatrix;

namespace ConstraintTests {

// Test result structure
struct TestResult {
    bool success;
    double final_objective;
    SerialDenseVector<int, double> final_point;
    double constraint_violation;
    int iterations;
    std::string message;
    
    TestResult() : success(false), final_objective(0.0), constraint_violation(0.0), iterations(0) {}
};

// ============================================================================
// LINEAR EQUALITY CONSTRAINT TESTS
// ============================================================================

// Test 1: Simple 2D quadratic with linear equality constraint
// minimize: (x-2)^2 + (y-1)^2
// subject to: x + y = 3
void init_linear_eq_test1(int n, SerialDenseVector<int,double>& x);
void linear_eq_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x, 
                         double& fx, SerialDenseVector<int,double>& g, int& result);
CompoundConstraint* create_linear_eq_test1_constraints(int n);
TestResult run_linear_eq_test1();

// Test 2: 3D quadratic with multiple linear equality constraints
// minimize: x^2 + y^2 + z^2
// subject to: x + y + z = 6, x - y = 1
void init_linear_eq_test2(int n, SerialDenseVector<int,double>& x);
void linear_eq_test2_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                         double& fx, SerialDenseVector<int,double>& g, int& result);
CompoundConstraint* create_linear_eq_test2_constraints(int n);
TestResult run_linear_eq_test2();

// ============================================================================
// LINEAR INEQUALITY CONSTRAINT TESTS
// ============================================================================

// Test 3: 2D quadratic with linear inequality constraint
// minimize: (x-3)^2 + (y-2)^2
// subject to: x + y <= 2
void init_linear_ineq_test1(int n, SerialDenseVector<int,double>& x);
void linear_ineq_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                           double& fx, SerialDenseVector<int,double>& g, int& result);
CompoundConstraint* create_linear_ineq_test1_constraints(int n);
TestResult run_linear_ineq_test1();

// Test 4: 3D quadratic with multiple linear inequality constraints
// minimize: x^2 + 2*y^2 + 3*z^2
// subject to: x + y <= 2, y + z <= 1, x + z <= 2
void init_linear_ineq_test2(int n, SerialDenseVector<int,double>& x);
void linear_ineq_test2_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                           double& fx, SerialDenseVector<int,double>& g, int& result);
CompoundConstraint* create_linear_ineq_test2_constraints(int n);
TestResult run_linear_ineq_test2();

// ============================================================================
// BOUND CONSTRAINT TESTS  
// ============================================================================

// Test 5: Simple bounds test
// minimize: (x-5)^2 + (y-5)^2
// subject to: 0 <= x,y <= 3
void init_bounds_test1(int n, SerialDenseVector<int,double>& x);
void bounds_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                      double& fx, SerialDenseVector<int,double>& g, int& result);
CompoundConstraint* create_bounds_test1_constraints(int n);
TestResult run_bounds_test1();

// Test 6: Asymmetric bounds test
// minimize: x^2 + 4*y^2 + 9*z^2
// subject to: -3 <= x <= 1, -2 <= y <= 0.5, -1 <= z <= 2
void init_bounds_test2(int n, SerialDenseVector<int,double>& x);
void bounds_test2_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                      double& fx, SerialDenseVector<int,double>& g, int& result);
CompoundConstraint* create_bounds_test2_constraints(int n);
TestResult run_bounds_test2();

// ============================================================================
// MIXED LINEAR CONSTRAINT TESTS
// ============================================================================

// Test 7: Mixed linear constraints
// minimize: x^2 + y^2 + z^2
// subject to: bounds: 0 <= x,y,z <= 5
//             equality: x + y + z = 3
//             inequality: x - y + z <= 2
void init_mixed_linear_test1(int n, SerialDenseVector<int,double>& x);
void mixed_linear_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                            double& fx, SerialDenseVector<int,double>& g, int& result);
CompoundConstraint* create_mixed_linear_test1_constraints(int n);
TestResult run_mixed_linear_test1();

// ============================================================================
// NONLINEAR CONSTRAINT TESTS
// ============================================================================

// Test 8: Nonlinear equality constraint
// minimize: x^2 + y^2
// subject to: x^2 + y^2 = 4 (circle constraint)
void init_nonlinear_eq_test1(int n, SerialDenseVector<int,double>& x);
void nonlinear_eq_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                            double& fx, SerialDenseVector<int,double>& g, int& result);
void nonlinear_eq_test1_con(int mode, int n, const SerialDenseVector<int,double>& x,
                            SerialDenseVector<int,double>& fx, SerialDenseMatrix<int,double>& g, int& result);
CompoundConstraint* create_nonlinear_eq_test1_constraints(int n);
TestResult run_nonlinear_eq_test1();

// Test 9: Nonlinear inequality constraint  
// minimize: (x-2)^2 + (y-2)^2
// subject to: x^2 + y^2 <= 1 (unit circle constraint)
void init_nonlinear_ineq_test1(int n, SerialDenseVector<int,double>& x);
void nonlinear_ineq_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                              double& fx, SerialDenseVector<int,double>& g, int& result);
void nonlinear_ineq_test1_con(int mode, int n, const SerialDenseVector<int,double>& x,
                              SerialDenseVector<int,double>& fx, SerialDenseMatrix<int,double>& g, int& result);
CompoundConstraint* create_nonlinear_ineq_test1_constraints(int n);
TestResult run_nonlinear_ineq_test1();

// Test 10: Mixed nonlinear constraints
// minimize: x^2 + y^2 + z^2
// subject to: bounds: -5 <= x,y,z <= 5
//             nonlinear equality: x^2 + y^2 = 1
//             nonlinear inequality: z^2 <= 4
void init_mixed_nonlinear_test1(int n, SerialDenseVector<int,double>& x);
void mixed_nonlinear_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                               double& fx, SerialDenseVector<int,double>& g, int& result);
void mixed_nonlinear_test1_con(int mode, int n, const SerialDenseVector<int,double>& x,
                               SerialDenseVector<int,double>& fx, SerialDenseMatrix<int,double>& g, int& result);
CompoundConstraint* create_mixed_nonlinear_test1_constraints(int n);
TestResult run_mixed_nonlinear_test1();

// ============================================================================
// DIAGNOSTIC TESTS FOR MULTIPLE LINEAR INEQUALITY CONSTRAINTS
// ============================================================================

// Diagnostic Test A: Same as failing test but with single constraint matrix
// minimize: x^2 + 2*y^2 + 3*z^2
// subject to: x + y <= 2, y + z <= 1, x + z <= 2 (combined in single matrix)
void init_diagnostic_single_matrix(int n, SerialDenseVector<int,double>& x);
void diagnostic_single_matrix_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                                  double& fx, SerialDenseVector<int,double>& g, int& result);
CompoundConstraint* create_diagnostic_single_matrix_constraints(int n);
TestResult run_diagnostic_single_matrix();

// Diagnostic Test B: Same as failing test but with separate CompoundConstraints
// minimize: x^2 + 2*y^2 + 3*z^2  
// subject to: x + y <= 2, y + z <= 1, x + z <= 2 (each in separate CompoundConstraint)
void init_diagnostic_separate_compounds(int n, SerialDenseVector<int,double>& x);
void diagnostic_separate_compounds_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                                       double& fx, SerialDenseVector<int,double>& g, int& result);
CompoundConstraint* create_diagnostic_separate_compounds_constraints(int n);
TestResult run_diagnostic_separate_compounds();

// Diagnostic Test C: Only first two constraints from failing test
// minimize: x^2 + 2*y^2 + 3*z^2
// subject to: x + y <= 2, y + z <= 1
void init_diagnostic_two_constraints(int n, SerialDenseVector<int,double>& x);
void diagnostic_two_constraints_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                                    double& fx, SerialDenseVector<int,double>& g, int& result);
CompoundConstraint* create_diagnostic_two_constraints_constraints(int n);
TestResult run_diagnostic_two_constraints();

// Diagnostic Test D: Only first constraint from failing test
// minimize: x^2 + 2*y^2 + 3*z^2
// subject to: x + y <= 2
void init_diagnostic_one_constraint(int n, SerialDenseVector<int,double>& x);
void diagnostic_one_constraint_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                                   double& fx, SerialDenseVector<int,double>& g, int& result);
CompoundConstraint* create_diagnostic_one_constraint_constraints(int n);
TestResult run_diagnostic_one_constraint();

// Diagnostic Test E: Different starting point for failing test
// minimize: x^2 + 2*y^2 + 3*z^2
// subject to: x + y <= 2, y + z <= 1, x + z <= 2 (different initial point)
void init_diagnostic_different_start(int n, SerialDenseVector<int,double>& x);
void diagnostic_different_start_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                                    double& fx, SerialDenseVector<int,double>& g, int& result);
CompoundConstraint* create_diagnostic_different_start_constraints(int n);
TestResult run_diagnostic_different_start();

// ============================================================================
// COMPREHENSIVE TEST RUNNER
// ============================================================================

// Run all constraint tests and return results
struct AllTestResults {
    TestResult linear_eq_test1;
    TestResult linear_eq_test2;
    TestResult linear_ineq_test1;
    TestResult linear_ineq_test2;
    TestResult bounds_test1;
    TestResult bounds_test2;
    TestResult mixed_linear_test1;
    TestResult nonlinear_eq_test1;
    TestResult nonlinear_ineq_test1;
    TestResult mixed_nonlinear_test1;
    
    // Diagnostic tests
    TestResult diagnostic_single_matrix;
    TestResult diagnostic_separate_compounds;
    TestResult diagnostic_two_constraints;
    TestResult diagnostic_one_constraint;
    TestResult diagnostic_different_start;
    
    int total_tests() const { return 15; }
    int passed_tests() const {
        int count = 0;
        if (linear_eq_test1.success) count++;
        if (linear_eq_test2.success) count++;
        if (linear_ineq_test1.success) count++;
        if (linear_ineq_test2.success) count++;
        if (bounds_test1.success) count++;
        if (bounds_test2.success) count++;
        if (mixed_linear_test1.success) count++;
        if (nonlinear_eq_test1.success) count++;
        if (nonlinear_ineq_test1.success) count++;
        if (mixed_nonlinear_test1.success) count++;
        if (diagnostic_single_matrix.success) count++;
        if (diagnostic_separate_compounds.success) count++;
        if (diagnostic_two_constraints.success) count++;
        if (diagnostic_one_constraint.success) count++;
        if (diagnostic_different_start.success) count++;
        return count;
    }
};

AllTestResults run_all_constraint_tests();

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Check constraint violation for a given point and constraint set
double check_constraint_violation(const SerialDenseVector<int,double>& x, CompoundConstraint* constraints);

// Pretty print test results
void print_test_result(const std::string& test_name, const TestResult& result);
void print_all_test_results(const AllTestResults& results);

} // namespace ConstraintTests

#endif