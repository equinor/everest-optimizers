#include "constraint_tests.h"
#include <iostream>
#include <iomanip>
#include <cmath>

namespace ConstraintTests {

// ============================================================================
// LINEAR EQUALITY CONSTRAINT TESTS
// ============================================================================

// Test 1: Simple 2D quadratic with linear equality constraint
// minimize: (x-2)^2 + (y-1)^2  
// subject to: x + y = 3
// Expected solution: x=2.5, y=0.5, f=0.5

void init_linear_eq_test1(int n, SerialDenseVector<int,double>& x) {
    if (n != 2) return;
    x(0) = 0.0;
    x(1) = 0.0;
}

void linear_eq_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                         double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 2) return;
    
    double x1 = x(0);
    double x2 = x(1);
    
    if (mode & NLPFunction) {
        fx = (x1 - 2.0) * (x1 - 2.0) + (x2 - 1.0) * (x2 - 1.0);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * (x1 - 2.0);
        g(1) = 2.0 * (x2 - 1.0);
        result = NLPGradient;
    }
}

CompoundConstraint* create_linear_eq_test1_constraints(int n) {
    if (n != 2) return nullptr;
    
    // Create linear equality constraint: x + y = 3
    SerialDenseMatrix<int,double> A(1, 2);
    A(0, 0) = 1.0;  // coefficient for x
    A(0, 1) = 1.0;  // coefficient for y
    
    SerialDenseVector<int,double> b(1);
    b(0) = 3.0;     // right-hand side
    
    LinearEquation* eq = new LinearEquation(A, b);
    
    // Create compound constraint
    OptppArray<Constraint> constraints(1);
    constraints[0] = Constraint(eq);
    
    return new CompoundConstraint(constraints);
}

TestResult run_linear_eq_test1() {
    TestResult result;
    
    try {
        // Create NLF1 problem
        NLF1 nlp(2, linear_eq_test1_obj, init_linear_eq_test1, create_linear_eq_test1_constraints);
        
        // Create optimizer
        OptQNIPS optimizer(&nlp);
        optimizer.setMaxIter(200);
        optimizer.setFcnTol(1.0e-6);
        optimizer.setGradTol(1.0e-6);
        optimizer.setConTol(1.0e-8);
        optimizer.setMeritFcn(ArgaezTapia);
        optimizer.setSearchStrategy(TrustRegion);
        
        // Solve
        optimizer.optimize();
        
        // Extract results
        result.final_point = nlp.getXc();
        result.final_objective = nlp.getF();
        result.constraint_violation = check_constraint_violation(result.final_point, 
                                                               create_linear_eq_test1_constraints(2));
        result.iterations = optimizer.getIter();
        
        // Check success conditions
        // Analytical solution: minimize (x-2)^2 + (y-1)^2 subject to x + y = 3
        // Using Lagrange multipliers: optimal point is [2, 1] with objective = 0
        double expected_x1 = 2.0, expected_x2 = 1.0, expected_f = 0.0;
        double tol = 1e-3;
        
        bool point_ok = (std::abs(result.final_point(0) - expected_x1) < tol) &&
                        (std::abs(result.final_point(1) - expected_x2) < tol);
        bool obj_ok = std::abs(result.final_objective - expected_f) < tol;
        bool constraint_ok = result.constraint_violation < 1e-6;
        
        result.success = point_ok && obj_ok && constraint_ok;
        result.message = result.success ? "PASSED" : "FAILED - tolerance check";
        
    } catch (const std::exception& e) {
        result.success = false;
        result.message = std::string("FAILED - exception: ") + e.what();
    } catch (...) {
        result.success = false;
        result.message = "FAILED - unknown exception";
    }
    
    return result;
}

// Test 2: 3D quadratic with multiple linear equality constraints
// minimize: x^2 + y^2 + z^2
// subject to: x + y + z = 6, x - y = 1
// Expected solution: x=2.5, y=1.5, z=2, f=12.5

void init_linear_eq_test2(int n, SerialDenseVector<int,double>& x) {
    if (n != 3) return;
    x(0) = 1.0;
    x(1) = 1.0; 
    x(2) = 1.0;
}

void linear_eq_test2_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                         double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 3) return;
    
    if (mode & NLPFunction) {
        fx = x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * x(0);
        g(1) = 2.0 * x(1);
        g(2) = 2.0 * x(2);
        result = NLPGradient;
    }
}

CompoundConstraint* create_linear_eq_test2_constraints(int n) {
    if (n != 3) return nullptr;
    
    // Create first linear equality constraint: x + y + z = 6
    SerialDenseMatrix<int,double> A1(1, 3);
    A1(0, 0) = 1.0; A1(0, 1) = 1.0; A1(0, 2) = 1.0;
    SerialDenseVector<int,double> b1(1);
    b1(0) = 6.0;
    LinearEquation* eq1 = new LinearEquation(A1, b1);
    
    // Create second linear equality constraint: x - y = 1
    SerialDenseMatrix<int,double> A2(1, 3);
    A2(0, 0) = 1.0; A2(0, 1) = -1.0; A2(0, 2) = 0.0;
    SerialDenseVector<int,double> b2(1);
    b2(0) = 1.0;
    LinearEquation* eq2 = new LinearEquation(A2, b2);
    
    // Create compound constraint
    OptppArray<Constraint> constraints(2);
    constraints[0] = Constraint(eq1);
    constraints[1] = Constraint(eq2);
    
    return new CompoundConstraint(constraints);
}

TestResult run_linear_eq_test2() {
    TestResult result;
    
    try {
        NLF1 nlp(3, linear_eq_test2_obj, init_linear_eq_test2, create_linear_eq_test2_constraints);
        
        OptQNIPS optimizer(&nlp);
        optimizer.setMaxIter(200);
        optimizer.setFcnTol(1.0e-6);
        optimizer.setGradTol(1.0e-6);
        optimizer.setConTol(1.0e-8);
        optimizer.setMeritFcn(ArgaezTapia);
        optimizer.setSearchStrategy(TrustRegion);
        
        optimizer.optimize();
        
        result.final_point = nlp.getXc();
        result.final_objective = nlp.getF();
        result.constraint_violation = check_constraint_violation(result.final_point,
                                                               create_linear_eq_test2_constraints(3));
        result.iterations = optimizer.getIter();
        
        // Expected: x=2.5, y=1.5, z=2, f=12.5
        double tol = 1e-2;
        bool constraint_ok = result.constraint_violation < 1e-5;
        bool obj_reasonable = result.final_objective > 10.0 && result.final_objective < 20.0;
        
        result.success = constraint_ok && obj_reasonable;
        result.message = result.success ? "PASSED" : "FAILED - constraint or objective check";
        
    } catch (...) {
        result.success = false;
        result.message = "FAILED - exception";
    }
    
    return result;
}

// ============================================================================
// LINEAR INEQUALITY CONSTRAINT TESTS  
// ============================================================================

// Test 3: 2D quadratic with linear inequality constraint
// minimize: (x-3)^2 + (y-2)^2
// subject to: x + y <= 2
// Expected solution: on boundary x + y = 2, around x=1.5, y=0.5

void init_linear_ineq_test1(int n, SerialDenseVector<int,double>& x) {
    if (n != 2) return;
    x(0) = 0.0;
    x(1) = 0.0;
}

void linear_ineq_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                           double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 2) return;
    
    if (mode & NLPFunction) {
        fx = (x(0) - 3.0) * (x(0) - 3.0) + (x(1) - 2.0) * (x(1) - 2.0);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * (x(0) - 3.0);
        g(1) = 2.0 * (x(1) - 2.0);
        result = NLPGradient;
    }
}

CompoundConstraint* create_linear_ineq_test1_constraints(int n) {
    if (n != 2) return nullptr;
    
    // Create linear inequality constraint: x + y <= 2
    // OPTPP uses Ax >= b convention, so for x + y <= 2, we need -x - y >= -2
    SerialDenseMatrix<int,double> A(1, 2);
    A(0, 0) = -1.0;  // -x
    A(0, 1) = -1.0;  // -y
    
    SerialDenseVector<int,double> b(1);
    b(0) = -2.0;     // >= -2, which means x + y <= 2
    
    LinearInequality* ineq = new LinearInequality(A, b);
    
    OptppArray<Constraint> constraints(1);
    constraints[0] = Constraint(ineq);
    
    return new CompoundConstraint(constraints);
}

TestResult run_linear_ineq_test1() {
    TestResult result;
    
    try {
        NLF1 nlp(2, linear_ineq_test1_obj, init_linear_ineq_test1, create_linear_ineq_test1_constraints);
        
        OptQNIPS optimizer(&nlp);
        optimizer.setMaxIter(200);
        optimizer.setFcnTol(1.0e-6);
        optimizer.setGradTol(1.0e-6);
        optimizer.setConTol(1.0e-8);
        optimizer.setMeritFcn(ArgaezTapia);
        optimizer.setSearchStrategy(TrustRegion);
        
        optimizer.optimize();
        
        result.final_point = nlp.getXc();
        result.final_objective = nlp.getF();
        result.constraint_violation = check_constraint_violation(result.final_point,
                                                               create_linear_ineq_test1_constraints(2));
        result.iterations = optimizer.getIter();
        
        // Check that constraint is satisfied: x + y <= 2
        double constraint_value = result.final_point(0) + result.final_point(1);
        bool constraint_ok = constraint_value <= 2.0 + 1e-5;
        bool obj_reasonable = result.final_objective > 0.0 && result.final_objective < 10.0;
        
        // For debugging: be more lenient if optimizer didn't respect constraints
        if (!constraint_ok) {
            result.success = false;
            result.message = "FAILED - constraint not respected by optimizer";
        } else {
            result.success = constraint_ok && obj_reasonable;
            result.message = result.success ? "PASSED" : "FAILED - unreasonable objective";
        }
        
    } catch (...) {
        result.success = false;
        result.message = "FAILED - exception";
    }
    
    return result;
}

// Test 4: 3D quadratic with multiple linear inequality constraints
void init_linear_ineq_test2(int n, SerialDenseVector<int,double>& x) {
    if (n != 3) return;
    x(0) = 0.5;
    x(1) = 0.5;
    x(2) = 0.5;
}

void linear_ineq_test2_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                           double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 3) return;
    
    if (mode & NLPFunction) {
        fx = x(0)*x(0) + 2.0*x(1)*x(1) + 3.0*x(2)*x(2);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * x(0);
        g(1) = 4.0 * x(1);
        g(2) = 6.0 * x(2);
        result = NLPGradient;
    }
}

CompoundConstraint* create_linear_ineq_test2_constraints(int n) {
    if (n != 3) return nullptr;
    
    // Multiple linear inequalities: x + y <= 2, y + z <= 1, x + z <= 2
    // OPTPP uses Ax >= b, so convert: x + y <= 2 becomes -x - y >= -2
    SerialDenseMatrix<int,double> A1(1, 3);
    A1(0, 0) = -1.0; A1(0, 1) = -1.0; A1(0, 2) = 0.0;  // -x - y >= -2
    SerialDenseVector<int,double> b1(1);
    b1(0) = -2.0;
    LinearInequality* ineq1 = new LinearInequality(A1, b1);
    
    SerialDenseMatrix<int,double> A2(1, 3);
    A2(0, 0) = 0.0; A2(0, 1) = -1.0; A2(0, 2) = -1.0;  // -y - z >= -1
    SerialDenseVector<int,double> b2(1);
    b2(0) = -1.0;
    LinearInequality* ineq2 = new LinearInequality(A2, b2);
    
    SerialDenseMatrix<int,double> A3(1, 3);
    A3(0, 0) = -1.0; A3(0, 1) = 0.0; A3(0, 2) = -1.0;  // -x - z >= -2
    SerialDenseVector<int,double> b3(1);
    b3(0) = -2.0;
    LinearInequality* ineq3 = new LinearInequality(A3, b3);
    
    OptppArray<Constraint> constraints(3);
    constraints[0] = Constraint(ineq1);
    constraints[1] = Constraint(ineq2);
    constraints[2] = Constraint(ineq3);
    
    return new CompoundConstraint(constraints);
}

TestResult run_linear_ineq_test2() {
    TestResult result;
    
    try {
        NLF1 nlp(3, linear_ineq_test2_obj, init_linear_ineq_test2, create_linear_ineq_test2_constraints);
        
        OptQNIPS optimizer(&nlp);
        optimizer.setMaxIter(300);
        optimizer.setFcnTol(1.0e-6);
        optimizer.setGradTol(1.0e-6);
        optimizer.setConTol(1.0e-7);
        optimizer.setMeritFcn(ArgaezTapia);
        optimizer.setSearchStrategy(TrustRegion);
        
        optimizer.optimize();
        
        result.final_point = nlp.getXc();
        result.final_objective = nlp.getF();
        result.constraint_violation = check_constraint_violation(result.final_point,
                                                               create_linear_ineq_test2_constraints(3));
        result.iterations = optimizer.getIter();
        
        result.success = result.constraint_violation < 1e-5;
        result.message = result.success ? "PASSED" : "FAILED - constraint violation";
        
    } catch (...) {
        result.success = false;
        result.message = "FAILED - exception";
    }
    
    return result;
}

// ============================================================================
// BOUND CONSTRAINT TESTS
// ============================================================================

// Test 5: Simple bounds test
void init_bounds_test1(int n, SerialDenseVector<int,double>& x) {
    if (n != 2) return;
    x(0) = 1.0;
    x(1) = 1.0;
}

void bounds_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                      double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 2) return;
    
    if (mode & NLPFunction) {
        fx = (x(0) - 5.0) * (x(0) - 5.0) + (x(1) - 5.0) * (x(1) - 5.0);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * (x(0) - 5.0);
        g(1) = 2.0 * (x(1) - 5.0);
        result = NLPGradient;
    }
}

CompoundConstraint* create_bounds_test1_constraints(int n) {
    if (n != 2) return nullptr;
    
    SerialDenseVector<int,double> lower(2);
    lower(0) = 0.0; lower(1) = 0.0;
    
    SerialDenseVector<int,double> upper(2);
    upper(0) = 3.0; upper(1) = 3.0;
    
    BoundConstraint* bounds = new BoundConstraint(2, lower, upper);
    
    OptppArray<Constraint> constraints(1);
    constraints[0] = Constraint(bounds);
    
    return new CompoundConstraint(constraints);
}

TestResult run_bounds_test1() {
    TestResult result;
    
    try {
        NLF1 nlp(2, bounds_test1_obj, init_bounds_test1, create_bounds_test1_constraints);
        
        OptQNIPS optimizer(&nlp);
        optimizer.setMaxIter(200);
        optimizer.setFcnTol(1.0e-6);
        optimizer.setGradTol(1.0e-6);
        optimizer.setConTol(1.0e-8);
        optimizer.setMeritFcn(ArgaezTapia);
        optimizer.setSearchStrategy(TrustRegion);
        
        optimizer.optimize();
        
        result.final_point = nlp.getXc();
        result.final_objective = nlp.getF();
        result.iterations = optimizer.getIter();
        
        // Check bounds satisfaction
        bool bounds_ok = (result.final_point(0) >= -1e-6 && result.final_point(0) <= 3.0 + 1e-6) &&
                         (result.final_point(1) >= -1e-6 && result.final_point(1) <= 3.0 + 1e-6);
        
        // Expected solution: [3, 3] (at boundary)
        double expected_x = 3.0, expected_y = 3.0;
        bool solution_ok = (std::abs(result.final_point(0) - expected_x) < 1e-2) &&
                           (std::abs(result.final_point(1) - expected_y) < 1e-2);
        
        result.success = bounds_ok && solution_ok;
        result.message = result.success ? "PASSED" : "FAILED - bounds or solution check";
        
    } catch (...) {
        result.success = false;
        result.message = "FAILED - exception";
    }
    
    return result;
}

// Test 6: Asymmetric bounds test  
void init_bounds_test2(int n, SerialDenseVector<int,double>& x) {
    if (n != 3) return;
    x(0) = 0.0;
    x(1) = 0.0;
    x(2) = 0.0;
}

void bounds_test2_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                      double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 3) return;
    
    if (mode & NLPFunction) {
        fx = x(0)*x(0) + 4.0*x(1)*x(1) + 9.0*x(2)*x(2);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * x(0);
        g(1) = 8.0 * x(1);
        g(2) = 18.0 * x(2);
        result = NLPGradient;
    }
}

CompoundConstraint* create_bounds_test2_constraints(int n) {
    if (n != 3) return nullptr;
    
    SerialDenseVector<int,double> lower(3);
    lower(0) = -3.0; lower(1) = -2.0; lower(2) = -1.0;
    
    SerialDenseVector<int,double> upper(3);
    upper(0) = 1.0; upper(1) = 0.5; upper(2) = 2.0;
    
    BoundConstraint* bounds = new BoundConstraint(3, lower, upper);
    
    OptppArray<Constraint> constraints(1);
    constraints[0] = Constraint(bounds);
    
    return new CompoundConstraint(constraints);
}

TestResult run_bounds_test2() {
    TestResult result;
    
    try {
        NLF1 nlp(3, bounds_test2_obj, init_bounds_test2, create_bounds_test2_constraints);
        
        OptQNIPS optimizer(&nlp);
        optimizer.setMaxIter(200);
        optimizer.setFcnTol(1.0e-6);
        optimizer.setGradTol(1.0e-6);
        optimizer.setConTol(1.0e-8);
        optimizer.setMeritFcn(ArgaezTapia);
        optimizer.setSearchStrategy(TrustRegion);
        
        optimizer.optimize();
        
        result.final_point = nlp.getXc();
        result.final_objective = nlp.getF();
        result.iterations = optimizer.getIter();
        
        // Check bounds satisfaction
        bool bounds_ok = (result.final_point(0) >= -3.0 - 1e-6 && result.final_point(0) <= 1.0 + 1e-6) &&
                         (result.final_point(1) >= -2.0 - 1e-6 && result.final_point(1) <= 0.5 + 1e-6) &&
                         (result.final_point(2) >= -1.0 - 1e-6 && result.final_point(2) <= 2.0 + 1e-6);
        
        // Expected solution: [0, 0, 0] (unconstrained minimum within bounds)
        bool solution_ok = (std::abs(result.final_point(0)) < 1e-3) &&
                           (std::abs(result.final_point(1)) < 1e-3) &&
                           (std::abs(result.final_point(2)) < 1e-3);
        
        result.success = bounds_ok && solution_ok;
        result.message = result.success ? "PASSED" : "FAILED - bounds or solution check";
        
    } catch (...) {
        result.success = false;
        result.message = "FAILED - exception";
    }
    
    return result;
}

// ============================================================================
// MIXED LINEAR CONSTRAINT TESTS
// ============================================================================

void init_mixed_linear_test1(int n, SerialDenseVector<int,double>& x) {
    if (n != 3) return;
    x(0) = 2.0;
    x(1) = 1.0;
    x(2) = 0.5;
}

void mixed_linear_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                            double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 3) return;
    
    if (mode & NLPFunction) {
        fx = x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * x(0);
        g(1) = 2.0 * x(1);
        g(2) = 2.0 * x(2);
        result = NLPGradient;
    }
}

CompoundConstraint* create_mixed_linear_test1_constraints(int n) {
    if (n != 3) return nullptr;
    
    OptppArray<Constraint> constraints(3);
    
    // Bounds: 0 <= x,y,z <= 5
    SerialDenseVector<int,double> lower(3);
    lower(0) = 0.0; lower(1) = 0.0; lower(2) = 0.0;
    SerialDenseVector<int,double> upper(3);
    upper(0) = 5.0; upper(1) = 5.0; upper(2) = 5.0;
    BoundConstraint* bounds = new BoundConstraint(3, lower, upper);
    constraints[0] = Constraint(bounds);
    
    // Equality: x + y + z = 3
    SerialDenseMatrix<int,double> A_eq(1, 3);
    A_eq(0, 0) = 1.0; A_eq(0, 1) = 1.0; A_eq(0, 2) = 1.0;
    SerialDenseVector<int,double> b_eq(1);
    b_eq(0) = 3.0;
    LinearEquation* eq = new LinearEquation(A_eq, b_eq);
    constraints[1] = Constraint(eq);
    
    // Inequality: x - y + z <= 2
    // OPTPP uses Ax >= b, so convert: x - y + z <= 2 becomes -x + y - z >= -2
    SerialDenseMatrix<int,double> A_ineq(1, 3);
    A_ineq(0, 0) = -1.0; A_ineq(0, 1) = 1.0; A_ineq(0, 2) = -1.0;  // -x + y - z >= -2
    SerialDenseVector<int,double> b_ineq(1);
    b_ineq(0) = -2.0;
    LinearInequality* ineq = new LinearInequality(A_ineq, b_ineq);
    constraints[2] = Constraint(ineq);
    
    return new CompoundConstraint(constraints);
}

TestResult run_mixed_linear_test1() {
    TestResult result;
    
    try {
        NLF1 nlp(3, mixed_linear_test1_obj, init_mixed_linear_test1, create_mixed_linear_test1_constraints);
        
        OptQNIPS optimizer(&nlp);
        optimizer.setMaxIter(300);
        optimizer.setFcnTol(1.0e-6);
        optimizer.setGradTol(1.0e-6);
        optimizer.setConTol(1.0e-7);
        optimizer.setMeritFcn(ArgaezTapia);
        optimizer.setSearchStrategy(TrustRegion);
        
        optimizer.optimize();
        
        result.final_point = nlp.getXc();
        result.final_objective = nlp.getF();
        result.constraint_violation = check_constraint_violation(result.final_point,
                                                               create_mixed_linear_test1_constraints(3));
        result.iterations = optimizer.getIter();
        
        // Check all constraints
        // Bounds: 0 <= x,y,z <= 5
        bool bounds_ok = (result.final_point(0) >= -1e-5 && result.final_point(0) <= 5.0 + 1e-5) &&
                         (result.final_point(1) >= -1e-5 && result.final_point(1) <= 5.0 + 1e-5) &&
                         (result.final_point(2) >= -1e-5 && result.final_point(2) <= 5.0 + 1e-5);
        
        // Equality: x + y + z = 3
        double eq_val = result.final_point(0) + result.final_point(1) + result.final_point(2);
        bool eq_ok = std::abs(eq_val - 3.0) < 1e-4;
        
        // Inequality: x - y + z <= 2
        double ineq_val = result.final_point(0) - result.final_point(1) + result.final_point(2);
        bool ineq_ok = ineq_val <= 2.0 + 1e-4;
        
        result.success = bounds_ok && eq_ok && ineq_ok;
        result.message = result.success ? "PASSED" : "FAILED - constraint check";
        
    } catch (...) {
        result.success = false;
        result.message = "FAILED - exception";
    }
    
    return result;
}

// ============================================================================
// NONLINEAR CONSTRAINT TESTS (Placeholder implementations)
// ============================================================================

// Note: Nonlinear constraints require more complex setup with NLP objects
// and specialized constraint function signatures. For now, we provide
// placeholder implementations that return "not implemented" results.

void init_nonlinear_eq_test1(int n, SerialDenseVector<int,double>& x) {
    if (n != 2) return;
    x(0) = 1.5;
    x(1) = 1.5;
}

void nonlinear_eq_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                            double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 2) return;
    
    if (mode & NLPFunction) {
        fx = x(0)*x(0) + x(1)*x(1);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * x(0);
        g(1) = 2.0 * x(1);
        result = NLPGradient;
    }
}

void nonlinear_eq_test1_con(int mode, int n, const SerialDenseVector<int,double>& x,
                            SerialDenseVector<int,double>& fx, SerialDenseMatrix<int,double>& g, int& result) {
    // Constraint: x^2 + y^2 = 4
    if (mode & NLPFunction) {
        fx(0) = x(0)*x(0) + x(1)*x(1) - 4.0;
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0, 0) = 2.0 * x(0);
        g(1, 0) = 2.0 * x(1);
        result = NLPGradient;
    }
}

CompoundConstraint* create_nonlinear_eq_test1_constraints(int n) {
    // Nonlinear constraints require specialized NLP setup
    // This is a placeholder - full implementation would need NLP objects
    return nullptr;
}

TestResult run_nonlinear_eq_test1() {
    TestResult result;
    result.success = false;
    result.message = "NOT IMPLEMENTED - nonlinear constraints require specialized NLP setup";
    return result;
}

// Similar placeholder implementations for other nonlinear tests
void init_nonlinear_ineq_test1(int n, SerialDenseVector<int,double>& x) {
    if (n != 2) return;
    x(0) = 0.0;
    x(1) = 0.0;
}

void nonlinear_ineq_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                              double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 2) return;
    
    if (mode & NLPFunction) {
        fx = (x(0) - 2.0) * (x(0) - 2.0) + (x(1) - 2.0) * (x(1) - 2.0);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * (x(0) - 2.0);
        g(1) = 2.0 * (x(1) - 2.0);
        result = NLPGradient;
    }
}

void nonlinear_ineq_test1_con(int mode, int n, const SerialDenseVector<int,double>& x,
                              SerialDenseVector<int,double>& fx, SerialDenseMatrix<int,double>& g, int& result) {
    // Constraint: x^2 + y^2 <= 1
    if (mode & NLPFunction) {
        fx(0) = 1.0 - x(0)*x(0) - x(1)*x(1);  // Convert <= to >= 0 form
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0, 0) = -2.0 * x(0);
        g(1, 0) = -2.0 * x(1);
        result = NLPGradient;
    }
}

CompoundConstraint* create_nonlinear_ineq_test1_constraints(int n) {
    return nullptr;  // Placeholder
}

TestResult run_nonlinear_ineq_test1() {
    TestResult result;
    result.success = false;
    result.message = "NOT IMPLEMENTED - nonlinear constraints require specialized NLP setup";
    return result;
}

void init_mixed_nonlinear_test1(int n, SerialDenseVector<int,double>& x) {
    if (n != 3) return;
    x(0) = 1.0;
    x(1) = 0.0;
    x(2) = 0.0;
}

void mixed_nonlinear_test1_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                               double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 3) return;
    
    if (mode & NLPFunction) {
        fx = x(0)*x(0) + x(1)*x(1) + x(2)*x(2);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * x(0);
        g(1) = 2.0 * x(1);
        g(2) = 2.0 * x(2);
        result = NLPGradient;
    }
}

void mixed_nonlinear_test1_con(int mode, int n, const SerialDenseVector<int,double>& x,
                               SerialDenseVector<int,double>& fx, SerialDenseMatrix<int,double>& g, int& result) {
    // Multiple nonlinear constraints
    if (mode & NLPFunction) {
        fx(0) = x(0)*x(0) + x(1)*x(1) - 1.0;  // x^2 + y^2 = 1
        fx(1) = 4.0 - x(2)*x(2);              // z^2 <= 4
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0, 0) = 2.0 * x(0); g(1, 0) = 2.0 * x(1); g(2, 0) = 0.0;
        g(0, 1) = 0.0;        g(1, 1) = 0.0;        g(2, 1) = -2.0 * x(2);
        result = NLPGradient;
    }
}

CompoundConstraint* create_mixed_nonlinear_test1_constraints(int n) {
    return nullptr;  // Placeholder
}

TestResult run_mixed_nonlinear_test1() {
    TestResult result;
    result.success = false;
    result.message = "NOT IMPLEMENTED - mixed nonlinear constraints require specialized NLP setup";
    return result;
}

// ============================================================================
// DIAGNOSTIC TESTS FOR MULTIPLE LINEAR INEQUALITY CONSTRAINTS
// ============================================================================

// Diagnostic Test A: Same as failing test but with single constraint matrix
void init_diagnostic_single_matrix(int n, SerialDenseVector<int,double>& x) {
    if (n != 3) return;
    x(0) = 0.5;
    x(1) = 0.5;
    x(2) = 0.5;
}

void diagnostic_single_matrix_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                                  double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 3) return;
    
    if (mode & NLPFunction) {
        fx = x(0)*x(0) + 2.0*x(1)*x(1) + 3.0*x(2)*x(2);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * x(0);
        g(1) = 4.0 * x(1);
        g(2) = 6.0 * x(2);
        result = NLPGradient;
    }
}

CompoundConstraint* create_diagnostic_single_matrix_constraints(int n) {
    if (n != 3) return nullptr;
    
    // All three constraints in single matrix: x + y <= 2, y + z <= 1, x + z <= 2
    // OPTPP uses Ax >= b, so convert to: -x - y >= -2, -y - z >= -1, -x - z >= -2
    SerialDenseMatrix<int,double> A(3, 3);
    A(0, 0) = -1.0; A(0, 1) = -1.0; A(0, 2) = 0.0;  // -x - y >= -2
    A(1, 0) = 0.0;  A(1, 1) = -1.0; A(1, 2) = -1.0; // -y - z >= -1
    A(2, 0) = -1.0; A(2, 1) = 0.0;  A(2, 2) = -1.0; // -x - z >= -2
    
    SerialDenseVector<int,double> b(3);
    b(0) = -2.0;
    b(1) = -1.0;
    b(2) = -2.0;
    
    LinearInequality* ineq = new LinearInequality(A, b);
    
    OptppArray<Constraint> constraints(1);
    constraints[0] = Constraint(ineq);
    
    return new CompoundConstraint(constraints);
}

TestResult run_diagnostic_single_matrix() {
    TestResult result;
    
    try {
        NLF1 nlp(3, diagnostic_single_matrix_obj, init_diagnostic_single_matrix, create_diagnostic_single_matrix_constraints);
        
        OptQNIPS optimizer(&nlp);
        optimizer.setMaxIter(300);
        optimizer.setFcnTol(1.0e-6);
        optimizer.setGradTol(1.0e-6);
        optimizer.setConTol(1.0e-7);
        optimizer.setMeritFcn(ArgaezTapia);
        optimizer.setSearchStrategy(TrustRegion);
        
        optimizer.optimize();
        
        result.final_point = nlp.getXc();
        result.final_objective = nlp.getF();
        result.constraint_violation = check_constraint_violation(result.final_point,
                                                               create_diagnostic_single_matrix_constraints(3));
        result.iterations = optimizer.getIter();
        
        result.success = result.constraint_violation < 1e-5;
        result.message = result.success ? "PASSED" : "FAILED - constraint violation";
        
    } catch (...) {
        result.success = false;
        result.message = "FAILED - exception";
    }
    
    return result;
}

// Diagnostic Test B: Same as failing test but with separate CompoundConstraints  
void init_diagnostic_separate_compounds(int n, SerialDenseVector<int,double>& x) {
    if (n != 3) return;
    x(0) = 0.5;
    x(1) = 0.5;
    x(2) = 0.5;
}

void diagnostic_separate_compounds_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                                       double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 3) return;
    
    if (mode & NLPFunction) {
        fx = x(0)*x(0) + 2.0*x(1)*x(1) + 3.0*x(2)*x(2);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * x(0);
        g(1) = 4.0 * x(1);
        g(2) = 6.0 * x(2);
        result = NLPGradient;
    }
}

CompoundConstraint* create_diagnostic_separate_compounds_constraints(int n) {
    // NOTE: This is not actually possible with OPTPP - you can't have multiple
    // CompoundConstraints. This test will demonstrate the limitation.
    // For now, we'll return the same as the original failing test
    return nullptr;  // Placeholder - not implementable
}

TestResult run_diagnostic_separate_compounds() {
    TestResult result;
    result.success = false;
    result.message = "NOT IMPLEMENTED - OPTPP doesn't support multiple CompoundConstraints";
    return result;
}

// Diagnostic Test C: Only first two constraints from failing test
void init_diagnostic_two_constraints(int n, SerialDenseVector<int,double>& x) {
    if (n != 3) return;
    x(0) = 0.5;
    x(1) = 0.5;
    x(2) = 0.5;
}

void diagnostic_two_constraints_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                                    double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 3) return;
    
    if (mode & NLPFunction) {
        fx = x(0)*x(0) + 2.0*x(1)*x(1) + 3.0*x(2)*x(2);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * x(0);
        g(1) = 4.0 * x(1);
        g(2) = 6.0 * x(2);
        result = NLPGradient;
    }
}

CompoundConstraint* create_diagnostic_two_constraints_constraints(int n) {
    if (n != 3) return nullptr;
    
    // Only x + y <= 2 and y + z <= 1
    SerialDenseMatrix<int,double> A1(1, 3);
    A1(0, 0) = -1.0; A1(0, 1) = -1.0; A1(0, 2) = 0.0;  // -x - y >= -2
    SerialDenseVector<int,double> b1(1);
    b1(0) = -2.0;
    LinearInequality* ineq1 = new LinearInequality(A1, b1);
    
    SerialDenseMatrix<int,double> A2(1, 3);
    A2(0, 0) = 0.0; A2(0, 1) = -1.0; A2(0, 2) = -1.0;  // -y - z >= -1
    SerialDenseVector<int,double> b2(1);
    b2(0) = -1.0;
    LinearInequality* ineq2 = new LinearInequality(A2, b2);
    
    OptppArray<Constraint> constraints(2);
    constraints[0] = Constraint(ineq1);
    constraints[1] = Constraint(ineq2);
    
    return new CompoundConstraint(constraints);
}

TestResult run_diagnostic_two_constraints() {
    TestResult result;
    
    try {
        NLF1 nlp(3, diagnostic_two_constraints_obj, init_diagnostic_two_constraints, create_diagnostic_two_constraints_constraints);
        
        OptQNIPS optimizer(&nlp);
        optimizer.setMaxIter(300);
        optimizer.setFcnTol(1.0e-6);
        optimizer.setGradTol(1.0e-6);
        optimizer.setConTol(1.0e-7);
        optimizer.setMeritFcn(ArgaezTapia);
        optimizer.setSearchStrategy(TrustRegion);
        
        optimizer.optimize();
        
        result.final_point = nlp.getXc();
        result.final_objective = nlp.getF();
        result.constraint_violation = check_constraint_violation(result.final_point,
                                                               create_diagnostic_two_constraints_constraints(3));
        result.iterations = optimizer.getIter();
        
        result.success = result.constraint_violation < 1e-5;
        result.message = result.success ? "PASSED" : "FAILED - constraint violation";
        
    } catch (...) {
        result.success = false;
        result.message = "FAILED - exception";
    }
    
    return result;
}

// Diagnostic Test D: Only first constraint from failing test
void init_diagnostic_one_constraint(int n, SerialDenseVector<int,double>& x) {
    if (n != 3) return;
    x(0) = 0.5;
    x(1) = 0.5;
    x(2) = 0.5;
}

void diagnostic_one_constraint_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                                   double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 3) return;
    
    if (mode & NLPFunction) {
        fx = x(0)*x(0) + 2.0*x(1)*x(1) + 3.0*x(2)*x(2);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * x(0);
        g(1) = 4.0 * x(1);
        g(2) = 6.0 * x(2);
        result = NLPGradient;
    }
}

CompoundConstraint* create_diagnostic_one_constraint_constraints(int n) {
    if (n != 3) return nullptr;
    
    // Only x + y <= 2
    SerialDenseMatrix<int,double> A(1, 3);
    A(0, 0) = -1.0; A(0, 1) = -1.0; A(0, 2) = 0.0;  // -x - y >= -2
    SerialDenseVector<int,double> b(1);
    b(0) = -2.0;
    
    LinearInequality* ineq = new LinearInequality(A, b);
    
    OptppArray<Constraint> constraints(1);
    constraints[0] = Constraint(ineq);
    
    return new CompoundConstraint(constraints);
}

TestResult run_diagnostic_one_constraint() {
    TestResult result;
    
    try {
        NLF1 nlp(3, diagnostic_one_constraint_obj, init_diagnostic_one_constraint, create_diagnostic_one_constraint_constraints);
        
        OptQNIPS optimizer(&nlp);
        optimizer.setMaxIter(300);
        optimizer.setFcnTol(1.0e-6);
        optimizer.setGradTol(1.0e-6);
        optimizer.setConTol(1.0e-7);
        optimizer.setMeritFcn(ArgaezTapia);
        optimizer.setSearchStrategy(TrustRegion);
        
        optimizer.optimize();
        
        result.final_point = nlp.getXc();
        result.final_objective = nlp.getF();
        result.constraint_violation = check_constraint_violation(result.final_point,
                                                               create_diagnostic_one_constraint_constraints(3));
        result.iterations = optimizer.getIter();
        
        result.success = result.constraint_violation < 1e-5;
        result.message = result.success ? "PASSED" : "FAILED - constraint violation";
        
    } catch (...) {
        result.success = false;
        result.message = "FAILED - exception";
    }
    
    return result;
}

// Diagnostic Test E: Different starting point for failing test
void init_diagnostic_different_start(int n, SerialDenseVector<int,double>& x) {
    if (n != 3) return;
    // Start at a feasible point that satisfies all constraints
    x(0) = 1.0;  // x + y = 1.5 <= 2 ✓
    x(1) = 0.5;  // y + z = 1.0 <= 1 ✓  
    x(2) = 0.5;  // x + z = 1.5 <= 2 ✓
}

void diagnostic_different_start_obj(int mode, int n, const SerialDenseVector<int,double>& x,
                                    double& fx, SerialDenseVector<int,double>& g, int& result) {
    if (n != 3) return;
    
    if (mode & NLPFunction) {
        fx = x(0)*x(0) + 2.0*x(1)*x(1) + 3.0*x(2)*x(2);
        result = NLPFunction;
    }
    
    if (mode & NLPGradient) {
        g(0) = 2.0 * x(0);
        g(1) = 4.0 * x(1);
        g(2) = 6.0 * x(2);
        result = NLPGradient;
    }
}

CompoundConstraint* create_diagnostic_different_start_constraints(int n) {
    if (n != 3) return nullptr;
    
    // Same as original failing test: x + y <= 2, y + z <= 1, x + z <= 2
    SerialDenseMatrix<int,double> A1(1, 3);
    A1(0, 0) = -1.0; A1(0, 1) = -1.0; A1(0, 2) = 0.0;  // -x - y >= -2
    SerialDenseVector<int,double> b1(1);
    b1(0) = -2.0;
    LinearInequality* ineq1 = new LinearInequality(A1, b1);
    
    SerialDenseMatrix<int,double> A2(1, 3);
    A2(0, 0) = 0.0; A2(0, 1) = -1.0; A2(0, 2) = -1.0;  // -y - z >= -1
    SerialDenseVector<int,double> b2(1);
    b2(0) = -1.0;
    LinearInequality* ineq2 = new LinearInequality(A2, b2);
    
    SerialDenseMatrix<int,double> A3(1, 3);
    A3(0, 0) = -1.0; A3(0, 1) = 0.0; A3(0, 2) = -1.0;  // -x - z >= -2
    SerialDenseVector<int,double> b3(1);
    b3(0) = -2.0;
    LinearInequality* ineq3 = new LinearInequality(A3, b3);
    
    OptppArray<Constraint> constraints(3);
    constraints[0] = Constraint(ineq1);
    constraints[1] = Constraint(ineq2);
    constraints[2] = Constraint(ineq3);
    
    return new CompoundConstraint(constraints);
}

TestResult run_diagnostic_different_start() {
    TestResult result;
    
    try {
        NLF1 nlp(3, diagnostic_different_start_obj, init_diagnostic_different_start, create_diagnostic_different_start_constraints);
        
        OptQNIPS optimizer(&nlp);
        optimizer.setMaxIter(300);
        optimizer.setFcnTol(1.0e-6);
        optimizer.setGradTol(1.0e-6);
        optimizer.setConTol(1.0e-7);
        optimizer.setMeritFcn(ArgaezTapia);
        optimizer.setSearchStrategy(TrustRegion);
        
        optimizer.optimize();
        
        result.final_point = nlp.getXc();
        result.final_objective = nlp.getF();
        result.constraint_violation = check_constraint_violation(result.final_point,
                                                               create_diagnostic_different_start_constraints(3));
        result.iterations = optimizer.getIter();
        
        result.success = result.constraint_violation < 1e-5;
        result.message = result.success ? "PASSED" : "FAILED - constraint violation";
        
    } catch (...) {
        result.success = false;
        result.message = "FAILED - exception";
    }
    
    return result;
}

// ============================================================================
// COMPREHENSIVE TEST RUNNER
// ============================================================================

AllTestResults run_all_constraint_tests() {
    AllTestResults results;
    
    std::cout << "\n=== Running Comprehensive Constraint Tests ===\n" << std::endl;
    
    results.linear_eq_test1 = run_linear_eq_test1();
    print_test_result("Linear Equality Test 1", results.linear_eq_test1);
    
    results.linear_eq_test2 = run_linear_eq_test2();
    print_test_result("Linear Equality Test 2", results.linear_eq_test2);
    
    results.linear_ineq_test1 = run_linear_ineq_test1();
    print_test_result("Linear Inequality Test 1", results.linear_ineq_test1);
    
    results.linear_ineq_test2 = run_linear_ineq_test2();
    print_test_result("Linear Inequality Test 2", results.linear_ineq_test2);
    
    results.bounds_test1 = run_bounds_test1();
    print_test_result("Bounds Test 1", results.bounds_test1);
    
    results.bounds_test2 = run_bounds_test2();
    print_test_result("Bounds Test 2", results.bounds_test2);
    
    results.mixed_linear_test1 = run_mixed_linear_test1();
    print_test_result("Mixed Linear Test 1", results.mixed_linear_test1);
    
    results.nonlinear_eq_test1 = run_nonlinear_eq_test1();
    print_test_result("Nonlinear Equality Test 1", results.nonlinear_eq_test1);
    
    results.nonlinear_ineq_test1 = run_nonlinear_ineq_test1();
    print_test_result("Nonlinear Inequality Test 1", results.nonlinear_ineq_test1);
    
    results.mixed_nonlinear_test1 = run_mixed_nonlinear_test1();
    print_test_result("Mixed Nonlinear Test 1", results.mixed_nonlinear_test1);
    
    std::cout << "\n--- Diagnostic Tests for Multiple Linear Inequality Constraints ---\n" << std::endl;
    
    results.diagnostic_single_matrix = run_diagnostic_single_matrix();
    print_test_result("Diagnostic Single Matrix", results.diagnostic_single_matrix);
    
    results.diagnostic_separate_compounds = run_diagnostic_separate_compounds();
    print_test_result("Diagnostic Separate Compounds", results.diagnostic_separate_compounds);
    
    results.diagnostic_two_constraints = run_diagnostic_two_constraints();
    print_test_result("Diagnostic Two Constraints", results.diagnostic_two_constraints);
    
    results.diagnostic_one_constraint = run_diagnostic_one_constraint();
    print_test_result("Diagnostic One Constraint", results.diagnostic_one_constraint);
    
    results.diagnostic_different_start = run_diagnostic_different_start();
    print_test_result("Diagnostic Different Start", results.diagnostic_different_start);
    
    print_all_test_results(results);
    
    return results;
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

double check_constraint_violation(const SerialDenseVector<int,double>& x, CompoundConstraint* constraints) {
    if (!constraints) return 0.0;
    
    try {
        SerialDenseVector<int,double> residual = constraints->evalResidual(x);
        double max_violation = 0.0;
        
        for (int i = 0; i < residual.length(); i++) {
            max_violation = std::max(max_violation, std::abs(residual(i)));
        }
        
        return max_violation;
    } catch (...) {
        return 1e6;  // Large value indicating evaluation failed
    }
}

void print_test_result(const std::string& test_name, const TestResult& result) {
    std::cout << std::setw(30) << std::left << test_name << ": ";
    
    if (result.success) {
        std::cout << "PASSED";
    } else {
        std::cout << "FAILED";
    }
    
    std::cout << " | f = " << std::setw(8) << std::fixed << std::setprecision(4) << result.final_objective;
    std::cout << " | violation = " << std::setw(8) << std::scientific << std::setprecision(2) << result.constraint_violation;
    std::cout << " | iter = " << std::setw(3) << result.iterations;
    std::cout << " | " << result.message << std::endl;
}

void print_all_test_results(const AllTestResults& results) {
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << results.passed_tests() << "/" << results.total_tests() << std::endl;
    
    if (results.passed_tests() == results.total_tests()) {
        std::cout << "All tests PASSED!" << std::endl;
    } else {
        std::cout << "Some tests FAILED. Review individual results above." << std::endl;
    }
    std::cout << std::endl;
}

} // namespace ConstraintTests