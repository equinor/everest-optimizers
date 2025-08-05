"""Tests for expected solutions using everest_optimizers.minimize() with method='optpp_q_nips'


Testing the OptQNIPS (Quasi-Newton Interior-Point Solver) method from everest_optimizers.minimize().
In dakota OPTPP this optimization algorithm is referred to as OptQNIPS.

Runs a basic but varied set of tests to validate that correct solution is achieved while handling
*unconstrained problem
*linear equality constraints
*linear inequality constraints
*bound constraints
*mixed constraints
*nonlinear equality constraints
*nonlinear inequality constraints
*nonlinear mixed constraints
"""


