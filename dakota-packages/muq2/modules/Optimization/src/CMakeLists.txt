# USAGE:
# CreateCompileGroup(
#     <group name>
#     <brief description>
#     <library_name>
#     <other compile group dependencies>
#     <required dependencies>
#     <optional dependencies>
#     <source file 1>
#     <source file 2>
#     ...
#     <source file N>
# )

CreateCompileGroup(
    OPTIMIZATION_CORE
    "The basic building blocks to define and solve optimization problems."
    muqOptimization
    "MODELING_CORE"
    ""
    ""
    CostFunction.cpp
    ModPieceCostFunction.cpp
    Optimizer.cpp
    NewtonTrust.cpp
)

CreateCompileGroup(
    OPTIMIZATION_NLOPT
    "Wrapper around the NLOPT optimization library."
    muqOptimization
    "OPTIMIZATION_CORE;MODELING_CORE"
    "NLOPT"
    ""
    NLoptOptimizer.cpp
)
