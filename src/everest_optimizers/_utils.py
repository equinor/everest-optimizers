from collections.abc import Generator
from contextlib import contextmanager
from os import devnull
from pathlib import Path
from typing import Any

from numpy.typing import NDArray
from scipy.optimize import OptimizeResult

from everest_optimizers import pyoptpp

from ._problem import NLF1Problem
from .pyoptpp import OptBCQNewton, OptQNewton, OptQNIPS

NewtonOptimizer = OptQNewton | OptBCQNewton | OptQNIPS

_DEFAULTS: dict[str, Any] = {
    "debug": False,
    "output_file": devnull,
    "search_method": "line_search",
    "search_pattern_size": 64,
    "max_step": 1000,
    "gradient_multiplier": 0.1,
    "max_iterations": 100,
    "max_function_evaluations": 1000,
    "convergence_tolerance": 1e-4,
    "gradient_tolerance": 1e-4,
}


def set_basic_newton_options(
    optimizer: NewtonOptimizer, options: dict[str, Any] | None
) -> None:
    if options is None:
        options = {}

    unknown_options = set(options.keys()) - set(_DEFAULTS.keys())
    if unknown_options:
        msg = f"Unknown optimization option(s): {unknown_options}"
        raise RuntimeError(msg)

    defaulted_options = {
        option: options.get(option, default_value)
        for option, default_value in _DEFAULTS.items()
    }

    if defaulted_options["debug"]:
        optimizer.setDebug()

    optimizer.setOutputFile(defaulted_options["output_file"], 0)

    match defaulted_options["search_method"].lower():
        case "trust_region" | "trustregion":
            if isinstance(optimizer, OptBCQNewton):
                msg = "OptBCQNewton does not support the 'trust_region' search method"
                raise ValueError(msg)
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustRegion)
        case "line_search" | "linesearch":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.LineSearch)
        case "trust_pds" | "trustpds":
            optimizer.setSearchStrategy(pyoptpp.SearchStrategy.TrustPDS)
        case _:
            msg = (
                f"Unknown search method: {defaulted_options['search_method']}. "
                "Valid options: trust_region, line_search, trust_pds"
            )
            raise ValueError(msg)

    max_step = defaulted_options["max_step"]
    if "max_step" in options:
        optimizer.setTRSize(max_step)
    else:
        optimizer.setTRSize(options.get("tr_size", max_step))

    optimizer.setGradMult(defaulted_options["gradient_multiplier"])
    optimizer.setSearchSize(defaulted_options["search_pattern_size"])
    optimizer.setMaxIter(defaulted_options["max_iterations"])
    optimizer.setMaxFeval(defaulted_options["max_function_evaluations"])
    optimizer.setFcnTol(defaulted_options["convergence_tolerance"])
    optimizer.setGradTol(defaulted_options["gradient_tolerance"])


def run_newton(
    optimizer: NewtonOptimizer, problem: NLF1Problem, x0: NDArray
) -> OptimizeResult:
    optimizer.optimize()

    solution_vector = problem.nlf1_problem.getXc()
    x_final = solution_vector.to_numpy()
    f_final = problem.nlf1_problem.getF()

    result = OptimizeResult(  # type: ignore[call-arg]
        x=x_final,
        fun=f_final,
        nfev=problem.nfev,
        njev=problem.njev,
        nit=0,  # iteration count not available
        success=True,
        status=0,
        message="Optimization terminated successfully",
        jac=problem.current_g if problem.current_g is not None else None,
    )

    optimizer.cleanup()
    return result


@contextmanager
def remove_default_output(
    options: dict[str, Any] | None = None,
) -> Generator[Any, Any, Any]:
    # This context manager is a hack to resolve the issue that OPT++ always
    # creates a file "OPT_DEFAULT.out" by default. Unless "output_file" is set,
    # no file should be created, hence we remove it.
    output_file = None if options is None else options.get("output_file")
    default_name = "OPT_DEFAULT.out"
    default_out_path = Path(default_name)
    default_out_exists = default_out_path.exists()
    try:
        yield
    finally:
        if not default_out_exists and output_file != default_name:
            default_out_path.unlink(missing_ok=True)
