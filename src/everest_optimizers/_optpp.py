"""Configure and run the OPT++ optimizers."""

from collections.abc import Generator
from contextlib import contextmanager
from os import devnull
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _pyoptpp
from ._problem import NLF1Problem

Bounds = tuple[NDArray[np.float64], NDArray[np.float64]]
LinearConstraints = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
_Optimizer = _pyoptpp.OptQNewton | _pyoptpp.OptBCQNewton | _pyoptpp.OptQNIPS

_SEARCH_STRATEGIES = {
    "trust_region": _pyoptpp.SearchStrategy.TrustRegion,
    "line_search": _pyoptpp.SearchStrategy.LineSearch,
}
_MERIT_FUNCTIONS = {
    "norm_fmu": _pyoptpp.MeritFcn.NormFmu,
    "argaez_tapia": _pyoptpp.MeritFcn.ArgaezTapia,
    "van_shanno": _pyoptpp.MeritFcn.VanShanno,
}


def run_q_newton(
    problem: NLF1Problem, settings: dict[str, Any]
) -> tuple[NDArray[np.float64], int]:
    with _no_default_output(settings):
        optimizer = _pyoptpp.OptQNewton(problem.nlf1_problem)
        _set_newton_settings(optimizer, problem, settings)
    return _run(optimizer, problem)


def run_bcq_newton(
    problem: NLF1Problem, bounds: Bounds, settings: dict[str, Any]
) -> tuple[NDArray[np.float64], int]:
    problem.nlf1_problem.setConstraints(
        _pyoptpp.create_compound_constraint([_bound_constraint(bounds)])
    )
    with _no_default_output(settings):
        optimizer = _pyoptpp.OptBCQNewton(problem.nlf1_problem)
        _set_newton_settings(optimizer, problem, settings)
    return _run(optimizer, problem)


def run_q_nips(
    problem: NLF1Problem,
    bounds: Bounds | None,
    linear_constraints: LinearConstraints | None,
    settings: dict[str, Any],
) -> tuple[NDArray[np.float64], int]:
    constraints: list[_pyoptpp.ConstraintBase] = []
    if bounds is not None:
        constraints.append(_bound_constraint(bounds))
    if linear_constraints is not None:
        constraints.extend(_linear_constraints(linear_constraints))
    nonlinear_constraint = problem.create_nonlinear_constraint()
    if nonlinear_constraint is not None:
        constraints.append(nonlinear_constraint)
    problem.nlf1_problem.setConstraints(
        _pyoptpp.create_compound_constraint(constraints)
    )
    with _no_default_output(settings):
        optimizer = _pyoptpp.OptQNIPS(problem.nlf1_problem)
        _set_common_settings(optimizer, problem, settings)
    optimizer.setMaxFeval(settings["max_feval"])
    optimizer.setMeritFcn(_MERIT_FUNCTIONS[settings["merit_fcn"]])
    optimizer.setCenteringParameter(settings["centering_parameter"])
    optimizer.setStepLengthToBdry(settings["step_length_to_bdry"])
    return _run(optimizer, problem)


def _bound_constraint(bounds: Bounds) -> _pyoptpp.BoundConstraint:
    lower, upper = bounds
    return _pyoptpp.BoundConstraint(
        lower.size, _pyoptpp.SerialDenseVector(lower), _pyoptpp.SerialDenseVector(upper)
    )


def _linear_constraints(
    linear_constraints: LinearConstraints,
) -> list[_pyoptpp.ConstraintBase]:
    coefficients, lower, upper = linear_constraints
    is_equality = np.isfinite(lower) & (lower == upper)
    constraints: list[_pyoptpp.ConstraintBase] = []
    if not is_equality.all():
        constraints.append(
            _pyoptpp.LinearInequality(
                _pyoptpp.SerialDenseMatrix(coefficients[~is_equality]),
                _pyoptpp.SerialDenseVector(lower[~is_equality]),
                _pyoptpp.SerialDenseVector(upper[~is_equality]),
            )
        )
    if is_equality.any():
        constraints.append(
            _pyoptpp.LinearEquation(
                _pyoptpp.SerialDenseMatrix(coefficients[is_equality]),
                _pyoptpp.SerialDenseVector(lower[is_equality]),
            )
        )
    return constraints


def _set_common_settings(
    optimizer: _Optimizer, problem: NLF1Problem, settings: dict[str, Any]
) -> None:
    if settings["debug"]:
        optimizer.setDebug()
    optimizer.setOutputFile(settings.get("output_file", devnull), 0)
    optimizer.setMaxIter(settings["max_iter"])
    optimizer.setFcnTol(settings["fcn_tol"])
    optimizer.setMaxStep(settings["max_step"])
    problem.nlf1_problem.setModeOverride(settings["mode_override"])


def _set_newton_settings(
    optimizer: _pyoptpp.OptQNewton | _pyoptpp.OptBCQNewton,
    problem: NLF1Problem,
    settings: dict[str, Any],
) -> None:
    _set_common_settings(optimizer, problem, settings)
    optimizer.setMaxFeval(settings["max_feval"])
    optimizer.setGradTol(settings["grad_tol"])
    optimizer.setSearchStrategy(_SEARCH_STRATEGIES[settings["search_strategy"]])
    if "tr_size" in settings:
        optimizer.setTRSize(settings["tr_size"])
    problem.nlf1_problem.setIsExpensive(settings["is_expensive"])


def _run(
    optimizer: _Optimizer, problem: NLF1Problem
) -> tuple[NDArray[np.float64], int]:
    optimizer.optimize()
    result = (problem.nlf1_problem.getXc().to_numpy(), optimizer.getReturnCode())
    optimizer.cleanup()
    return result


@contextmanager
def _no_default_output(settings: dict[str, Any]) -> Generator[None, None, None]:
    # OPT++ creates "OPT_DEFAULT.out" when an optimizer is constructed; remove it
    # unless it existed before or is the requested output file. Wrap setOutputFile
    # too: it closes the file, and Windows cannot delete a file that is open.
    default_output = Path("OPT_DEFAULT.out")
    existed = default_output.exists()
    try:
        yield
    finally:
        if not existed and settings.get("output_file") != default_output.name:
            default_output.unlink(missing_ok=True)
