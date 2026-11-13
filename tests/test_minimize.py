from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from everest_optimizers import EvaluationMode, minimize

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

    from everest_optimizers import EvaluationResult
    from everest_optimizers._optpp import Bounds, LinearConstraints

    Evaluate = Callable[[NDArray[np.float64], EvaluationMode], EvaluationResult]


pytestmark = pytest.mark.usefixtures("work_dir")


@pytest.fixture
def work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _quadratic(target: list[float]) -> Evaluate:
    def evaluate(x: NDArray[np.float64], mode: EvaluationMode) -> EvaluationResult:
        delta = x - target
        values = np.array([delta @ delta]) if mode & EvaluationMode.VALUES else None
        gradients = (
            2 * delta[np.newaxis, :] if mode & EvaluationMode.GRADIENTS else None
        )
        return values, gradients

    return evaluate


def _rosenbrock(x: NDArray[np.float64], mode: EvaluationMode) -> EvaluationResult:
    even, odd = x[0::2], x[1::2]
    values = None
    gradients = None
    if mode & EvaluationMode.VALUES:
        values = np.array([np.sum(100 * (odd - even**2) ** 2 + (1 - even) ** 2)])
    if mode & EvaluationMode.GRADIENTS:
        gradient = np.zeros_like(x)
        gradient[0::2] = -400 * even * (odd - even**2) - 2 * (1 - even)
        gradient[1::2] = 200 * (odd - even**2)
        gradients = gradient[np.newaxis, :]
    return values, gradients


def _with_constraints(
    target: list[float],
    constraints: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    jacobian: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> Evaluate:
    objective = _quadratic(target)

    def evaluate(x: NDArray[np.float64], mode: EvaluationMode) -> EvaluationResult:
        values, gradients = objective(x, mode)
        if values is not None:
            values = np.concatenate((values, constraints(x)))
        if gradients is not None:
            gradients = np.vstack((gradients, jacobian(x)))
        return values, gradients

    return evaluate


TOLERANCES = {"convergence_tolerance": 1e-8, "gradient_tolerance": 1e-8}


def test_minimize_solves_unconstrained_problem() -> None:
    x, status = minimize(
        np.zeros(2),
        evaluate=_rosenbrock,
        options=TOLERANCES | {"max_iterations": 200},
    )
    assert status > 0
    np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-4)


@pytest.mark.parametrize("x0", [[10.0, 10.0], [-5.0, 8.0], [0.1, 0.9], [-2.0, -2.0]])
def test_minimize_solves_quadratic_from_different_starts(x0: list[float]) -> None:
    x, status = minimize(np.array(x0), evaluate=_quadratic([2.0, -1.0]))
    assert status > 0
    np.testing.assert_allclose(x, [2.0, -1.0], atol=1e-4)


def test_minimize_solves_large_problem() -> None:
    x, status = minimize(
        np.tile([-1.2, 1.0], 50),
        evaluate=_rosenbrock,
        options=TOLERANCES | {"max_iterations": 1000},
    )
    assert status > 0
    np.testing.assert_allclose(x, np.ones(100), atol=1e-4)


@pytest.mark.parametrize(
    ("lower", "upper", "x0", "expected"),
    [
        ([2.5, -3.0], [3.0, 3.0], [3.0, 0.0], [2.5, -1.0]),
        ([-np.inf, -np.inf], [np.inf, -1.5], [0.0, -2.0], [2.0, -1.5]),
        ([2.5, -np.inf], [np.inf, -1.5], [3.0, -2.0], [2.5, -1.5]),
        ([2.5, -1.5], [3.0, -1.0], [2.8, -1.2], [2.5, -1.0]),
        ([2.5, -np.inf], [np.inf, np.inf], [10.0, 10.0], [2.5, -1.0]),
    ],
)
def test_minimize_respects_bounds(
    lower: list[float], upper: list[float], x0: list[float], expected: list[float]
) -> None:
    x, _ = minimize(
        np.array(x0),
        evaluate=_quadratic([2.0, -1.0]),
        bounds=(np.array(lower), np.array(upper)),
        options=TOLERANCES,
    )
    np.testing.assert_allclose(x, expected, atol=1e-4)


def test_minimize_passes_an_infeasible_initial_point_unchanged() -> None:
    evaluated: list[NDArray[np.float64]] = []
    objective = _quadratic([2.0, -1.0])

    def evaluate(x: NDArray[np.float64], mode: EvaluationMode) -> EvaluationResult:
        evaluated.append(x.copy())
        return objective(x, mode)

    minimize(
        np.zeros(2),
        evaluate=evaluate,
        bounds=(np.array([2.5, -np.inf]), np.full(2, np.inf)),
    )
    np.testing.assert_array_equal(evaluated[0], np.zeros(2))


@pytest.mark.parametrize(
    ("coefficients", "lower", "upper", "expected"),
    [
        ([[1.0, 1.0]], [1.0], [1.0], [2.0, -1.0]),
        ([[1.0, 1.0]], [-np.inf], [0.5], [1.75, -1.25]),
        ([[1.0, 1.0], [1.0, 0.0]], [1.0, -np.inf], [1.0, 1.5], [1.5, -0.5]),
    ],
)
def test_minimize_respects_linear_constraints(
    coefficients: list[list[float]],
    lower: list[float],
    upper: list[float],
    expected: list[float],
) -> None:
    x, status = minimize(
        np.zeros(2),
        evaluate=_quadratic([2.0, -1.0]),
        linear_constraints=(np.array(coefficients), np.array(lower), np.array(upper)),
        options=TOLERANCES,
    )
    assert status > 0
    np.testing.assert_allclose(x, expected, atol=1e-4)


def test_minimize_handles_mixed_linear_constraint_rows() -> None:
    linear_constraints = (
        np.array(
            [[1.0, 1.0, 0.0], [1.0, -2.0, 0.0], [0.0, 0.0, 1.0], [2.0, 0.0, -1.0]]
        ),
        np.array([1.0, -np.inf, 0.5, -0.2]),
        np.array([1.0, -1.0, 0.5, 0.4]),
    )
    x, status = minimize(
        np.array([0.3, 0.7, 0.4]),
        evaluate=_quadratic([0.2, 0.8, 0.5]),
        linear_constraints=linear_constraints,
        options=TOLERANCES,
    )
    assert status > 0
    np.testing.assert_allclose(x, [0.2, 0.8, 0.5], atol=1e-4)


@pytest.mark.parametrize("bound", [np.inf, 1e20])
def test_minimize_ignores_linear_constraints_without_active_bounds(
    bound: float,
) -> None:
    x, status = minimize(
        np.zeros(2),
        evaluate=_quadratic([2.0, -1.0]),
        bounds=(np.full(2, -3.0), np.full(2, 3.0)),
        linear_constraints=(np.ones((1, 2)), np.array([-bound]), np.array([bound])),
        options=TOLERANCES,
    )
    assert status > 0
    np.testing.assert_allclose(x, [2.0, -1.0], atol=1e-4)


def test_minimize_combines_bounds_and_linear_constraints() -> None:
    x, status = minimize(
        np.zeros(2),
        evaluate=_quadratic([2.0, -1.0]),
        bounds=(np.array([0.0, -np.inf]), np.array([1.5, np.inf])),
        linear_constraints=(
            np.array([[1.0, 1.0], [1.0, 0.0]]),
            np.array([1.0, -np.inf]),
            np.array([1.0, 1.5]),
        ),
        options=TOLERANCES,
    )
    assert status > 0
    np.testing.assert_allclose(x, [1.5, -0.5], atol=1e-4)


def test_minimize_respects_a_nonlinear_equality() -> None:
    x, status = minimize(
        np.array([1.5, 1.5]),
        evaluate=_with_constraints(
            [2.0, 2.0],
            lambda x: np.array([x @ x - 4.0]),
            lambda x: 2 * x[np.newaxis, :],
        ),
        bounds=(np.full(2, -5.0), np.full(2, 5.0)),
        nonlinear_equalities=np.array([True]),
        options=TOLERANCES,
    )
    assert status > 0
    np.testing.assert_allclose(x, np.sqrt([2.0, 2.0]), atol=1e-4)


def test_minimize_respects_a_nonlinear_inequality() -> None:
    x, status = minimize(
        np.zeros(2),
        evaluate=_with_constraints(
            [2.0, 2.0],
            lambda x: np.array([1.0 - x @ x]),
            lambda x: -2 * x[np.newaxis, :],
        ),
        bounds=(np.full(2, -2.0), np.full(2, 2.0)),
        nonlinear_equalities=np.array([False]),
        options=TOLERANCES,
    )
    assert status > 0
    np.testing.assert_allclose(x, np.sqrt([0.5, 0.5]), atol=1e-4)


def test_minimize_accepts_nonlinear_equalities_in_any_position() -> None:
    x, status = minimize(
        np.array([0.3, 0.7]),
        evaluate=_with_constraints(
            [2.0, 2.0],
            lambda x: np.array([x[0], x[0] + x[1] - 1.0, 1.0 - x[0]]),
            lambda _: np.array([[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]]),
        ),
        nonlinear_equalities=np.array([False, True, False]),
        options=TOLERANCES,
    )
    assert status > 0
    np.testing.assert_allclose(x, [0.5, 0.5], atol=1e-4)


@pytest.mark.parametrize(
    ("x0", "bounds", "linear_constraints"),
    [
        (np.zeros(2), None, None),
        (np.zeros(2), (np.full(2, -100.0), np.full(2, 100.0)), None),
        (np.zeros(100), None, None),
        (
            np.zeros(2),
            (np.full(2, -100.0), np.full(2, 100.0)),
            (np.ones((1, 2)), np.array([-np.inf]), np.array([100.0])),
        ),
    ],
)
def test_minimize_max_step_limits_the_distance_between_trial_points(
    x0: NDArray[np.float64],
    bounds: Bounds | None,
    linear_constraints: LinearConstraints | None,
) -> None:
    def largest_step(options: dict[str, Any]) -> float:
        points: list[NDArray[np.float64]] = []
        objective = _quadratic([10.0] * x0.size)

        def evaluate(x: NDArray[np.float64], mode: EvaluationMode) -> EvaluationResult:
            points.append(x.copy())
            return objective(x, mode)

        minimize(
            x0,
            evaluate=evaluate,
            bounds=bounds,
            linear_constraints=linear_constraints,
            options=options,
        )
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).max())

    assert largest_step({}) > 1.0
    assert largest_step({"max_step": 1.0}) <= 1.0 + 1e-12


def test_minimize_stops_at_max_iterations() -> None:
    _, status = minimize(
        np.zeros(2),
        evaluate=_rosenbrock,
        bounds=(np.full(2, -5.0), np.full(2, 5.0)),
        options={"max_iterations": 1},
    )
    assert status == -4


def test_minimize_writes_no_default_output_file(work_dir: Path) -> None:
    minimize(np.zeros(2), evaluate=_quadratic([2.0, -1.0]))
    assert list(work_dir.iterdir()) == []


def test_minimize_leaves_an_existing_default_output_file_intact(
    work_dir: Path,
) -> None:
    default_output = work_dir / "OPT_DEFAULT.out"
    default_output.write_text("unrelated\n", encoding="utf-8")
    minimize(np.zeros(2), evaluate=_quadratic([2.0, -1.0]))
    assert default_output.read_text(encoding="utf-8") == "unrelated\n"


def test_minimize_writes_the_output_file(work_dir: Path) -> None:
    minimize(
        np.zeros(2),
        evaluate=_quadratic([2.0, -1.0]),
        options={"output_file": "optimizer.out"},
    )
    assert [path.name for path in work_dir.iterdir()] == ["optimizer.out"]


@pytest.mark.parametrize(
    ("x0", "bounds", "linear_constraints", "nonlinear_equalities", "match"),
    [
        (np.zeros((2, 1)), None, None, None, "x0 must be one-dimensional"),
        (np.zeros(2), (np.zeros(3), np.ones(3)), None, None, "broadcast"),
        (
            np.zeros(2),
            None,
            (np.ones((1, 3)), np.zeros(1), np.ones(1)),
            None,
            "coefficients must have shape",
        ),
        (
            np.zeros(2),
            None,
            None,
            np.zeros((1, 1), dtype=np.bool_),
            "nonlinear_equalities must be one-dimensional",
        ),
    ],
)
def test_minimize_rejects_inputs_with_invalid_shapes(
    x0: NDArray[np.float64],
    bounds: Bounds | None,
    linear_constraints: LinearConstraints | None,
    nonlinear_equalities: NDArray[np.bool_] | None,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        minimize(
            x0,
            evaluate=_quadratic([2.0, -1.0]),
            bounds=bounds,
            linear_constraints=linear_constraints,
            nonlinear_equalities=nonlinear_equalities,
        )


def test_minimize_rejects_lower_bounds_above_upper_bounds() -> None:
    with pytest.raises(ValueError, match="lower bounds must not exceed"):
        minimize(
            np.zeros(2),
            evaluate=_quadratic([2.0, -1.0]),
            bounds=(np.array([1.0, -1.0]), np.array([0.0, 1.0])),
        )
