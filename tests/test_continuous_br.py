import numpy as np
import pytest

from soccer_nash.continuous_br import bisect_br, gradient_br, quadratic_br

METHODS = [bisect_br, gradient_br, quadratic_br]


@pytest.mark.parametrize("br", METHODS)
def test_finds_interior_peak_of_a_parabola(br):
    f = lambda a: -(a - 1.3) ** 2  # noqa: E731
    assert br(f, 0.0, 2 * np.pi) == pytest.approx(1.3, abs=1e-3)


@pytest.mark.parametrize("br", METHODS)
def test_clips_to_the_boundary_when_the_peak_is_outside(br):
    f = lambda a: -(a - 5.0) ** 2  # noqa: E731
    assert br(f, 0.0, 2.0) == pytest.approx(2.0, abs=1e-3)


@pytest.mark.parametrize("br", [bisect_br, gradient_br])
def test_non_quadratic_single_peak(br):
    f = lambda a: np.sin(a)  # noqa: E731  peak at pi/2 on [0, pi]
    assert br(f, 0.0, np.pi) == pytest.approx(np.pi / 2, abs=1e-3)


def test_best_response_to_a_fixed_opponent_action():
    q = lambda a1, a2: a1 * a2 - a1**2 / 2  # noqa: E731  row best reply is a1 = a2
    assert bisect_br(lambda a: q(a, 0.7), -2.0, 2.0) == pytest.approx(0.7, abs=1e-3)
