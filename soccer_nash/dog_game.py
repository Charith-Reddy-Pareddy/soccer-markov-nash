"""The dog / debate game -- a discretised version of the research group's
continuous 2D game, used to exercise the general-sum solver against a known
closed-form answer.

Two players move on a grid, each trying to pull a "dog" toward their own house.
The dog sits at a weighted average of the two players' positions:

    dog = w_a * pos_a + w_b * pos_b        (w_a + w_b = 1)

Player ``a`` wants the dog near house ``a``, player ``b`` near house ``b``.
Rewards are negative distances, so the game is **general-sum** in 2D (the two
distances are not complementary). The only Nash equilibrium is both players
parked in their own corners, with the dog at ``w_a * house_a + w_b * house_b``.

A one-dimensional variant (``line=True``) is zero-sum -- the two distances sum
to a constant -- and can be solved with the fast :class:`NashQIteration`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 5 moves: stay, up, down, left, right.
_MOVES = ((0, 0), (0, 1), (0, -1), (-1, 0), (1, 0))


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


@dataclass(frozen=True)
class DogGame:
    size: int = 5
    w_a: float = 0.5
    house_a: tuple[int, int] = (0, 0)
    house_b: tuple[int, int] | None = None  # defaults to the opposite corner
    line: bool = False  # 1D board (y fixed at 0) -> zero-sum

    n_actions: tuple[int, int] = field(init=False, default=(5, 5))

    def __post_init__(self) -> None:
        if self.house_b is None:
            far = (self.size - 1, 0) if self.line else (self.size - 1, self.size - 1)
            object.__setattr__(self, "house_b", far)
        if self.line:
            object.__setattr__(self, "n_actions", (2, 2))  # left / right only

    # ------------------------------------------------------------------ states
    def positions(self):
        s = self.size
        if self.line:
            return [(x, 0) for x in range(s)]
        return [(x, y) for x in range(s) for y in range(s)]

    def states(self):
        return [(a, b) for a in self.positions() for b in self.positions()]

    def dog(self, pa: tuple[int, int], pb: tuple[int, int]) -> tuple[float, float]:
        return (
            self.w_a * pa[0] + (1 - self.w_a) * pb[0],
            self.w_a * pa[1] + (1 - self.w_a) * pb[1],
        )

    # ------------------------------------------------------------- transition
    def _moves(self):
        return ((-1, 0), (1, 0)) if self.line else _MOVES

    def _apply(self, pos: tuple[int, int], move: int) -> tuple[int, int]:
        dx, dy = self._moves()[move]
        return (
            _clamp(pos[0] + dx, 0, self.size - 1),
            _clamp(pos[1] + dy, 0, self.size - 1),
        )

    def transition(self, state, a0: int, a1: int):
        pa, pb = state
        return [(1.0, (self._apply(pa, a0), self._apply(pb, a1)))]

    def reward(self, state, a0: int, a1: int, s_next):
        pa, pb = s_next
        dx, dy = self.dog(pa, pb)
        da = abs(dx - self.house_a[0]) + abs(dy - self.house_a[1])
        db = abs(dx - self.house_b[0]) + abs(dy - self.house_b[1])
        return -da, -db

    # ------------------------------------------------------------- reference
    def equilibrium_dog(self) -> tuple[float, float]:
        return self.dog(self.house_a, self.house_b)
