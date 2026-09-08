"""Which non-terminal states can actually occur in a game.

The mixed-strategy fraction is more meaningful against the states a game can
reach from its kickoff than against the full cross-product of positions, so the
phase diagram (``scripts/phase_diagram.py``) reports ``mixed / reachable``.
"""

from __future__ import annotations

from collections import deque

from soccer_nash.game import SoccerGame, State


def reachable_states(game: SoccerGame, start: State | None = None) -> set[State]:
    """Every non-terminal state reachable from ``start`` (default: the kickoff)
    under some sequence of joint actions, including outcomes of stochastic
    transitions."""
    root = start if start is not None else game.initial_state()
    seen: set[State] = {root}
    queue: deque[State] = deque([root])
    joint = game.joint_actions()
    while queue:
        s = queue.popleft()
        for a0, a1 in joint:
            for _prob, ns, _r in game.transitions(s, a0, a1):
                if game.is_terminal(ns) or ns in seen:
                    continue
                seen.add(ns)
                queue.append(ns)
    return seen
