"""A10 Part 1 deliverables: successor states (Q1/Q3) and rewards (Q2/Q4).

For a state ``(x0, y0, x1, y1, b)`` the assignment asks for the successor after
each of the 16 joint actions, in the order

    UU UD UL UR DU DD DL DR LU LD LL LR RU RD RL RR

(the first letter is player 0's move, the second is player 1's). A winning
transition is reported as ``(-1, -1, -1, -1, p)`` with ``p`` the winner, and its
reward is ``+1`` for the reference player and ``-1`` against.

The public A10 page parameterises the start position and goal rows by student
ID, so the geometry here is configurable; it defaults to the Littman soccer
layout used across this repo. Part 1 uses the deterministic move order.
"""

from __future__ import annotations

from soccer_nash.game import JOINT_ACTIONS, SoccerGame, State

# A10 joint-action labels, matching JOINT_ACTIONS order.
ACTION_LABELS: list[str] = [f"{a0.name}{a1.name}" for a0, a1 in JOINT_ACTIONS]


def successors(game: SoccerGame, state: State) -> list[State]:
    """The 16 successor states, in A10 action order."""
    if game.move_order != "deterministic":
        raise ValueError("A10 Part 1 uses the deterministic move order")
    return [game.step(state, a0, a1)[0] for a0, a1 in JOINT_ACTIONS]


def rewards(game: SoccerGame, state: State, player: int = 0) -> list[int]:
    """The 16 transition rewards from ``player``'s perspective (+1/-1/0)."""
    if game.move_order != "deterministic":
        raise ValueError("A10 Part 1 uses the deterministic move order")
    out: list[int] = []
    for a0, a1 in JOINT_ACTIONS:
        _, (r0, r1), _ = game.step(state, a0, a1)
        out.append(r0 if player == 0 else r1)
    return out


def format_successors(game: SoccerGame, state: State) -> str:
    """Q1/Q3 answer: 16 lines, five comma-separated integers each."""
    return "\n".join(",".join(map(str, s)) for s in successors(game, state))


def format_rewards(game: SoccerGame, state: State, player: int = 0) -> str:
    """Q2/Q4 answer: the 16 rewards, comma-separated, on one line."""
    return ",".join(str(r) for r in rewards(game, state, player))
