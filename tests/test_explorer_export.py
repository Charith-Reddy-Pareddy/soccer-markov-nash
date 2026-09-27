"""Exporter must preserve the transition engine's actual reward objective."""
from scripts.explorer_data import _outcome_repr, _reward_repr
from soccer_nash.game import Action, SoccerGame


def test_territory_export_preserves_nonterminal_reward():
    g = SoccerGame(scoring="territory", territory_reward=.05)
    s = (5, 2, 0, 2, 0)
    assert _reward_repr(g, s, Action.U, Action.D) == .05


def test_stochastic_rewards_stay_aligned_with_exported_outcomes():
    g = SoccerGame(move_order="random", scoring="territory", territory_reward=.05)
    index = {s: i for i, s in enumerate(g.states())}
    for s in [(5, 2, 6, 2, 0), (6, 2, 5, 2, 0)]:
        for a0, a1 in g.joint_actions():
            outs = g.transitions(s, a0, a1)
            trans = _outcome_repr(g, s, a0, a1, index)
            rewards = _reward_repr(g, s, a0, a1)
            if len(outs) == 1:
                assert rewards == outs[0][2][0]
            else:
                assert len(trans) == len(rewards) == len(outs)
                for (idx, prob), r0, (p, ns, r) in zip(trans, rewards, outs):
                    assert prob == p
                    assert r0 == r[0]
                    assert idx == (-(ns[4] + 1) if g.is_terminal(ns) else index[ns])
