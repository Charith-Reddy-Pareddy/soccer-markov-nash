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


# Validate the artifact used by the actual browser, not a separately solved fixture.
def test_all_exported_states_preserve_transitions_rewards_and_equilibrium():
    import json
    from pathlib import Path

    import numpy as np

    from scripts.explorer_data import BOARDS

    payload = json.loads((Path(__file__).parents[1] / 'docs/data/explorer.json').read_text())
    for bid, kw in BOARDS.items():
        board = payload['boards'][bid]
        game = SoccerGame(**kw)
        states = list(game.states())
        index = {s: i for i, s in enumerate(states)}
        assert board['state_count'] == len(states)
        assert board['state_list'] == [','.join(map(str, s)) for s in states]
        for key, record in board['states'].items():
            state = tuple(map(int, key.split(',')))
            value, p, q, matrix, trans, _, rewards = record
            m = np.array(matrix)
            p, q = np.array(p), np.array(q)
            assert abs(p.sum() - 1) < 1e-12, (bid, key)
            assert abs(q.sum() - 1) < 1e-12, (bid, key)
            assert min(p) >= 0 and min(q) >= 0, (bid, key)
            lower, upper = min(p @ m), max(m @ q)
            assert upper - lower < 1e-6, (bid, key, lower, upper)
            assert lower - 1e-6 <= value <= upper + 1e-6, (bid, key)
            for k, (a0, a1) in enumerate(game.joint_actions()):
                assert trans[k] == _outcome_repr(game, state, a0, a1, index), (bid, key)
                assert rewards[k] == _reward_repr(game, state, a0, a1), (bid, key)
                expected = 0.0
                for prob, ns, reward in game.transitions(state, a0, a1):
                    if game.is_terminal(ns):
                        continuation = 0.0
                    else:
                        assert ns[:2] != ns[2:4], (bid, key, ns)
                        continuation = board['states'][','.join(map(str, ns))][0]
                    expected += prob * (reward[0] + payload['gamma'] * continuation)
                assert abs(m[int(a0), int(a1)] - expected) < 1e-12, (bid, key, a0, a1)
            # Any selected pure action must avoid weak dominance within security ties.
            for payoffs, policy in ((m, p), (-m.T, q)):
                if np.count_nonzero(policy) != 1:
                    continue
                chosen = int(np.argmax(policy))
                for alternative in payoffs:
                    dominates = (np.all(alternative >= payoffs[chosen])
                                 and np.any(alternative > payoffs[chosen] + 1e-9))
                    assert not dominates, (bid, key, chosen)
