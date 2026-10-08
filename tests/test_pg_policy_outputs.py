import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load_script("pg_policy_outputs")
DATA = json.loads((ROOT / "experiments" / "pg_policy_outputs.json").read_text())


def part(algo, mode, value):
    row = [[value, 1 - value, 0.0, 0.0]] * len(mod.STATES)
    return {"exact": {"row": row, "col": row},
            "learner": {"algo": algo, "mode": mode, "seed": 0, "iterations": 1,
                        "label": f"{algo}, {mode}", "row": row, "col": row}}


def test_merge_combines_parts_in_a_fixed_order_and_writes_both_copies(tmp_path, monkeypatch):
    (tmp_path / "experiments").mkdir()
    (tmp_path / "site" / "src").mkdir(parents=True)
    monkeypatch.setattr(mod, "EXP", tmp_path / "experiments")
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    for algo, mode in [("ppo", "fictitious"), ("reinforce", "selfplay"), ("a2c", "selfplay")]:
        (tmp_path / "experiments" / f"pg_policy_outputs_{algo}_{mode}.json").write_text(
            json.dumps(part(algo, mode, 0.5)))
    mod.merge()
    merged = json.loads((tmp_path / "experiments" / "pg_policy_outputs.json").read_text())
    assert [(lr["algo"], lr["mode"]) for lr in merged["learners"]] == [
        ("reinforce", "selfplay"), ("a2c", "selfplay"), ("ppo", "fictitious")]
    assert merged["actions"] == ["U", "D", "L", "R"]
    assert (tmp_path / "site" / "src" / "pgPolicies.json").read_text() == (
        tmp_path / "experiments" / "pg_policy_outputs.json").read_text()


def test_committed_outputs_are_probability_vectors_for_every_learner_and_state():
    assert [s["state"] for s in DATA["states"]] == [list(s) for _, _, s in mod.STATES]
    assert len(DATA["learners"]) == 6
    assert len({lr["label"] for lr in DATA["learners"]}) == 6
    vectors = [DATA["exact"]["row"], DATA["exact"]["col"]]
    for lr in DATA["learners"]:
        vectors += [lr["row"], lr["col"]]
    for block in vectors:
        assert len(block) == len(DATA["states"])
        for v in block:
            assert len(v) == 4 and min(v) >= 0 and sum(v) == pytest.approx(1.0, abs=1e-3)


def test_the_site_reads_the_same_data_as_the_report():
    site = (ROOT / "site" / "src" / "pgPolicies.json").read_text()
    assert site == (ROOT / "experiments" / "pg_policy_outputs.json").read_text()


def test_exact_row_in_the_outputs_is_what_the_solver_gives():
    from soccer_nash import finite_horizon as fh
    from soccer_nash.game import A10SoccerGame
    from soccer_nash.nash_q import NashQIteration

    game = A10SoccerGame()
    solver = NashQIteration(game, gamma=0.9, mode="hybrid", tol=1e-10)
    _, row_t, col_t = fh.solve_finite_horizon(solver, 0.9, game.max_steps)
    for k, (_, _, state) in enumerate(mod.STATES):
        assert np.allclose(DATA["exact"]["row"][k], row_t[0][state], atol=1e-4)
        assert np.allclose(DATA["exact"]["col"][k], col_t[0][state], atol=1e-4)
