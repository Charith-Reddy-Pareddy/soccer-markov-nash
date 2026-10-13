import importlib.util
import json
import re
import shutil
import subprocess

import pytest
import test_pg_policy_outputs as shared  # for ROOT only

ROOT = shared.ROOT
SITE = ROOT / "site"
LANDING = (SITE / "src" / "Landing.jsx").read_text()
PAGE = (SITE / "src" / "PolicyPage.jsx").read_text()
RESULTS = json.loads((SITE / "src" / "pgResults.json").read_text())

spec = importlib.util.spec_from_file_location("pg_site_data", ROOT / "scripts" / "pg_site_data.py")
pg_site_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg_site_data)


def learner(label, training):
    return next(x for x in RESULTS["learners"] if x["label"] == label and x["training"] == training)


# ---- the page shows what the runs produced -------------------------------------------------
def test_the_results_file_is_exactly_what_the_result_csvs_produce():
    assert pg_site_data.build() == RESULTS


def test_there_is_one_row_per_learner_and_a_longer_run_for_each_selfplay_learner():
    assert [(x["label"], x["training"]) for x in RESULTS["learners"]] == [
        (a, t) for a, t, _, _ in pg_site_data.LEARNERS]
    assert [x["label"] for x in RESULTS["longer"]] == ["REINFORCE", "A2C", "PPO"]
    assert all(len(x["runs"]) == 2 for x in RESULTS["longer"])


def test_win_tie_loss_triples_sum_to_one():
    rows = [RESULTS["exact"], *RESULTS["learners"],
            RESULTS["random"]["exact"], *RESULTS["random"]["learners"]]
    for r in rows:
        for key in ("vs_random", "vs_nash", "vs_best_response"):
            assert sum(r[key]) == pytest.approx(1.0, abs=1e-4)


def test_the_exact_solver_is_unexploitable_and_ties_itself():
    assert RESULTS["exact"]["exploitability"]["mean"] == 0
    assert RESULTS["exact"]["vs_nash"] == [0.0, 1.0, 0.0]


# ---- every sentence on the page that makes a claim is true of the data ---------------------
@pytest.mark.parametrize("label", ["REINFORCE", "A2C", "PPO"])
def test_fictitious_play_wins_more_against_random_and_selfplay_ties_the_equilibrium_more(label):
    fp, sp = learner(label, "fictitious play"), learner(label, "standard")
    assert fp["vs_random"][0] > sp["vs_random"][0]
    assert sp["vs_nash"][1] > fp["vs_nash"][1]


def test_the_best_response_runs_end_higher_than_they_start():
    assert RESULTS["fp_br"]
    for run in RESULTS["fp_br"]:
        first, last = run["checkpoints"][0], run["checkpoints"][-1]
        assert last["exploitability"] > first["exploitability"]


def test_no_learner_is_a_mirror_image_pair():
    assert min(x["mirror_gap"] for x in RESULTS["learners"]) > 0.3


def test_the_random_board_has_every_learner_with_three_seeds_and_an_unexploitable_reference():
    rnd = RESULTS["random"]
    assert [(x["label"], x["training"]) for x in rnd["learners"]] == [
        (a, t) for a, t, _, _ in pg_site_data.LEARNERS]
    assert all(x["seeds"] == 3 for x in rnd["learners"])
    assert rnd["exact"]["exploitability"]["mean"] == 0


def test_every_claim_the_random_board_text_makes_is_true():
    det = [x["exploitability"]["mean"] for x in RESULTS["learners"]]
    rnd = RESULTS["random"]["learners"]
    mean_rnd = [x["exploitability"]["mean"] for x in rnd]
    # "lower than on the deterministic board": both ends of the range, and learner by learner
    assert max(mean_rnd) < max(det) and min(mean_rnd) < min(det)
    assert all(r["exploitability"]["mean"] < d["exploitability"]["mean"]
               for r, d in zip(rnd, RESULTS["learners"]))
    # "the exact solver wins more than any learner against the exact Nash policy"
    assert RESULTS["random"]["exact"]["vs_nash"][0] > max(x["vs_nash"][0] for x in rnd)
    # "not mirror images"
    assert min(x["mirror_gap"] for x in rnd) > 0.2


def test_no_learner_wins_a_game_against_the_exact_nash_policy_on_the_deterministic_board():
    assert all(x["vs_nash"][0] == 0 for x in RESULTS["learners"])
    assert RESULTS["exact"]["vs_nash"] == [0.0, 1.0, 0.0]


def test_the_hero_numbers_are_computed_not_typed():
    assert "Math.min(...rnd.learners.map" in PAGE and "mixed.exact.vs_nash[0]" in PAGE


# ---- what the pages must not say ------------------------------------------------------------
def test_the_landing_page_has_no_star_button_and_points_to_the_tab():
    assert "Star on GitHub" not in LANDING
    assert "View the code on GitHub" in LANDING
    assert 'href="policy.html"' in LANDING
    assert "Not yet: every learner" not in LANDING
    assert "<tbody>" not in LANDING[LANDING.index('id="policy"'):LANDING.index('id="why"')]


def test_the_policy_page_sources_never_name_the_professor_or_the_dog_game():
    for text in (PAGE, json.dumps(RESULTS)):
        assert not re.search(r"professor|\bdog\b|sheep|not yet", text, re.I)


def test_the_tab_is_wired_into_the_build_the_nav_and_the_footer():
    assert 'policy: resolve(root, "policy.html")' in (SITE / "vite.config.js").read_text()
    assert 'href="policy.html"' in (SITE / "src" / "Nav.jsx").read_text()
    assert 'href="policy.html"' in (SITE / "src" / "Footer.jsx").read_text()
    built = (ROOT / "docs" / "policy.html").read_text()
    asset = re.search(r'assets/(policy-[A-Za-z0-9_-]+\.js)', built).group(1)
    assert (ROOT / "docs" / "assets" / asset).exists()


def test_the_built_policy_page_bundle_is_clean_and_has_the_probabilities():
    built = (ROOT / "docs" / "policy.html").read_text()
    asset = re.search(r'assets/(policy-[A-Za-z0-9_-]+\.js)', built).group(1)
    js = (ROOT / "docs" / "assets" / asset).read_text()
    assert not re.search(r"professor|\bdog\b|sheep|Not yet", js, re.I)
    assert "What the trained policies output" in js


# ---- the report and the PDF -----------------------------------------------------------------
def test_the_report_html_never_mentions_the_professor_and_has_the_probabilities():
    html = (ROOT / "docs" / "policy_gradient.html").read_text().lower()
    assert "professor" not in html
    assert "assumptions and open questions" not in html


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")
def test_the_published_pdf_never_mentions_the_professor():
    text = subprocess.run(["pdftotext", str(ROOT / "docs" / "policy_gradient.pdf"), "-"],
                          capture_output=True, text=True, check=True).stdout.lower()
    assert "professor" not in text
    assert "assumptions and open questions" not in text


def test_no_published_page_mentions_the_professor():
    pages = [*(ROOT / "docs").glob("*.md"), *(ROOT / "docs").glob("*.html"), ROOT / "README.md"]
    offenders = [p.name for p in pages if "professor" in p.read_text().lower()]
    assert offenders == []


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")
def test_no_published_pdf_mentions_the_professor():
    offenders = []
    for pdf in (ROOT / "docs").glob("*.pdf"):
        text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                              text=True, check=True).stdout.lower()
        if "professor" in text:
            offenders.append(pdf.name)
    assert offenders == []


def test_no_published_page_or_pdf_mentions_the_dog_game():
    pages = [*(ROOT / "docs").glob("*.md"), *(ROOT / "docs").glob("*.html"), ROOT / "README.md"]
    offenders = [p.name for p in pages if re.search(r"\b(dog|sheep)\b", p.read_text().lower())]
    if shutil.which("pdftotext"):
        for pdf in (ROOT / "docs").glob("*.pdf"):
            text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                                  text=True, check=True).stdout.lower()
            if re.search(r"\b(dog|sheep)\b", text):
                offenders.append(pdf.name)
    assert offenders == []


def test_the_dog_game_code_and_data_are_gone():
    for rel in ("soccer_nash/dog_game.py", "scripts/dog_game.py", "tests/test_dog_game.py",
                "experiments/dog_game.csv", "experiments/dog_game_dqn.csv", "docs/dog_game.md"):
        assert not (ROOT / rel).exists(), rel


def test_the_tab_leads_with_the_win_rate_against_the_best_response():
    for part in ("function WinCell", "function BestResponseBars", "function RpsPlot"):
        assert part in PAGE
    assert "Win rate against the best response" in PAGE
    assert "Exploitability</th>" not in PAGE and "Mirror gap" not in PAGE
    assert PAGE.index("Win rate against the best response") < PAGE.index("against each opponent")
    for k in ("Deterministic board (A10)", "Random move-order board", "Continuing game"):
        assert k in PAGE


def test_the_tab_uses_the_agreed_terms():
    assert "self-play" not in PAGE.lower()
    assert "standard" in PAGE.lower() and "fictitious play" in PAGE.lower()


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")
def test_the_pdf_has_the_random_board_and_shows_wins_as_percentages():
    text = subprocess.run(["pdftotext", str(ROOT / "docs" / "policy_gradient.pdf"), "-"],
                          capture_output=True, text=True, check=True).stdout
    text = " ".join(text.split())  # table headers wrap across lines
    assert "Random move-order board" in text
    assert "Win rates against the best response" in text
    assert "W/T/L" not in text


def test_the_mixed_state_claims_on_the_page_hold_for_the_data():
    mixed = RESULTS["mixed"]
    assert mixed["mixed_states"] == 94 and sum(mixed["support_sizes"].values()) == 94
    assert len(mixed["learners"]) == 6 and all(x["seeds"] == 3 for x in mixed["learners"])
    # the exact solver wins more than any learner, against a random player and the exact Nash policy
    assert mixed["exact"]["vs_random"][0] > max(x["vs_random"][0] for x in mixed["learners"])
    assert mixed["exact"]["vs_nash"][0] > max(x["vs_nash"][0] for x in mixed["learners"])
    # nobody is close to the exact mix
    assert min(x["tv_row"] for x in mixed["learners"]) > 0.3
    for x in mixed["learners"]:
        for key in ("vs_random", "vs_nash", "vs_best_response"):
            assert sum(x[key]) == pytest.approx(1.0, abs=1e-4)
    assert "Where the exact answer has to mix" in PAGE
    assert "PolicyOutputs data={mixed.policy_outputs}" in PAGE


def test_the_win_rates_come_first_on_the_tab_and_in_the_pdf():
    assert PAGE.index("Win rate against the best response") < PAGE.index("against each opponent")
    if shutil.which("pdftotext"):
        pdf = str(ROOT / "docs" / "policy_gradient.pdf")
        out = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True, check=True)
        text = " ".join(out.stdout.split())
        assert text.index("1. Win rates against the best response") < text.index("2. Setup")
        assert "exploitability (mean" not in text.lower()


def test_the_variants_cover_a2c_and_ppo_in_both_schemes_with_three_seeds_each():
    for board in ("random", "deterministic"):
        rows = RESULTS["variants"][board]
        assert [(r["label"], r["training"]) for r in rows] == [
            ("A2C", "standard"), ("A2C", "fictitious play"),
            ("PPO", "standard"), ("PPO", "fictitious play")]
        for r in rows:
            for key in ("baseline", "shared", "trimmed"):
                assert len(r[key]["runs"]) == len(r[key]["tie_nash_runs"]) == 3
                assert r[key]["mean"] == pytest.approx(sum(r[key]["runs"]) / 3, abs=1e-5)
                assert sum(r[key]["vs_random"]) == pytest.approx(1.0, abs=1e-4)


def test_the_baseline_variant_is_the_published_board_run_and_none_beats_the_exact_solver():
    boards = {"random": RESULTS["random"], "deterministic": RESULTS}
    for board, res in boards.items():
        exact = res["exact"]["vs_best_response"][0]
        by = {(x["label"], x["training"]): x for x in res["learners"]}
        for r in RESULTS["variants"][board]:
            base = by[(r["label"], r["training"])]
            assert r["baseline"]["mean"] == pytest.approx(base["vs_best_response"][0], abs=1e-5)
            assert r["baseline"]["tie_nash_mean"] == pytest.approx(base["vs_nash"][1], abs=1e-5)
            assert all(r[k]["mean"] <= exact + 0.05 for k in ("baseline", "shared", "trimmed"))


def test_the_trimmed_ppo_claim_in_the_text_is_true():
    rows = RESULTS["variants"]["random"]
    ppo = next(r for r in rows if r["label"] == "PPO" and r["training"] == "standard")
    assert ppo["trimmed"]["vs_random"][0] < ppo["baseline"]["vs_random"][0] - 0.3
    assert "stops scoring" in PAGE


def test_the_deterministic_tie_caveat_in_the_text_is_true():
    rows = RESULTS["variants"]["deterministic"]
    ppo = next(r for r in rows if r["label"] == "PPO" and r["training"] == "standard")
    assert ppo["trimmed"]["tie_nash_mean"] > 0.99
    assert ppo["trimmed"]["vs_random"][0] < ppo["baseline"]["vs_random"][0]
    assert "A tie does not mean equilibrium play" in PAGE


def test_the_continuing_variants_have_three_seeds_and_the_text_claims_hold():
    rows = RESULTS["variants"]["continuing"]
    assert len(rows) == 4
    ref = RESULTS["continuing"]["exact"]["vs_best_response"][0]
    for r in rows:
        for key in ("baseline", "shared", "trimmed"):
            assert len(r[key]["runs"]) == len(r[key]["tie_nash_runs"]) == 3
            assert r[key]["mean"] <= ref + 0.01  # nobody wins against the best response
    by = {(x["label"], x["training"]): x for x in RESULTS["continuing"]["learners"]}
    for r in rows:
        base = by[(r["label"], r["training"])]
        assert r["baseline"]["tie_nash_mean"] == pytest.approx(base["vs_nash"][1], abs=1e-5)
    spread = max(max(x["tie_nash_runs"]) - min(x["tie_nash_runs"])
                 for r in rows for x in (r["baseline"], r["shared"]))
    assert spread > 0.5  # single seeds of one setup differ by half the scale or more
    assert "three seeds cannot settle changes of this size" in PAGE


def test_only_the_explanatory_tables_and_the_win_rate_table_remain():
    assert PAGE.count("<table") == 5 and "function WinRateTable" in PAGE and "<WinRateTable" in PAGE
    html = (ROOT / "docs" / "policy_gradient.html").read_text()
    assert html.count("<table") == 2 and html.count('<table class="wins">') == 1
    assert html.index('<table class="wins">') == html.index("<table")  # the win rates come first
    assert "<table" not in (SITE / "src" / "PolicyOutputs.jsx").read_text()


def test_the_win_rate_table_in_the_report_matches_the_data():
    import re

    html = (ROOT / "docs" / "policy_gradient.html").read_text()
    i = html.index('<table class="wins">')
    block = html[i:html.index("</table>", i)]
    rows = [re.sub(r"<[^>]+>", " ", r).split() for r in block.split("<tr>")[1:]]
    exact = next(r for r in rows if r[:2] == ["Exact", "solver"])
    boards = {"deterministic": RESULTS["exact"], "random": RESULTS["random"]["exact"]}
    want = []
    for e in boards.values():
        ball, other = e["vs_best_response"][0], e["vs_best_response_column"][0]
        want += [f"{ball * 100:.0f}%", f"{other * 100:.0f}%", f"{(ball + other) * 50:.0f}%"]
    assert exact[2:8] == want
    a2c = next(r for r in rows if r[0] == "A2C," and r[1] == "standard")
    learner = RESULTS["learners"][2]
    ball, other = learner["vs_best_response"][0], learner["vs_best_response_column"][0]
    assert a2c[2:5] == [f"{round(v * 100)}%" for v in (ball, other, (ball + other) / 2)]
    assert "Average of both" in PAGE and "starts with the ball" in html


def test_the_entropy_bonus_is_described_as_an_ablation_not_a_fix():
    html = (ROOT / "docs" / "policy_gradient.html").read_text()
    for text in (PAGE, html):
        assert "stabilization" in text or "entropy" not in text
    assert "An ablation: the entropy bonus raised to 0.2" in PAGE
    assert "never network weights" in PAGE and "never weights" in html


def test_the_seat_explanation_matches_the_data():
    rnd = RESULTS["random"]
    assert rnd["kickoff_value"] > 0.1  # the ball-holding seat is ahead at the kickoff
    row, col = rnd["exact"]["vs_best_response"], rnd["exact"]["vs_best_response_column"]
    assert row[0] > 0.5 > col[0]  # more than half of games from one seat, fewer from the other
    assert row[1] == 0.0 and col[1] == 0.0
    assert RESULTS["exact"]["vs_best_response"][1] == 1.0  # the deterministic board ties
    assert RESULTS["exact"]["vs_best_response_column"][1] == 1.0
    assert "Why the exact solver does not win 0% against the best response" in PAGE


def test_the_argmax_averages_cover_three_algorithms_on_three_boards_with_three_seeds():
    data = RESULTS["argmax"]
    assert set(data) == {"random", "deterministic", "continuing"}
    for board in data:
        assert [r["label"] for r in data[board]] == ["REINFORCE", "A2C", "PPO"]
        for r in data[board]:
            for key in ("softmax", "argmax"):
                assert len(r[key]["runs"]) == len(r[key]["tie_nash_runs"]) == 3
    assert "Averaging pure policies" in PAGE
    top = max(r["argmax"]["mean"] for r in data["continuing"])
    assert top < 0.05  # "no argmax learner wins more than ..." in the text


def test_the_larger_entropy_runs_cover_nine_learners_on_three_boards_and_the_text_says_ablation():
    data = RESULTS["entropy"]
    assert set(data) == {"random", "deterministic", "continuing"}
    for board, rows in data.items():
        assert [(r["label"], r["training"]) for r in rows] == [
            (a, m) for a in ("REINFORCE", "A2C", "PPO")
            for m in ("standard", "fictitious play", "fictitious play, argmax")]
        for r in rows:
            for key in ("baseline", "larger"):
                assert len(r[key]["runs"]) == len(r[key]["tie_nash_runs"]) == 3
                assert sum(r[key]["vs_random"]) == pytest.approx(1.0, abs=1e-4)
    html = (ROOT / "docs" / "policy_gradient.html").read_text()
    for text in (PAGE, html):
        assert "does not show convergence to the unregularized equilibrium" in text
    assert "not a way to recover the unregularized equilibrium" in PAGE


def test_the_baseline_of_the_entropy_comparison_is_the_published_run():
    by = {(x["label"], x["training"]): x for x in RESULTS["random"]["learners"]}
    for r in RESULTS["entropy"]["random"]:
        if r["training"] in ("standard", "fictitious play"):
            published = by[(r["label"], r["training"])]["vs_best_response"][0]
            assert r["baseline"]["mean"] == pytest.approx(published, abs=1e-5)


def test_the_neural_network_rock_paper_scissors_runs_cover_both_settings_and_all_modes():
    data = RESULTS["rps_nn"]
    assert set(data["configs"]) == {"soccer settings", "larger entropy bonus"}
    for cfg in data["configs"].values():
        assert set(cfg["modes"]) == {"standard", "fictitious", "fictitious_argmax"}
        for m in cfg["modes"].values():
            assert len(m["current"]) == len(m["aggregate"]) == data["iterations"]
            assert len(m["final_exploitability"]) == data["seeds"]


def test_the_training_choices_cover_every_comparison_with_three_seeds_per_setting():
    studies = RESULTS["choices"]
    assert [s["title"] for s in studies][:2] == [
        "Leaving out the last steps: how many?",
        "Leaving out the last steps in the continuing game"]
    assert len(studies) == 9
    for s in studies:
        assert s["metric"] in ("br", "tie") and len(s["keys"]) >= 2 and s["rows"]
        for row in s["rows"]:
            for key, _ in s["keys"]:
                assert len(row[key]["runs"]) == len(row[key]["tie_nash_runs"]) == 3
                assert row[key]["mean"] == pytest.approx(sum(row[key]["runs"]) / 3, abs=1e-5)


def test_the_baseline_settings_of_the_training_choices_are_the_published_runs():
    by = {(x["label"], x["training"]): x for x in RESULTS["random"]["learners"]}
    trims = RESULTS["choices"][0]
    for row in trims["rows"]:
        published = by[(row["label"], row["training"])]["vs_best_response"][0]
        assert row["t0"]["mean"] == pytest.approx(published, abs=1e-5)
    choices = {s["title"]: s for s in RESULTS["choices"]}
    snaps = choices["How long to train between snapshots"]
    for row in snaps["rows"]:
        published = by[(row["label"], "fictitious play")]["vs_best_response"][0]
        assert row["s5"]["mean"] == pytest.approx(published, abs=1e-5)
    iters = choices["Five times as many iterations"]
    for row in iters["rows"]:
        assert row["i2000"]["mean"] == pytest.approx(
            by[(row["label"], row["training"])]["vs_best_response"][0], abs=1e-5)


def test_the_training_curves_have_three_seeds_and_end_at_the_published_results():
    curves = RESULTS["curves"]
    assert set(curves) == {"random", "deterministic"}
    for board, c in curves.items():
        assert c["iterations"][0] == 100 and c["iterations"][-1] == 2000
        assert [(s["label"], s["mode"]) for s in c["series"]] == [
            ("A2C", "selfplay"), ("A2C", "fictitious"), ("PPO", "selfplay"), ("PPO", "fictitious")]
        for s in c["series"]:
            n = len(c["iterations"])
            assert len(s["win_br"]) == len(s["current_move"]) == len(s["aggregate_move"]) == n
            assert len(s["seed_win_br"]) == 3 and all(len(v) == n for v in s["seed_win_br"])
            assert all(0.0 <= v <= 1.0 for v in s["win_br"] + s["aggregate_move"])
    assert "Does training converge?" in PAGE


def test_the_related_work_list_has_links_and_the_page_hedges_what_was_checked():
    items = RESULTS["related"]
    assert len(items) >= 5 and all(i["url"].startswith("https://") and i["says"] for i in items)
    assert any("Neural Fictitious" in i["says"] or "NFSP" in i["says"] for i in items)
    assert "have not checked the proofs" in " ".join(PAGE.split())
    html = (ROOT / "docs" / "policy_gradient.html").read_text()
    assert "have not checked the proofs" in " ".join(html.split())
    assert all(i["url"] in html for i in items)


def _site_data_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "pg_site_data_for_tests", ROOT / "scripts" / "pg_site_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_interval_and_difference_helpers_follow_student_t():
    mod = _site_data_module()
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mean, lo, hi = mod.interval(values)
    import statistics

    half = 2.262 * statistics.stdev(values) / 10 ** 0.5
    assert mean == pytest.approx(0.55) and lo == pytest.approx(mean - half, abs=1e-5)
    assert hi == pytest.approx(mean + half, abs=1e-5)
    same = mod.difference(values, values)
    assert same[0] == 0.0 and same[1] < 0.0 < same[2]
    shifted = mod.difference(values, [v + 5 for v in values])
    assert shifted[0] == pytest.approx(5.0) and shifted[1] > 4.0
    assert mod.t95(9) == 2.262 and mod.t95(1000) == mod.T95[-1]
    near_zero = mod.interval([0.0, 0.0, 0.001, 0.0, 0.002, 0.0, 0.0, 0.0, 0.001, 0.0])
    assert near_zero[1] == 0.0 and 0.0 < near_zero[0] < near_zero[2] <= 1.0


def test_the_more_seeds_comparison_has_ten_seeds_per_condition_and_consistent_intervals():
    data = RESULTS["more_seeds"]
    assert data["seeds"] == 10
    assert set(data["conditions"]) == {"a2c_fp_base", "a2c_fp_10k", "ppo_sp_base", "ppo_sp_replay4"}
    for c in data["conditions"].values():
        assert c["seeds"] == list(range(10))
        for m in ("ball", "other", "balanced"):
            assert len(c[m]) == 10
            mean, lo, hi = c[f"{m}_ci"]
            assert lo <= mean <= hi and mean == pytest.approx(sum(c[m]) / 10, abs=1e-5)
        assert c["balanced"] == pytest.approx([(x + y) / 2 for x, y in zip(c["ball"], c["other"])])
    for pair in data["pairs"]:
        base = data["conditions"][pair["baseline_key"]]["ball_ci"][0]
        var = data["conditions"][pair["variant_key"]]["ball_ci"][0]
        assert pair["ball_diff"][0] == pytest.approx(var - base, abs=1e-5)
    assert "More seeds" in PAGE


def test_the_first_three_seeds_of_the_more_seeds_baselines_are_the_published_runs():
    by = {(x["label"], x["training"]): x for x in RESULTS["random"]["learners"]}
    data = RESULTS["more_seeds"]["conditions"]
    for key, label, training in (("a2c_fp_base", "A2C", "fictitious play"),
                                 ("ppo_sp_base", "PPO", "standard")):
        published = by[(label, training)]["vs_best_response"][0]
        assert sum(data[key]["ball"][:3]) / 3 == pytest.approx(published, abs=1e-3)


def test_the_related_work_covers_older_and_recent_papers_and_says_how_it_was_checked():
    items = RESULTS["related"]
    for i in items:
        assert i["group"] in ("older", "recent") and i["venue"] and i["authors"] and i["year"]
    older = [i for i in items if i["group"] == "older"]
    recent = [i for i in items if i["group"] == "recent"]
    assert len(older) >= 8 and len(recent) >= 8
    assert min(i["year"] for i in older) == 1951 and max(i["year"] for i in recent) >= 2024
    assert len({i["title"] for i in items}) == len(items)
    html = (ROOT / "docs" / "policy_gradient.html").read_text()
    for text in (" ".join(PAGE.split()), " ".join(html.split())):
        assert "Google Scholar" in text and "JSTOR" in text and "Older work" in text
    assert "Brown" in html and "Robinson" in html and "Littman" in html


def test_the_symmetry_table_covers_every_board_and_the_exact_solver_is_symmetric():
    boards = RESULTS["symmetry"]
    assert [b["board"] for b in boards] == ["deterministic", "random", "continuing"]
    for b in boards:
        assert b["rows"][0] == {"name": "Exact solver", "gap": 0.0}
        assert all(0.0 < r["gap"] <= 1.0 for r in b["rows"][1:])
    published = {(x["label"], x["training"]): x["mirror_gap"] for x in RESULTS["learners"]}
    first = {r["name"]: r["gap"] for r in boards[0]["rows"]}
    for (label, training), gap in published.items():
        assert first[f"{label}, {training}"] == gap
    assert "Are the two players mirror images?" in PAGE


def test_the_small_games_show_the_known_behaviour_of_best_response_dynamics():
    games = {g["game"]: g for g in RESULTS["small_games"]}
    bos = games["Battle of the sexes (coordination)"]
    assert bos["pure"] == ["BB", "SS"] and bos["mixed"] == [0.667, 0.333]
    ends = {d["start"]: d["simultaneous"] for d in bos["dynamics"]}
    assert ends["BB"] == "BB" and ends["SS"] == "SS" and ends["BS"] == "keeps switching"
    pennies = games["Matching pennies (zero-sum)"]
    assert pennies["pure"] == []
    assert all(d["alternating"] == "keeps switching" for d in pennies["dynamics"])
    assert all(abs(v - 0.5) < 0.03 for freqs in pennies["fictitious_play"].values() for v in freqs)
    assert "Coordination, anti-coordination and a zero-sum game" in PAGE


def test_the_rock_paper_scissors_phase_runs_cover_three_lengths_and_two_averages():
    data = RESULTS["rps_phases"]
    assert data["phases"] == [5, 50, 200] and data["seeds"] == 3
    modes = ("fictitious", "fictitious_argmax")
    assert set(data["runs"]) == {f"{p}/{m}" for p in (5, 50, 200) for m in modes}
    for run in data["runs"].values():
        assert len(run["final_exploitability"]) == 3 and len(run["aggregate"]) > 100
        assert all(0.0 <= v <= 2.0 for v in run["final_exploitability"])


def test_the_batch_size_study_uses_the_published_runs_for_64_episodes():
    study = next(s for s in RESULTS["choices"] if s["title"] == "How much data per iteration")
    assert [k for k, _ in study["keys"]] == ["b64", "b8", "b256"]
    by = {(x["label"], x["training"]): x for x in RESULTS["random"]["learners"]}
    for row in study["rows"]:
        published = by[(row["label"], row["training"])]["vs_best_response"][0]
        assert row["b64"]["mean"] == pytest.approx(published, abs=1e-5)
