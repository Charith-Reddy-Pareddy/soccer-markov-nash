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
    assert "action probabilities" in html
    assert "assumptions and open questions" not in html


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")
def test_the_published_pdf_never_mentions_the_professor():
    text = subprocess.run(["pdftotext", str(ROOT / "docs" / "policy_gradient.pdf"), "-"],
                          capture_output=True, text=True, check=True).stdout.lower()
    assert "professor" not in text
    assert "action probabilities" in text
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
    assert "The random move-order board" in text
    assert "Win rate against the best response" in text
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
        first = text.index("Against the best response, the exact solver wins")
        assert first < text.index("How to read the numbers")
        assert "exploitability (mean" not in text.lower()
