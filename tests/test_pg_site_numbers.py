import csv
import re
import shutil
import statistics as st
import subprocess

import pytest
import test_pg_policy_outputs as shared  # for ROOT only

ROOT = shared.ROOT
LANDING = (ROOT / "site" / "src" / "Landing.jsx").read_text()
ROWS = list(csv.DictReader((ROOT / "experiments" / "pg_finite_a10.csv").open()))
LEARNERS = [("REINFORCE", "self-play", "reinforce", "selfplay"),
            ("REINFORCE", "fictitious play", "reinforce", "fictitious"),
            ("A2C", "self-play", "a2c", "selfplay"),
            ("A2C", "fictitious play", "a2c", "fictitious"),
            ("PPO", "self-play", "ppo", "selfplay"),
            ("PPO", "fictitious play", "ppo", "fictitious")]


def mean(rows, key):
    return st.mean(float(r[key]) for r in rows)


def wtl(rows, opp):
    return " / ".join(f"{mean(rows, f'row_{k}_vs_{opp}'):.2f}" for k in ("win", "tie", "loss"))


@pytest.mark.parametrize("label,training,algo,mode", LEARNERS)
def test_every_row_of_the_site_table_matches_the_result_file(label, training, algo, mode):
    rs = [r for r in ROWS if r["algo"] == algo and r["mode"] == mode]
    pattern = (f"<td>{label}</td><td>{training}</td><td[^>]*>{mean(rs, 'exploitability'):.2f}</td>"
               f"<td>{wtl(rs, 'random')}</td><td>{wtl(rs, 'nash')}</td>")
    assert re.search(pattern, LANDING), pattern


def test_the_exact_solver_row_matches_the_result_file():
    ex = [r for r in ROWS if r["algo"] == "exact"]
    pattern = (f'<td>Exact solver</td><td>&mdash;</td><td[^>]*>0</td><td>{wtl(ex, "random")}</td>'
               f'<td>{wtl(ex, "nash")}</td>')
    assert re.search(pattern, LANDING), pattern


def test_the_longer_training_sentence_quotes_every_long_run():
    import glob
    runs = []
    for p in glob.glob(str(ROOT / "experiments" / "pg_finite_a10_long_*.csv")):
        with open(p) as fh:
            runs += [r for r in csv.DictReader(fh) if r["algo"] != "exact"]
    assert len(runs) == 6
    for r in runs:
        assert f"{float(r['exploitability']):.2f}" in LANDING


def test_the_site_heading_does_not_claim_failure():
    assert "Not yet: every learner" not in LANDING
    assert "Policy gradient on the soccer game" in LANDING


def test_the_report_html_never_mentions_the_professor_and_has_the_probabilities():
    html = (ROOT / "docs" / "policy_gradient.html").read_text().lower()
    assert "professor" not in html
    assert "action probabilities" in html


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")
def test_the_published_pdf_never_mentions_the_professor():
    text = subprocess.run(["pdftotext", str(ROOT / "docs" / "policy_gradient.pdf"), "-"],
                          capture_output=True, text=True, check=True).stdout.lower()
    assert "professor" not in text
    assert "action probabilities" in text
