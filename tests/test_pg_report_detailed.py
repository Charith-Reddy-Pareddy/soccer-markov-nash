# ruff: noqa: E501
import importlib.util
import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pg_report_detailed", ROOT / "scripts" / "pg_report_detailed.py")
detailed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(detailed)
RES = detailed.load_data()
HTML = detailed.build_html(RES)
TEXT = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", HTML))
HEADS = re.findall(r"<h2>(.*?)</h2>", HTML)


def test_the_win_rate_table_comes_first_after_the_summary_and_contents():
    first_table = HTML.index("<table")
    assert HTML[first_table:first_table + 25] == '<table class="wins"><tr><'
    assert HTML.index("<h2>1. Win rates against the best response</h2>") < first_table
    assert first_table < HTML.index("<svg")


def test_the_sections_are_numbered_in_order_and_listed_in_the_contents():
    assert [h.split(".")[0] for h in HEADS] == [str(k) for k in range(1, len(HEADS) + 1)]
    assert len(HEADS) >= 14
    toc = HTML.split('<ul class="toc">')[1].split("</ul>")[0]
    assert re.findall(r"<li>(.*?)</li>", toc) == HEADS
    for needed in ("Setup", "Results by board", "Mixed states of the random board", "Action probabilities",
                   "Does training converge?", "Training choices", "More seeds", "Related work",
                   "Reproducing the numbers"):
        assert any(needed in h for h in HEADS)


def test_the_report_is_detailed_with_many_tables_and_plots():
    assert HTML.count("<table") >= 20 and HTML.count("<svg") >= 35


def test_every_board_has_a_wins_ties_and_losses_table_and_the_mixed_states_have_a_table():
    for board in ("The deterministic board (A10)", "The random move-order board", "The continuing game"):
        assert board in TEXT
    assert HTML.count("vs. a random player</th><th>vs. the exact Nash policy</th><th>vs. the best response") >= 3
    assert "distance from the exact mix" in HTML and "player 0 mixes" in HTML
    assert HTML.count('<table class="probs">') >= 7  # three mixed states and four deterministic positions


def test_the_numbers_in_the_summary_and_the_seat_table_come_from_the_data():
    rnd = RES["random"]["exact"]
    assert f"the exact solver wins {rnd['vs_best_response'][0]:.0%} from the ball seat" in TEXT
    assert f"{rnd['vs_best_response_column'][0]:.0%} from the other seat" in TEXT
    more = RES["more_seeds"]["conditions"]
    assert f"wins {more['a2c_fp_10k']['ball_ci'][0]:.0%} from the ball seat" in TEXT


def test_every_training_choice_has_a_chart_and_a_table_with_ranges_of_single_seeds():
    for study in RES["choices"]:
        assert study["title"] in HTML
    assert HTML.count('class="sub">') > 100
    block = HTML.split("<h2>13. Training choices</h2>")[1].split("<h2>14.")[0]
    assert block.count("<table") == len(RES["choices"]) and block.count("<svg") == len(RES["choices"])


def test_the_ten_seed_table_lists_the_four_settings_with_intervals():
    block = HTML.split("More seeds</h2>")[1].split("<h2>")[0]
    for label in ("A2C, fictitious play, 10,000 iterations", "PPO, standard, replay of 4 iterations"):
        assert label in block
    assert block.count("<table") == 1 and block.count("<svg") == 2 and "95% intervals" in block


def test_the_experiments_table_is_escaped_and_lists_the_runner_commands():
    block = HTML.split('<table class="small methods">')[1].split("</table>")[0]
    assert "pg_finite.py" in block and "pg_curves.py" in block and "rps_nn.py" in block
    assert "&lt;setting&gt;" in block and "<setting>" not in block and "**" not in block


def test_the_related_work_has_both_groups_and_says_how_it_was_checked():
    assert "Older work" in TEXT and "Recent work" in TEXT and "Google Scholar" in TEXT
    for item in RES["related"]:
        assert item["url"] in HTML


def test_the_report_names_no_one_and_leaves_exploitability_to_a_definition():
    assert not re.search(r"professor|\bdog\b|sheep|not yet", TEXT, re.I)
    assert "mirror gap" not in TEXT.lower()
    assert "Not reported: it measures the attacker" in TEXT


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext not installed")
def test_the_published_detailed_pdf_is_long_and_never_mentions_the_professor():
    pdf = ROOT / "docs" / "policy_gradient_detailed.pdf"
    assert pdf.exists()
    pages = int(re.search(r"Pages:\s+(\d+)", subprocess.run(
        ["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout).group(1))
    assert pages >= 25
    text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    assert "professor" not in text.lower() and "1. Win rates against the best response" in text


def test_the_tab_the_footer_and_the_docs_index_link_the_detailed_pdf():
    site = ROOT / "site" / "src"
    assert "policy_gradient_detailed.pdf" in (site / "PolicyPage.jsx").read_text()
    assert "policy_gradient_detailed.pdf" in (site / "Footer.jsx").read_text()
    assert "policy_gradient_detailed.pdf" in (ROOT / "docs" / "README.md").read_text()
    assert "pg-detailed-pdf" in (ROOT / "Makefile").read_text()
