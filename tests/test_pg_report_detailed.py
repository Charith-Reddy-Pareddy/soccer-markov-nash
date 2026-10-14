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
                   "Reproducing the numbers", "Are the two players mirror images?", "Other kinds of games"):
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
    assert f"the exact solver wins {rnd['vs_best_response'][0]:.0%} when it starts with the ball" in TEXT
    assert f"{rnd['vs_best_response_column'][0]:.0%} when it starts without the ball" in TEXT
    more = RES["more_seeds"]["conditions"]
    assert f"wins {more['a2c_fp_10k']['ball_ci'][0]:.0%} when it starts with the ball" in TEXT


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
    assert block.count("<table") == 1 and block.count("<svg") == 2 and "95% interval" in block


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
    assert "Mirror gap" not in TEXT  # the measure is called by its lower-case name
    assert "Nash gap (exploitability)" in TEXT
    assert "sum of both players' gains in expected discounted score" in TEXT
    assert "Nash gap</th>" in HTML and "<td><b>0.000</b></td>" in HTML
    assert "not a win rate" in TEXT
    assert "not a theoretical ceiling" in TEXT


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


def _tables():
    from html.parser import HTMLParser

    class Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.tables, self.row, self.cell, self.cur = [], None, None, None

        def handle_starttag(self, tag, attrs):
            if tag == "table":
                self.cur = []
                self.tables.append((dict(attrs).get("class", ""), self.cur))
            elif tag == "tr":
                self.row = []
            elif tag in ("td", "th"):
                self.cell = ""

        def handle_endtag(self, tag):
            if tag in ("td", "th"):
                self.row.append(self.cell.strip())
                self.cell = None
            elif tag == "tr":
                self.cur.append(self.row)
                self.row = None

        def handle_data(self, data):
            if self.cell is not None:
                self.cell += data

    parser = Parser()
    parser.feed(HTML)
    return parser.tables


def _wtl(cell):
    m = re.match(r"(\d+)%\s*tie\s*(\d+)%\s*·\s*loss\s*(\d+)%", cell)
    return tuple(int(x) for x in m.groups())


def _name(x):
    return f"{x['label']}, {x['training']}"


def test_every_win_tie_loss_table_matches_the_data_and_each_row_sums_to_a_hundred():
    boards = [RES, RES["random"], RES["continuing"]]
    tables = [t for _, t in _tables() if t and t[0][:2] == ["learner", "vs. a random player"]]
    assert len(tables) == 3
    for board, table in zip(boards, tables):
        rows = {r[0]: r[1:] for r in table[1:]}
        for label, entry in [("Exact solver", board["exact"])] + [(_name(x), x) for x in board["learners"]]:
            got = [_wtl(c) for c in rows[label]]
            want = [tuple(int(f"{v:.0%}"[:-1]) for v in entry[k]) for k in ("vs_random", "vs_nash", "vs_best_response")]
            assert got == want, label
            assert all(abs(sum(g) - 100) <= 2 for g in got)


def test_the_win_rate_and_mixed_state_tables_match_the_data():
    tabs = _tables()
    wins = next(t for c, t in tabs if c == "wins")
    boards = [RES, RES["random"], RES["continuing"]]

    def seat(entry):
        ball, other = entry["vs_best_response"][0], entry["vs_best_response_column"][0]
        return [f"{ball:.0%}", f"{other:.0%}", f"{(ball + other) / 2:.0%}",
                f"{entry['exploitability']['mean']:.3f}"]

    exact = next(r for r in wins if r and r[0] == "Exact solver")
    assert exact[1:] == sum((seat(b["exact"]) for b in boards), [])
    for row in wins[3:]:
        want = []
        for b in boards:
            hit = next((x for x in b["learners"] if _name(x) == row[0]), None)
            want += seat(hit) if hit else ["—"] * 4
        assert row[1:] == want, row[0]
    mixed = next(t for _, t in tabs if t and "distance from the exact mix" in t[0])
    m = RES["mixed"]
    by = {r[0]: r for r in mixed[1:]}
    for x in m["learners"]:
        assert by[_name(x)][3] == f"{x['tv_row']:.2f}" and by[_name(x)][4] == f"{x['share_mixing_row']:.0%}"


def test_the_action_probability_tables_match_the_saved_policy_outputs():
    import json

    outputs = json.loads((ROOT / "experiments" / "pg_policy_outputs.json").read_text())
    probs = [t for c, t in _tables() if c == "probs"]
    mixed_states = len(RES["mixed"]["policy_outputs"]["states"])
    for table, k in zip(probs[mixed_states:], range(len(outputs["states"]))):
        for row in table[2:]:
            if row[0] == "Exact solver":
                want = outputs["exact"]["row"][k] + outputs["exact"]["col"][k]
            else:
                learner = next(x for x in outputs["learners"] if x["label"] == row[0])
                want = learner["row"][k] + learner["col"][k]
            assert [int(c[:-1]) for c in row[1:9]] == [int(f"{v * 100:.0f}") for v in want]


def test_ranges_never_read_zero_to_zero_and_zero_wins_are_not_called_more_than_zero():
    assert not re.search(r"\b(\d+)% to \1%", TEXT)
    assert "more than 0%" not in TEXT


def test_the_symmetry_and_small_game_tables_and_the_phase_table_are_in_the_report():
    block = HTML.split("Are the two players mirror images?</h2>")[1].split("<h2>")[0]
    assert block.count("<table") == 1 and "mirror gap" in block and "Exact solver" in block
    games = HTML.split("Other kinds of games</h2>")[1].split("<h2>")[0]
    assert "Battle of the sexes" in games and "keeps switching" in games and "Matching pennies" in games
    assert "How long to train each best response" in HTML and "time between snapshots" in HTML
