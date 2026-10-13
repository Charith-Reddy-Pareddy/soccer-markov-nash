import importlib.util
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pg_report", ROOT / "scripts" / "pg_report.py")
pg_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg_report)
RES = pg_report.load_data()
HTML = pg_report.build_html(RES)
TEXT = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", HTML))


def test_the_win_rate_table_is_the_first_table_and_comes_before_everything_but_the_title():
    first_table = HTML.index("<table")
    assert HTML[first_table:first_table + 25] == '<table class="wins"><tr><'
    assert first_table < HTML.index("2. Setup") and first_table < HTML.index("<svg")
    assert HTML.index("1. Win rates against the best response") < first_table


def test_the_report_is_short_with_two_tables_and_five_charts():
    assert HTML.count("<table") == 2
    charts = 2 + 1 + len(RES["more_seeds"]["pairs"])  # curves, rps, ten seeds
    assert HTML.count("<svg") == charts
    for dropped in ("Action probabilities", "Mixed states of the random board", "More training",
                    "Fictitious play with long best responses", "Network sharing and trimming"):
        assert dropped not in TEXT


def test_the_report_defines_the_win_rate_and_the_seats_and_explains_exploitability():
    for phrase in ("Win rate", "Starts with the ball", "Starts without the ball",
                   "Average of both", "best response to it",
                   "a win is more goals than the opponent in 100 steps"):
        assert phrase in TEXT
    assert "Exploitability is not used for soccer" in TEXT
    assert "perfect attacker could gain" in TEXT and "not a win rate" in TEXT


def test_the_report_uses_the_agreed_terms_and_names_no_one():
    assert "Standard" in TEXT and "Fictitious play" in TEXT
    ours = TEXT.split("Related work")[0]  # related-work titles are quoted as published
    assert "self-play" not in ours.lower()
    assert not re.search(r"professor|\bdog\b|sheep|not yet", TEXT, re.I)
    assert "mirror gap" not in TEXT.lower()


def test_the_report_has_every_section_in_order():
    heads = ["1. Win rates against the best response", "2. Setup", "3. Findings",
             "4. Does training converge?", "5. Rock-paper-scissors with neural networks",
             "6. Training choices", "7. Related work"]
    positions = [HTML.index(h) for h in heads]
    assert positions == sorted(positions)


def test_every_headline_number_in_the_findings_comes_from_the_data():
    rnd = RES["random"]["exact"]
    ball = f"{rnd['vs_best_response'][0]:.0%}"
    assert f"the exact solver wins {ball} when it starts with the ball" in TEXT
    assert f"{rnd['vs_best_response_column'][0]:.0%} when it starts without the ball" in TEXT
    mixed = RES["mixed"]
    assert f"the exact solver wins {mixed['exact']['vs_nash'][0]:.0%}" in TEXT
    more = RES["more_seeds"]["conditions"]
    assert f"{more['a2c_fp_10k']['ball_ci'][0]:.0%} when it starts with the ball" in TEXT


def test_the_training_choice_lines_cover_every_comparison_and_the_ablation_is_labelled():
    lines = pg_report.finding_lines(RES)
    assert len(lines) >= 4 + len(RES["choices"])
    assert "does not show convergence to the unregularized equilibrium" in TEXT
    assert "never weights" in TEXT


def test_change_range_reports_the_smallest_and_largest_change_in_points():
    rows = [{"a": {"mean": 0.1}, "b": {"mean": 0.3}}, {"a": {"mean": 0.2}, "b": {"mean": 0.15}}]
    assert pg_report.change_range(rows, "b", "a") == "−5 to +20"
    assert pg_report.pts(0.0) == "+0" and pg_report.pts(-0.034) == "−3"
    assert pg_report.pts(-0.003) == "+0"  # never "−0"


def test_variant_bars_draw_a_bar_per_variant_and_a_dot_per_seed():
    run = {"mean": 0.2, "runs": [0.1, 0.2, 0.3]}
    row = {"label": "A2C", "training": "standard", "a": run, "b": run, "c": run}
    svg = pg_report.variant_bars([row], 0.67, keys={"a": "#111", "b": "#222", "c": "#333"})
    assert svg.count("<rect") == 3 and svg.count("<circle") == 9 and "exact solver 67%" in svg
    tie = {**run, "tie_nash_mean": 0.5, "tie_nash_runs": [0.4, 0.6]}
    row = {"label": "A2C", "training": "", "a": tie, "b": tie}
    svg = pg_report.variant_bars([row], 1.0, field="tie_nash_", keys={"a": "#111", "b": "#222"})
    assert svg.count("<circle") == 4 and ">A2C<" in svg


def test_the_line_chart_has_a_line_per_run_and_honours_a_fixed_top():
    svg = pg_report.line_chart([("a", [0.0, 0.1]), ("b", [0.05, None])], [1, 2])
    assert svg.count("<path") == 2
    assert ">100%<" in pg_report.line_chart([("a", [0.2, 0.3])], [1, 2], top=1.0)


def test_the_win_rate_table_has_seat_columns_a_balanced_average_and_dashes_for_missing_learners():
    def entry(label, ball, other):
        return {"label": label, "training": "standard", "vs_best_response": [ball, 0, 1 - ball],
                "vs_best_response_column": [other, 0, 1 - other]}

    a, b = entry("A2C", 0.1, 0.3), entry("A2C, exact critic", 0.0, 0.0)
    exact = entry("x", 0.5, 0.3)
    html = pg_report.win_rate_table([("first", {"exact": exact, "learners": [a]}),
                                     ("second", {"exact": exact, "learners": [a, b]})])
    assert html.count("<tr>") == 5 and html.count("&mdash;") == 3
    for cell in ("<b>50%</b>", "<b>30%</b>", "<b>40%</b>"):  # exact solver: ball, other, balanced
        assert cell in html
    for cell in ("<td>10%</td>", "<td>30%</td>", "<td>20%</td>"):  # A2C
        assert cell in html
    assert pg_report.seat_rates(a) == [0.1, 0.3, 0.2]


def test_the_before_after_chart_has_a_legend_and_three_bars_per_starting_position():
    group = {"label": "Starts with the ball", "base": (0.1, 0.05, 0.15),
             "variant": (0.3, 0.2, 0.4), "exact": 0.67}
    names = ("2,000 iterations", "10,000 iterations")
    svg = pg_report.before_after_chart([group, {**group, "label": "Average of both"}], names)
    assert svg.count("<rect") == 3 + 2 * 3  # legend squares and three bars for each of two groups
    assert "2,000 iterations" in svg and "10,000 iterations" in svg and "Exact solver" in svg
    assert "10%" in svg and "(5% to 15%)" in svg and "67%" in svg
    flat = {**group, "base": (0.0, 0.0, 0.0)}
    assert "(0% to 0%)" not in pg_report.before_after_chart([flat], ("a", "b"))


def test_the_ten_seed_charts_are_before_after_charts_with_readable_names():
    pair = RES["more_seeds"]["pairs"][0]
    assert pair["baseline_name"] == "2,000 iterations"
    assert pair["variant_name"] == "10,000 iterations"
    groups = pg_report.pair_groups(RES["more_seeds"], pair, RES["random"]["exact"])
    assert [g["label"] for g in groups] == [
        "Starts with the ball", "Starts without the ball", "Average of both"]
    assert groups[0]["exact"] == RES["random"]["exact"]["vs_best_response"][0]
    assert "against 10,000 iterations" in TEXT
