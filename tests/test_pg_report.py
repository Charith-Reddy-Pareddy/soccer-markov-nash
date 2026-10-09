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


def test_best_response_bars_have_one_bar_per_row_and_a_reference_line():
    svg = pg_report.br_bars([("a", 0.2), ("b", 0.9)], 0.5)
    assert svg.count("<rect") == 2 and svg.count("stroke-dasharray") == 1
    assert "exact solver 50%" in svg


def test_stacked_chart_segments_fill_each_panel():
    svg = pg_report.stacked_chart([("a", [[0.5, 0.25, 0.25], [1.0, 0.0, 0.0]])], ["left", "right"])
    assert svg.count("<rect") == 6 and "left" in svg and "right" in svg


def test_rock_paper_scissors_plot_draws_both_curves_and_the_one_third_line():
    svg = pg_report.rps_plot(RES["rps"])
    assert svg.count("<path") == 2 and ">1/3<" in svg


def test_win_cell_shows_the_win_percentage_with_ties_and_losses_beneath():
    cell = pg_report.win_cell([0.29, 0.0, 0.71])
    assert "<b>29%</b>" in cell and "tie 0%" in cell and "loss 71%" in cell


def test_policy_tables_have_one_table_per_state_and_one_row_per_learner():
    d = RES["mixed"]["policy_outputs"]
    html = pg_report.policy_tables(d)
    assert html.count("<table") == len(d["states"])
    assert html.count("Exact solver") == len(d["states"])
    assert html.count(d["learners"][0]["label"]) == len(d["states"])


def test_the_report_leads_with_win_rates_and_defines_every_number():
    first = TEXT.index("Against the best response, the exact solver wins")
    assert first < TEXT.index("How to read the numbers")
    for phrase in ("Win rate against the best response", "Win rate against the exact Nash policy",
                   "Win rate against a random player", "A win"):
        assert phrase in TEXT
    assert "Exploitability, the best-response player's gain, is not reported" in TEXT


def test_the_report_uses_the_agreed_terms_and_names_no_one():
    assert "standard" in TEXT and "fictitious play" in TEXT.lower()
    assert "self-play" not in TEXT.lower()
    assert not re.search(r"professor|\bdog\b|sheep|not yet", TEXT, re.I)
    assert "mirror gap" not in TEXT.lower()


def test_the_report_has_a_section_per_board_and_the_plots():
    assert "The deterministic board (A10)" in TEXT and "The random move-order board" in TEXT
    assert ("The continuing game" in TEXT) == (RES.get("continuing") is not None)
    boards = 3 if RES.get("continuing") is not None else 2
    assert HTML.count("<svg") == 2 + 2 * boards  # rps, two charts a board, variants
    assert "best-response dynamics" in TEXT


def test_every_percentage_in_the_summary_comes_from_the_data():
    rnd = RES["random"]
    exact = f"{rnd['exact']['vs_best_response'][0]:.0%}"
    assert f"the exact solver wins {exact} of games on the random board" in TEXT
    lo = min(x["vs_best_response"][0] for x in rnd["learners"])
    assert f"the six learners win {lo:.0%}" in TEXT


def test_variant_bars_draw_a_bar_per_variant_and_a_dot_per_seed():
    run = {"mean": 0.2, "runs": [0.1, 0.2, 0.3], "vs_random": [0.9, 0.05, 0.05]}
    row = {"label": "A2C", "training": "standard", "baseline": run, "shared": run, "trimmed": run}
    svg = pg_report.variant_bars([row], 0.67)
    assert svg.count("<rect") == 3 and svg.count("<circle") == 9
    assert "exact solver 67%" in svg


def test_the_report_has_the_network_sharing_and_trimming_section():
    assert "Network sharing and trimming" in HTML and "Last 10 steps left out" in HTML
