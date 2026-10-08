import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "pg_report", pathlib.Path(__file__).resolve().parent.parent / "scripts" / "pg_report.py")
pg_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg_report)


def test_bar_chart_has_one_bar_per_learner_and_one_dot_per_seed():
    svg = pg_report.bar_chart([("a", [0.2, 0.4]), ("b", [0.9])])
    assert svg.count("<rect") == 2 and svg.count("<circle") == 3


def test_stacked_chart_segments_fill_each_panel():
    svg = pg_report.stacked_chart(
        [("a", [[0.5, 0.25, 0.25], [1.0, 0.0, 0.0]])], ["left", "right"])
    assert svg.count("<rect") == 6 and "left" in svg and "right" in svg


def test_win_tie_loss_helper_averages_over_seeds():
    rows = [{"row_win_vs_x": "1", "row_tie_vs_x": "0", "row_loss_vs_x": "0"},
            {"row_win_vs_x": "0", "row_tie_vs_x": "0", "row_loss_vs_x": "1"}]
    assert pg_report.wtl(rows, "row", "x") == [0.5, 0.0, 0.5]


def test_the_committed_results_cover_every_learner_in_the_report():
    rows = pg_report.read(pg_report.EXP / "pg_finite_a10.csv") + pg_report.read(
        pg_report.EXP / "dqn_finite_a10.csv")
    for _, algo, mode in pg_report.LEARNERS:
        assert pg_report.pick(rows, algo, mode), (algo, mode)


def test_fp_br_table_reports_every_saved_run_with_a_verdict_that_matches_the_data():
    html = pg_report.fp_br_html()
    assert html.count("<tr>") == 1 + len(pg_report.glob.glob(str(pg_report.EXP / "pg_fp_br_*.csv")))
    for run in pg_report.glob.glob(str(pg_report.EXP / "pg_fp_br_*.csv")):
        rows = pg_report.read(run)
        by_round = {int(r["round"]): float(r["exploitability"]) for r in rows}
        rose = by_round[max(by_round)] > by_round[min(by_round)]
        if not rose:
            assert "higher than at the first" not in html


def test_policy_table_has_one_row_per_learner_for_each_state():
    html = pg_report.policy_html()
    assert html.count("<table") == 4
    assert html.count("Exact solver") == 4
    assert html.count("REINFORCE, self-play") == 4
