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
