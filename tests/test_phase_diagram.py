"""Helpers behind scripts/phase_diagram.py."""

import importlib.util
import pathlib
import xml.etree.ElementTree as ET

_spec = importlib.util.spec_from_file_location(
    "phase_diagram",
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "phase_diagram.py",
)
phase = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phase)


def test_centered_goal_rows():
    assert phase.centered_goal_rows(5, 1) == (2,)
    assert phase.centered_goal_rows(5, 3) == (1, 2, 3)
    assert phase.centered_goal_rows(7, 2) == (2, 3)


def test_configs_respect_the_area_cap():
    full = phase.configs(quick=False)
    for w, h in full:
        assert w * h <= 45
        assert 3 <= w <= 11 and 3 <= h <= 9
    assert phase.configs(quick=True) != full
    assert (11, 3) in full and (3, 9) in full


def test_heat_is_pale_at_zero_and_darker_higher():
    assert phase._heat(0.0, 0.1) == "#eef2ec"
    light = phase._heat(0.02, 0.12)
    dark = phase._heat(0.12, 0.12)
    assert light != dark
    # darker = lower channel values
    assert int(dark[1:3], 16) < int(light[1:3], 16)


def test_write_heatmap_is_valid_svg(tmp_path):
    rows = [
        {"width": 3, "height": 3, "goal_width": 1, "mixed_fraction": 0.0},
        {"width": 3, "height": 5, "goal_width": 1, "mixed_fraction": 0.0},
        {"width": 3, "height": 5, "goal_width": 2, "mixed_fraction": 0.09},
    ]
    out = tmp_path / "h.svg"
    phase.write_heatmap(rows, out)
    root = ET.fromstring(out.read_text())
    assert root.tag.endswith("svg")
