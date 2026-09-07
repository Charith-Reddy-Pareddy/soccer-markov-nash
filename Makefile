PY ?= ./.venv/bin/python

.PHONY: help test test-all report experiments figures a10 clean

help:
	@echo "make test         - fast test suite"
	@echo "make test-all     - full suite including slow tests (~10 min)"
	@echo "make experiments  - regenerate experiments/*.csv (board sweep is slow)"
	@echo "make report       - regenerate docs/report.pdf from docs/report.html"
	@echo "make a10           - regenerate the A10 Part 2 + competition artifacts"
	@echo "make figures      - redraw docs/figures/*.svg from the renderer"

test:
	$(PY) -m pytest -m "not slow" -q

test-all:
	$(PY) -m pytest -q

experiments:
	$(PY) scripts/experiments.py baseline
	$(PY) scripts/experiments.py gamma
	$(PY) scripts/experiments.py tolerance
	$(PY) scripts/experiments.py goalmouth
	$(PY) scripts/experiments.py board

report:
	@command -v chromium >/dev/null 2>&1 && BROWSER=chromium || BROWSER="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"; \
	"$$BROWSER" --headless --disable-gpu --no-pdf-header-footer \
	  --print-to-pdf=docs/report.pdf --virtual-time-budget=10000 \
	  "file://$(CURDIR)/docs/report.html"
	@echo "wrote docs/report.pdf"

figures:
	$(PY) scripts/render.py 0,2,6,2,0 -o docs/figures/kickoff.svg
	$(PY) scripts/render.py --trajectory -o docs/figures/trajectory.svg

a10:
	$(PY) scripts/a10_part2.py --out results/
	$(PY) scripts/a10_competition.py --out results/

clean:
	rm -rf results/ .pytest_cache __pycache__ */__pycache__
