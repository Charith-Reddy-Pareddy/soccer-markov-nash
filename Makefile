PY ?= ./.venv/bin/python

.PHONY: help lint test test-all coverage report experiments phase benchmark dqn figures templates proof a10 clean

help:
	@echo "make lint         - ruff check (style + unused code)"
	@echo "make test         - fast test suite"
	@echo "make test-all     - full suite including slow tests (~10 min)"
	@echo "make coverage     - fast suite under coverage, report gaps"
	@echo "make experiments  - regenerate experiments/*.csv (board sweep is slow)"
	@echo "make phase        - regenerate experiments/phase_diagram.csv (~8 min)"
	@echo "make benchmark    - repeated-run timing + LP-call rate of the hybrid"
	@echo "make dqn          - multi-seed neural Nash-Q vs the exact solver (~5 min)"
	@echo "make report       - regenerate docs/report.pdf from docs/report.html"
	@echo "make a10          - regenerate the A10 Part 2 + competition artifacts"
	@echo "make figures      - redraw docs/figures/*.svg from the renderer"
	@echo "make templates    - print the mixed-state geometric templates"
	@echo "make proof        - run the single-cell pure-saddle certificate (~4 min)"

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest -m "not slow" -q

test-all:
	$(PY) -m pytest -q

coverage:
	$(PY) -m coverage run -m pytest -q -m "not slow"
	$(PY) -m coverage report

experiments:
	$(PY) scripts/experiments.py baseline
	$(PY) scripts/experiments.py undiscounted
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

phase:
	$(PY) scripts/phase_diagram.py

benchmark:
	$(PY) scripts/benchmark.py --repeats 5

dqn:
	$(PY) scripts/nash_dqn.py --seeds 5

figures:
	$(PY) scripts/render.py -o docs/figures/kickoff.svg
	$(PY) scripts/render.py --trajectory -o docs/figures/trajectory.svg

templates:
	$(PY) scripts/templates.py

proof:
	$(PY) scripts/onecell_proof.py

a10:
	$(PY) scripts/a10_part2.py --out results/
	$(PY) scripts/a10_competition.py --out results/

clean:
	rm -rf results/ .pytest_cache .ruff_cache .coverage htmlcov __pycache__ */__pycache__
