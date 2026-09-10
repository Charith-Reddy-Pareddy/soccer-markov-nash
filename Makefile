PY ?= ./.venv/bin/python

.PHONY: help lint test test-all coverage report experiments phase benchmark dqn figures png gallery littman mixing reward blend occupancy tournament numerics templates proof verify generalize tackle positional a10 clean

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
	@echo "make figures      - redraw docs/figures/*.svg (+ their PNGs)"
	@echo "make png          - rasterize every docs/figures/*.svg to docs/figures/png/ (for slides / GitHub)"
	@echo "make gallery      - redraw the visualization gallery + docs/figures/png/ (docs/gallery.html)"
	@echo "make littman      - reproduce Littman 1994 Figure 2 (the stand action)"
	@echo "make mixing       - how mixed the mixed states are, and the value of mixing"
	@echo "make reward       - the win vs rate reward objectives (goal reset)"
	@echo "make blend        - sweep the move-resolution rule (deterministic -> random)"
	@echo "make occupancy    - equilibrium-path occupancy of the mixed states"
	@echo "make tournament   - reproduce Littman 1994 Table 3 (minimax vs greedy robustness)"
	@echo "make numerics     - numerical-robustness analysis + figures"
	@echo "make templates    - print the mixed-state geometric templates"
	@echo "make proof        - single-cell pure-saddle certificate (~4 min)"
	@echo "make verify       - goal-width mixed-equilibrium certificate + robustness (~90 s)"
	@echo "make generalize   - how far the goal-width switch holds: scale, goal shape, slip (~9 min)"
	@echo "make tackle       - the project's own 'tackle' collision rule + rule-fingerprint panel"
	@echo "make positional   - the deterministic game's pure memoryless equilibrium"

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
	$(PY) scripts/figures_png.py kickoff
	$(PY) scripts/figures_png.py trajectory

png:
	$(PY) scripts/figures_png.py

gallery:
	$(PY) scripts/gallery.py
	$(PY) scripts/story.py
	$(PY) scripts/figures_png.py

littman:
	$(PY) scripts/littman.py

mixing:
	$(PY) scripts/mixing.py

reward:
	$(PY) scripts/reward.py

blend:
	$(PY) scripts/blend.py

occupancy:
	$(PY) scripts/occupancy.py

tournament:
	$(PY) scripts/tournament.py

numerics:
	$(PY) scripts/numerics.py --move-orders deterministic random --figures

templates:
	$(PY) scripts/templates.py

proof:
	$(PY) scripts/onecell_proof.py

verify:
	$(PY) scripts/verify_mechanism.py

generalize:
	$(PY) scripts/generalize.py

tackle:
	$(PY) scripts/tackle.py

positional:
	$(PY) scripts/positional.py

a10:
	$(PY) scripts/a10_part2.py --out results/
	$(PY) scripts/a10_competition.py --out results/

clean:
	rm -rf results/ .pytest_cache .ruff_cache .coverage htmlcov __pycache__ */__pycache__
