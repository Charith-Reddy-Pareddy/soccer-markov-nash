PY ?= ./.venv/bin/python

.PHONY: help lint test test-all coverage report positions-pdf reproduce-core experiments phase benchmark dqn dqn-ablation dqn-random policy-gradient policy-gradient-warmstart policy-gradient-batch policy-gradient-batch-updates policy-gradient-architectures policy-gradient-ablation degeneracy distance-table pure-vs-mixed figures png gallery littman mixing reward blend occupancy tournament tournament4 tournament-deepdive numerics templates proof verify generalize tackle positional positions positions-matrix templates-n5 explorer-data site a10 clean

help:
	@echo "make lint         - ruff check (style + unused code)"
	@echo "make test         - fast test suite"
	@echo "make test-all     - full suite including slow tests (~10 min)"
	@echo "make coverage     - fast suite under coverage, report gaps"
	@echo "make experiments  - regenerate experiments/*.csv (board sweep is slow)"
	@echo "make phase        - regenerate experiments/phase_diagram.csv (~8 min)"
	@echo "make benchmark    - repeated-run timing + LP-call rate of the hybrid"
	@echo "make dqn          - multi-seed neural Nash-Q vs the exact solver (~16 min)"
	@echo "make dqn-ablation - Q-net width/depth sweep vs the exact solver (~17 min)"
	@echo "make dqn-random   - neural Nash-Q on the RANDOM move-order game (mixed equilibria) (~30 min)"
	@echo "make policy-gradient - self-play REINFORCE vs the exact solver's Nash-equilibrium check"
	@echo "make policy-gradient-warmstart - does pre-training onto the exact policy help self-play?"
	@echo "make policy-gradient-batch - does batching independent rollouts beat one correlated trajectory?"
	@echo "make policy-gradient-batch-updates - same comparison, matched by update count instead of total steps"
	@echo "make policy-gradient-architectures - separate vs. shared vs. partial-share policy nets"
	@echo "make policy-gradient-ablation - learned baseline vs. entropy bonus, tested separately"
	@echo "make degeneracy   - classify the 94 no-pure-saddle states: unique vs degenerate"
	@echo "make distance-table - no-pure-saddle counts by Manhattan player distance"
	@echo "make pure-vs-mixed  - greedy-pure vs Nash-mixed exploitability at the positions.md cases"
	@echo "make report       - regenerate docs/report.pdf from docs/report.html"
	@echo "make reproduce-core - regenerate only the headline figures/tables the paper cites (~5 min, not the full experiment/dqn/phase sweeps)"
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
	@echo "make tournament4  - the same table with plain 4x4 stage games (no stand action)"
	@echo "make tournament-deepdive - causal test: patch greedy at the mixed states only, gamma sweep (~80s)"
	@echo "make showcase     - 6 hand-picked mixed states: weighted-arrow diagrams + exact matrices"
	@echo "make positions    - player positions + stage game as a best-response-graph"
	@echo "make positions-matrix - minimax/always-left/random/best-response policy matrix"
	@echo "make positions-pdf - regenerate docs/positions.pdf from docs/positions.html"
	@echo "make explorer-data - regenerate docs/data/explorer.json for the interactive board explorer"
	@echo "make site         - build the React site (site/) into docs/ -- index.html + explorer.html"
	@echo "make numerics     - numerical-robustness analysis + figures"
	@echo "make templates    - print the mixed-state geometric templates"
	@echo "make templates-n5 - same, with STAND: does the 5th action change the template count?"
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

positions-pdf:
	@command -v chromium >/dev/null 2>&1 && BROWSER=chromium || BROWSER="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"; \
	"$$BROWSER" --headless --disable-gpu --no-pdf-header-footer \
	  --print-to-pdf=docs/positions.pdf --virtual-time-budget=10000 \
	  "file://$(CURDIR)/docs/positions.html"
	@echo "wrote docs/positions.pdf"

# The headline figures/tables the report and positions.pdf actually cite --
# not the full experiment sweep, the phase diagram, or the DQN trainings,
# which stay separate because they are slow and not needed to regenerate the
# paper's core claims. Run `make report` / `make positions-pdf` after this.
reproduce-core: benchmark verify templates degeneracy distance-table tournament4 tournament-deepdive positions pure-vs-mixed gallery
	@echo "core headline figures and tables regenerated -- run 'make report' / 'make positions-pdf' next"

phase:
	$(PY) scripts/phase_diagram.py

benchmark:
	$(PY) scripts/benchmark.py --repeats 5

dqn:
	$(PY) scripts/nash_dqn.py --seeds 5

policy-gradient:
	$(PY) scripts/policy_gradient.py --seeds 5

policy-gradient-warmstart:
	$(PY) scripts/policy_gradient_warmstart.py --seeds 3

policy-gradient-batch:
	$(PY) scripts/policy_gradient_batch.py --seeds 5

policy-gradient-batch-updates:
	$(PY) scripts/policy_gradient_batch.py --seeds 5 --match updates

policy-gradient-architectures:
	$(PY) scripts/policy_gradient_architectures.py --seeds 5

policy-gradient-ablation:
	$(PY) scripts/policy_gradient_ablation.py --seeds 3

dqn-ablation:
	$(PY) scripts/nash_dqn_ablation.py --seeds 2

dqn-random:
	$(PY) scripts/nash_dqn_random.py --seeds 5

degeneracy:
	$(PY) scripts/degeneracy.py

distance-table:
	$(PY) scripts/distance_table.py

pure-vs-mixed:
	$(PY) scripts/pure_vs_mixed_exploit.py

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

tournament4:
	$(PY) scripts/tournament4.py
	$(PY) scripts/figures_png.py tournament4

tournament-deepdive:
	$(PY) scripts/tournament_deepdive.py

showcase:
	$(PY) scripts/showcase.py
	$(PY) scripts/figures_png.py showcase

positions:
	$(PY) scripts/positions.py
	$(PY) scripts/figures_png.py positions

positions-matrix:
	$(PY) scripts/positions_policy_matrix.py

explorer-data:
	$(PY) scripts/explorer_data.py

site:
	cd site && npm install && npm run build

numerics:
	$(PY) scripts/numerics.py --move-orders deterministic random --figures

templates:
	$(PY) scripts/templates.py

templates-n5:
	$(PY) scripts/templates.py --n-actions 5

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
