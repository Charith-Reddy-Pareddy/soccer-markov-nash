"""Put the repo root on sys.path at collection time.

``soccer_nash`` is always importable (installed via ``pip install -e .``),
but ``scripts/`` is a plain top-level directory, not part of that package --
it only becomes importable when the repo root happens to be on ``sys.path``.
Running ``python -m pytest`` adds the current directory automatically, which
is why this went unnoticed locally; CI runs the bare ``pytest`` entry point,
which does not.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
