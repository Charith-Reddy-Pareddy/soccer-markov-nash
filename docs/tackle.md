# The `tackle` collision rule

`scripts/tackle.py` (`make tackle`), `experiments/tackle.csv`,
`soccer_nash/game.py`.

A collision rule of this project's own design, added to answer the meeting ask
to build the game's transition model rather than inherit Littman's.

## The rule

The defender **commits to a challenge** by moving onto the ball carrier's cell.

- **No challenge** (the defender moves anywhere else): the turn resolves by the
  deterministic rule.
- **Challenge**: a duel.
  - With probability `tackle_prob` the challenge **wins the ball**. The carrier
    is shoved one cell along the defender's line of approach (clamped at a
    wall); the defender takes the vacated cell.
  - With probability `1 - tackle_prob` the challenge **fails**. The defender is
    bounced back to its own cell. The carrier keeps the ball and completes its
    move only if it was already running into open space; otherwise it is held
    up.

"Dive in or contain" for the defender, and "hold the ball or break away" for
the carrier, are now a real gamble against each other.

## What it does to the equilibrium

- **Mixed stage games appear at any goal width, single cell included.** On a
  5x4 board with a one-cell goal the deterministic and random rules give 0
  mixed stage games; `tackle` at `tackle_prob = 0.5` gives 82, and the count
  climbs from 56 at `tackle_prob = 0.1` to 106 at `0.9`. Like `slip`, and
  unlike Littman's random move order, the guess here does not depend on the
  goal geometry. It is the duel, on any square where a challenge is reachable,
  that has no safe pure answer. This is a third confirmation that the
  goal-width switch is specific to the move-order rule
  ([generalize.md](generalize.md), [result.md](result.md)).
- **The mixed region is the "dive-in zone".** Every `tackle`-mixed state has the
  defender within one to three moves of the carrier (distance 1: 20 states,
  distance 2: 28, distance 3: 8), where a committed challenge is on or near the
  table. Far from the defender the carrier has a safe pure move.
- **More challenge power, more mixing.** The count rises monotonically with
  `tackle_prob`: a stronger tackle makes the "should I stay near the defender"
  question sharper for the carrier across more of the board. At `tackle_prob = 0`
  the challenge always fails, the defender never has a reason to try it, and the
  game is pure everywhere again.

## The rule fingerprint

`docs/figures/gallery/rule_fingerprints.svg` puts the six collision rules
side by side: the same board, the defender pinned at the goal mouth, every
carrier cell shaded by how far its stage game is from a pure saddle. Each rule
leaves a distinct mark. `deterministic` and `coinflip` are blank (pure
everywhere). `random` and `blend` light up a thin band next to the goal.
`slip` and `tackle` light up a broad region around the defender.
