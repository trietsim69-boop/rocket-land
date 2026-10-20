# UI Plan: styled click-to-land app (static design, data-driven replay)

Goal: give the Gradio app the look in `UI/designs/` (dark space scene, rocket art, HUD, agent cards, end cards) with as little code as possible. Animation is optional: the only thing that moves is the replay itself (the rocket stepping through the episode). Designs are reference only (they use a design tool's `support.js` / `<x-dc>` wrapper); copy their SVG fragments, don't load the files.

## Decision: server runs the episode, the browser draws it
The server keeps running the real Box2D episode with the real policy (no sim mismatch). It stops rendering pixels and returns a **trajectory**: per step the lander's pixel x, y, angle, leg contacts, action, running true score, plus the terrain once. The browser draws it as SVG using the design's markup and moves the rocket with one `transform` per step.

**Bridge (verified in the Step 0 spike):** one `gr.HTML` with `html_template` (the design's SVG), `js_on_load` (our JS) and `server_functions=[trajectory]`. JS calls `await server.trajectory({x, y, agent})`.
- The Python function receives exactly one argument. The frontend sends `server.fn(a)` as `a` and `server.fn(a, b)` as `[a, b]`, so always pass one object → Python gets a dict.
- Server functions go to `/component_server` and **skip Gradio's queue**, so `demo.queue` limits don't apply → guard the episode with a `threading.Lock` (one at a time on the free CPU). Queue position can't be shown.
- Only functions listed in `server_functions` are callable; anything else returns 404 (checked).

## Scope

| Keep (static, data-driven) | Dropped (animation) |
|---|---|
| Dark scene, fixed starfield, moon disc, **real terrain from the env** in the design's colours, pad, flags, beacon dots | Twinkle, parallax, blinking beacons |
| Dashed "LAUNCH ZONE · CLICK TO DROP" box (x ∈ [−0.6, 0.6], y ∈ [0.6, 1.4]); ghost rocket at the click while waiting | Countdown T-3 … IGNITION, pulsing ring, "fuelling" hold |
| Rocket moved along the trajectory; flame / left / right puff shown per action; feet turn green on leg contact | Flicker, exhaust/dust particles |
| Dotted trail, kept after the run | — |
| HUD: SCORE (green ≥ 0 / red < 0, always signed), ALT, SPEED, TILT, FUEL; STEP n / 1000 | — |
| End cards LANDED / CRASHED / OUT OF TIME with final score; static burst marks at a crash; dimmed rocket on timeout | Explosion, screen shake, score count-up |
| 4 agent cards as real radio inputs (name, one-line reward, "N% land in zone"); Replay button; 1× / 2× speed | Hover crosshair + clamp hint (add later if wanted) |
| Status line under the scene ("pbrs is descending from (x, y). Main engine on.", then the outcome) | — |

Card percentages are read at startup from `results/heatmap_*.csv` for the shipped model's run (`REWARD_RUNS` map: default→s1, sparse→s2, pbrs→s2, hack→s1, inner region |x| ≤ 0.6), never hand-typed.

## Steps

### Step 0: Spike the bridge (XS)
- [x] Python half: `server_functions` function called over HTTP receives a dict, returns JSON (200); non-server function → 404
- [ ] JS half: in a browser, a `gr.HTML` button calls `server.trajectory(...)` and shows the reply (checked with the real app in Step 2)

### Step 1: Trajectory + terrain (S)
`trajectory({x, y, agent})` → `{states: [[px, py, angle, legL, legR, action, score], …], outcome, score, start, clamped}`; `terrain()` → terrain polygon, pad x1/x2/y, flags (render pixels, seed `APP_SEED`). Same validation as today: clamp click, reject unknown agent, ≤ 1000 steps, lock.
- [x] Outcome and score identical to the current GIF `land()` for 5 clicks × 4 agents (compared against the same episode without rendering: the GIF `land()` itself hit a MemoryError, since a 1000-step episode keeps ~720 MB of raw frames; gone once the GIF code is removed in Step 2)
- [x] Server time per click measured; payload < 100 KB (≤ 0.44 s, ≤ 19 KB, vs ~2 s for the GIF)
- [x] `test_shaping.py`: trajectory length = episode length, last score = summed true reward (replays the recorded actions on a fresh env)

### Step 2: Scene + replay (S)
`gr.HTML` template = design SVG (scene, launch zone, rocket group with flame/puff/feet sub-groups, HUD, end card). JS: click → map displayed size to 600×400 → ghost → `server.trajectory` → replay with `requestAnimationFrame`, time-based at 50 FPS × speed.
- [ ] Replay path matches the current GIF for the same click and agent
- [ ] Each outcome shows its end card (find a `hack` timeout case)
- [ ] Click mapping works when the scene is scaled (narrow window)

### Step 3: Cards + controls + page (XS)
Agent radio cards, Replay, 1× / 2×, header, dark page CSS, Space Grotesk / JetBrains Mono from Google Fonts (system-font fallback). Agent change replays the last click.
- [ ] `python app.py`: 5 clicks × 4 agents by hand, as in Task 10; keyboard can pick an agent

## Files
- `app.py`: `trajectory()`, `terrain()`, `REWARD_RUNS`, the `gr.HTML` page; GIF code removed
- `test_shaping.py`: one trajectory assert
- `CLAUDE.md`: file list unchanged unless the JS moves to its own file
- No new dependencies

## Order vs. deploy (Task 11)
Task 11 still waits on the HF Space name and branch decision. Deploying the current app first gives a public link sooner; this UI then ships as an update.

## Stretch (only if wanted)
- Race mode: all 4 agents from one click, replayed together in different colours.
- The dropped animations, one at a time.
