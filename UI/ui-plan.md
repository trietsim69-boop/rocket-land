# UI Plan: animated click-to-land app

Goal: make the Gradio app feel like a game, with a styled space scene, an animated rocket (flame, side thrusters, landing dust, crash explosion), and a launch countdown that ends exactly when the rocket appears. No app code changes until this plan is approved.

## Where the app is now (`app.py`)
- Server runs one episode, renders every frame with `env.render()`, stamps a SCORE overlay, encodes a GIF (1.9–2.4 s total on CPU), shows it in a second `gr.Image`.
- Click target is a static render of the empty scene with a grey box; agent picked from a dropdown.
- During the ~2 s wait the user sees only Gradio's spinner. That dead time is where a countdown fits.

## Key decision: replay on a canvas in the browser, physics stays on the server
The server keeps running the real Box2D episode with the real policy (no change to the rule "policy runs server-side", no sim mismatch). It stops rendering pixels. It returns a small **trajectory**: per step the lander's x, y, angle, leg contacts, action and running score, plus the terrain once at startup. The browser draws it on a `<canvas>` with custom art at 50 FPS.

Why this beats improving the GIF:
| | Better GIF (server) | Canvas replay (browser) |
|---|---|---|
| Animations (flame, particles, dust, explosion) | Possible in PIL, but slow and baked in | Native, smooth, cheap |
| Server time per click | ~2 s (render + encode) | expected well under 0.5 s (no render, no encode); measure in Step 1 |
| Payload | ~1–3 MB GIF | ~1000 steps × 6 numbers ≈ 30–60 KB JSON |
| Countdown, hover preview, click on the same scene | No (two separate images) | Yes, one scene |
| HF free CPU, one episode at a time | Each click blocks ~2 s | Queue clears ~4× faster |
| New dependencies | none | none (plain JS canvas, no libraries) |

Cost: about 300–400 lines of JavaScript in one new file `scene.js`, and `land()` returns data instead of a GIF path. The README/results GIFs come from `plot.py` and are not affected.

Feasibility risk: getting custom JS into Gradio 6.30 and sending the click back to Python. Step 0 settles that before anything else is built.

## The countdown (recommended: yes, and it is cheap)
The countdown runs **in parallel** with the server, and launch is **gated on the real data**:

1. Click → a faint ghost rocket appears at the clicked spot (already clamped client-side to the grey box), the big readout says **T-3**, and the request goes out at that same instant.
2. Ticks every ~0.6 s: 3, 2, 1.
3. At zero: if the trajectory has arrived → "IGNITION", the ghost turns solid, the flame lights, the replay starts. If not (busy Space, queue) → hold on "1" with a pulsing "fuelling…" (and the queue position if Gradio reports one), then launch the moment data lands.

So the rocket never appears before the episode exists, and the user never waits longer than they would anyway, except for the ~1.5–2 s of show on a fast server. A countdown whose numbers track the exact remaining compute is not possible (the server time isn't known in advance, and a progress-driven count would jump), so "gated on the data" is the honest version. Tick length gets set after Step 1 measures real server time. The same short countdown plays when the agent changes and the last click replays. With `prefers-reduced-motion` it becomes plain text with no animation.

## Visual design
- **Scene:** dark space background with a twinkling starfield (2 layers, slight parallax), the real terrain polygon from the env filled as grey moon rock with a lit rim, the pad between the flags with blinking beacon lights. Terrain is drawn from the env's own coordinates so it matches the physics exactly.
- **Rocket:** drawn from the env's lander polygon and leg boxes (same shape and size as the physics body), with a small window and a shade. Legs flash when they touch the ground (contact flags).
- **Engines:** main flame (action 2) is a flickering cone plus exhaust particles; side thrusters (actions 1 / 3) give short puffs on the right side of the hull.
- **Trail:** faint dotted path from start to finish, kept after the run so the user sees the route.
- **Outcomes:**
  - landed: dust ring kicked up from the pad, a "LANDED" card, score counts up to the final value.
  - crashed: explosion particles, a small screen shake, a "CRASHED" card.
  - timeout: rocket dims, "OUT OF TIME (1000 steps)" card.
- **HUD (top left):** SCORE (keeps today's green ≥ 0 / red < 0), altitude, speed, tilt, fuel burnt, all from the trajectory.
- **Hover:** crosshair and ghost rocket inside the grey box; outside it the cursor shows where the click would be clamped to (server still clamps, the client view is a hint only).
- **Agent picker:** 4 styled radio cards (name + one-line reward description) instead of the dropdown. Same values, still validated server-side.
- **Controls:** Replay button and a 1× / 2× speed toggle.
- **Layout:** one canvas (600×400 logical, scales to the column width; click mapping uses the canvas's displayed size so it works on phones), result line underneath, dark theme via `gr.themes` + a small CSS block.

## Steps
Each step is small and checked before the next. Ticks go in this file.

### Step 0: Spike the Gradio bridge (S)
Prove in a throwaway file: a canvas inside Gradio 6.30 (try `gr.HTML` with its JS hooks first; fallback is a `<script>` in `Blocks(head=...)` + a hidden `gr.Textbox` that JS writes the click into and Python listens to) can (a) receive JSON from a Python event, (b) send a click (x, y) back to Python.
- [ ] Click on the canvas reaches a Python handler; Python's reply reaches the canvas
- [ ] Works in the HF Spaces iframe (test locally with `share=False`, then on the Space in Task 11)

### Step 1: Trajectory instead of GIF (S)
`land(x, y, agent)` returns `{states: [[x, y, angle, legL, legR, action, score], …], outcome, score, start, clamped}`; `terrain()` returns the terrain polygon + pad + flags once.
- [ ] Outcome and score identical to today's `land()` for 5 fixed clicks × 4 agents
- [ ] Server time per click measured (target < 0.5 s); payload < 100 KB
- [ ] One assert in `test_shaping.py`: trajectory length equals episode length and last score equals the summed true reward

### Step 2: Static scene + plain replay (S)
`scene.js` draws stars, terrain, pad, rocket from a trajectory at 50 FPS (time-based, so it plays at true speed on any machine).
- [ ] Replay path visually matches today's GIF for the same click/agent (side-by-side check)

### Step 3: Click, hover, countdown (S)
- [ ] Clicking the canvas launches a run; hover ghost and clamp hint work
- [ ] Countdown ends exactly when the rocket appears; holds on "1" when the server is slow (test with an artificial `time.sleep(3)`)
- [ ] Agent change replays the last click with the countdown

### Step 4: Effects (S)
Flame, thrusters, particles, dust, explosion, shake, outcome cards, HUD, trail.
- [ ] Each outcome (landed / crashed / timeout) shows its own ending; `hack` timeout case found and checked
- [ ] Stays smooth (no visible stutter) on a mid laptop; particle count capped

### Step 5: Polish (XS)
Dark theme, agent cards, replay + speed buttons, reduced-motion, phone width.
- [ ] Usable on a phone-width window; reduced-motion disables shake, particles, countdown animation
- [ ] `python app.py`: 5 clicks × 4 agents by hand, as in Task 10

### Stretch (only if wanted): race mode
Run all 4 agents from the same click and replay them together in different colours. Cheap with trajectories (4 × ~0.3 s server time), and it shows the reward-shaping story better than switching the dropdown.

## Files
- `app.py`: `land()` returns a trajectory, `terrain()`, new layout; GIF code removed
- `scene.js` (new): all drawing, countdown, particles, click handling
- `test_shaping.py`: one trajectory assert
- `CLAUDE.md`: add `scene.js` to the file list
- No new dependencies.

## Order vs. deploy (Task 11)
Task 11 is still waiting on the HF Space name and branch decision. Two options: deploy today's app first so a public link exists, then ship this UI as an update (safer), or build this first and deploy once. Recommend deploying first; the Step 0 iframe check also needs the Space.

## Open questions
- Race mode: wanted, or keep one agent at a time?
- Any colour or style reference (realistic NASA look vs. arcade / pixel style)? Default in this plan: clean dark space, soft glow, not pixel art.
