# Todo: Lunar Lander Reward Shaping + Click-to-Land App

## Task 1: Env runs; heuristic lander scores ≥ 200
**Description:** Install `gymnasium[box2d]`, `stable-baselines3`, `matplotlib`, `imageio`, `gradio`; pin in `requirements.txt`. Run Gymnasium's own `heuristic()` controller to prove physics + scoring work.

**Acceptance criteria:**
- [x] `requirements.txt` installs cleanly in a fresh venv
- [x] Heuristic lander mean true score ≥ 200 over 10 seeded episodes (265.4, seeds 0–9)

**Verification:**
- [x] `python -c "import gymnasium as g; from gymnasium.envs.box2d.lunar_lander import demo_heuristic_lander as d; print(d(g.make('LunarLander-v3')))"` ≥ 200

**Dependencies:** None
**Files:** `requirements.txt`
**Scope:** XS

## Task 2: Sparse-reward wrapper
**Description:** `SparseReward(gym.Wrapper)` in `shaping.py`: computes `Φ_env(obs)` with the exact formula from `lunar_lander.py`; returns `r − (Φ_env(s') − Φ_env(s))` on non-terminal steps, `r` unchanged on termination.

**Acceptance criteria:**
- [x] Non-terminal sparse reward equals −fuel cost only (−0.3·main − 0.03·side)
- [x] Terminal reward is ±100 unchanged

**Verification:**
- [x] `python test_shaping.py` — asserts decomposition over 5k random-policy steps (tol 1e-4)

**Dependencies:** 1
**Files:** `shaping.py`, `test_shaping.py`
**Scope:** S

## Task 3: End-to-end slice on `default` reward
**Description:** `train.py --reward default --seed 0 --steps N`: `Monitor` inside reward wrapper (logs true score), SB3 PPO with RL-Zoo hyperparams, periodic 10-episode deterministic eval → `runs/{reward}_s{seed}/eval.csv` (step, episodes_so_far, mean_true_score, success_rate) + `model.zip`. Also save a model checkpoint at each eval (`ckpt_{episodes}.zip`) so the progression strip can be rendered later.

**Acceptance criteria:**
- [x] 1M-step run produces CSV + model (300k was too short for RL-Zoo hparams: −71, 10% success)
- [x] Scores rise above random-policy level (≈ −150 to −200) (final 263.5, 100% success)
- [x] Monitor return under `SparseReward` equals unwrapped return for same seed/actions

**Verification:**
- [x] `python train.py --reward default --seed 0 --steps 1000000`; inspect CSV

**Dependencies:** 2
**Files:** `train.py`
**Scope:** S

### Checkpoint A
- [x] `test_shaping.py` passes
- [x] Default run learns
- [ ] Review with human before spending compute

## Task 4: `sparse` condition
**Description:** Wire `--reward sparse`. Same steps/seed as Task 3.

**Acceptance criteria:**
- [x] Curve recorded on true score
- [x] One-line note on observed behavior: collapses to "never fire, free-fall crash" (action 0 every step) by ~300k; final −118.2, 0% success. Fuel is the only dense signal, so not firing beats every non-landing policy.

**Verification:**
- [x] `python train.py --reward sparse --seed 0 --steps 1000000`

**Dependencies:** 3
**Files:** `train.py`
**Scope:** XS

## Task 5: Potential-based shaping (`pbrs`)
**Description:** `PotentialShaping` on top of `SparseReward`: adds `γΦ(s') − Φ(s)` with my own Φ (closer to pad, slower, upright, leg contact; own weights). γ from PPO config; Φ(terminal) = 0. Optional `--time-penalty c` (−c per step, like the reel); keep c small and check it doesn't teach the agent to crash early to end the episode.

**Acceptance criteria:**
- [x] Telescoping: Σ γ^t F_t = γ^T Φ(s_T) − Φ(s_0) over a full episode
- [x] Learns faster than `sparse` at equal steps (GAE λ=0.9, now the default): pbrs first ≥200 at 606k, 737k: 221 / 90%, final 245 / 100%; sparse first ≥200 at 950k, 737k: 17 / 10%, final 155 / 80% (slow, near-hover landings). At λ=0.98 both pbrs and sparse collapse to free-fall (runs/*_lam0.98), and `default` prefers 0.98 (263 / 100% vs 134 / 70% at 0.9).

**Verification:**
- [x] `python test_shaping.py` (telescoping assert)
- [x] `python train.py --reward pbrs --seed 0 --steps 1000000`

**Dependencies:** 4
**Files:** `shaping.py`, `test_shaping.py`, `train.py`
**Scope:** S

## Task 6: Non-potential shaping (`hack`)
**Description:** `NaiveBonus` on `SparseReward`: e.g. +0.5 every step within a radius of the pad. Not potential-based → optimal policy may change.

**Acceptance criteria:**
- [x] True score + success rate recorded: radius 0.5 → final 85 / 40%, unstable (688k: 200 / 80%, 884k: 16 / 0%). Radius 0.2 (runs/hack_s0_r0.2) → −119 / 0%.
- [x] Rollout shows hovering/stalling, or a note on why not: radius 0.5 hacks. Final model lingers on/near the pad instead of settling: one episode keeps its legs scraping the ground for 657 steps until the 1000-step timeout (hack reward 299, true 121) vs a clean landing episode (hack 249, true 209), i.e. not landing pays more under the hack reward. Radius 0.2 never found the bonus (0 in-radius steps at any checkpoint) and just stalled at spawn height to avoid the −100 crash.

**Verification:**
- [x] `python train.py --reward hack --seed 0 --steps 1000000`; watch one rollout

**Dependencies:** 4
**Files:** `shaping.py`, `train.py`
**Scope:** S

### Checkpoint B
- [x] All 4 conditions run on seed 0, each with a one-line explanation
- [x] Decide step budget + seeds for the random-start sweep: 2M steps × seeds 0–2 (pbrs plateaus ~1.3M)

## Task 7: Custom start position
**Description:** Small subclass/wrapper (`env.py`): `reset(options={"start": (x, y)})` moves lander body + both legs by the same offset after the normal reset, zeroes velocities, returns a recomputed observation. No start → uniform random start in an allowed sky region (above terrain, inside viewport, below spawn ceiling). Invalid coords → clamped into the region.

**Acceptance criteria:**
- [x] Lander spawns within 1e-3 of requested `(x, y)`; observation matches the new position (`(x, y)` = obs coords, pad at (0, 0); region x ∈ [−0.8, 0.8], y ∈ [0.6, 1.4])
- [x] With no-op actions from 10 different starts, the lander falls without joint glitches (legs stay attached): joint gap 0 until first contact; crash impact stretches joints in the stock env too (up to 0.43), so that's excluded
- [x] Out-of-range coords clamp, never raise (incl. NaN / ±inf)

**Verification:**
- [x] `python test_shaping.py` (start-position asserts) + render one no-op drop to GIF and look

**Dependencies:** 3
**Files:** `env.py`, `test_shaping.py`, `train.py` (`--random-start` flag)
**Scope:** S

## Task 8: Random-start sweep + heatmap
**Description:** Train each condition × seeds with random starts (background, sequential). `plot.py`: mean ± std true-score curves (200 line), final score/success table, and a success-rate heatmap over a start-position grid (e.g. 8×5, 10 episodes per cell) per condition.

**Acceptance criteria:**
- [x] All runs complete; `results/learning_curves.png`, `results/heatmap_{reward}.png`, `results/summary.md` generated from saved data only (4 conditions × seeds 0–2, 2M steps, random starts, λ=0.9; 4 parallel runs ≈ 50 min)
- [x] Best agent (expected `pbrs`) ≥ 80% success over the clickable region: on the full 8×5 grid `hack` 85% ± 6, `default` 80% ± 10, `pbrs` 74% ± 4, `sparse` 58% ± 18. The grid gives the extreme columns (x = ±0.8) a quarter of the weight, where pbrs is weakest (left edge 33–50%). Ad-hoc check on 60 uniform random starts (not in results/): pbrs 54/60 landed, default 51, hack 51, sparse 38; pbrs also has the lowest fuel and shortest episodes. Mean true score plateaus at 100–150 for all, so "first ≥ 200" is never hit on random starts. Clickable region → Checkpoint C.
- [x] `results/progression.gif`: rollouts from checkpoints side by side, above the learning curve with matching markers. Uses untrained / ep ≈3000 / final (8821) of `pbrs_s0_rand`: on random starts ep 500 and 3000 still crash everywhere, so 1/500/3000 showed no landing.

**Verification:**
- [x] `python plot.py`; open PNGs

**Dependencies:** 5, 6, 7
**Files:** `train.py`, `plot.py`
**Scope:** M

### Checkpoint C
- [x] Heatmaps reviewed; clickable region decided (where best agent succeeds): app clamps clicks to x ∈ [−0.6, 0.6], y ∈ [0.6, 1.4] (training region stays x ±0.8)
- [x] Pick which agents the app exposes (default: `pbrs`, `sparse`, `hack`): all 4, best seed by heatmap success inside the trimmed region → `models/{reward}.zip`: default s1 (94%), sparse s2 (84%), pbrs s2 (84%), hack s1 (97%). Picked on the same grid they're scored on, so these are app picks, not unbiased results (`results/summary.md` stays mean over seeds).

## Task 9: `land(x, y, agent)` inference
**Description:** In `app.py` (or `env.py`): load model, run one deterministic episode from `(x, y)`, return frames + outcome (`landed` / `crashed` / `timeout`) + true score. Each frame gets a running SCORE overlay (green ≥ 0, red < 0). Models loaded once at startup.

**Acceptance criteria:**
- [x] Returns within ~2 s on CPU (1.9–2.4 s incl. GIF encode)
- [x] Outcome matches the episode's terminal reward

**Verification:**
- [x] `python -c "from app import land; print(land(0.3, 0.8, 'pbrs')[1:])"`

**Dependencies:** 8
**Files:** `app.py`
**Scope:** S

## Task 10: Gradio click-to-land app
**Description:** Background image of the empty scene; `gr.Image(...).select` gives click pixel → map to env coords → `land()` → show GIF + outcome text. Dropdown chooses agent. Queue so runs don't overlap.

**Acceptance criteria:**
- [x] Clicking anywhere in the sky plays a landing from that spot
- [x] Clicks outside the allowed region are clamped (and the user is told)
- [x] Switching agent from same click shows different behavior (dropdown change replays the last click)

**Verification:**
- [x] `python app.py`; click 5 spots across the sky with each agent (tested by hand)

**Dependencies:** 9
**Files:** `app.py` (background rendered from the env at startup, no assets file)
**Scope:** S

## Task 11: Deploy + README
**Description:** Push to a Hugging Face Space (Gradio SDK, pinned Python with working box2d; model zips under ~50 MB total). README: reward shaping explained, conditions, Φ used, curves + heatmaps + GIFs, link to source `lunar_lander.py`, Space link, reproduce commands.

**Decided 2026-10-20:** README written (front matter = Space config). Owner pushes the Space with `hf upload` (app.py, env.py, requirements.txt, README.md, models/, results/). Owner merges `front-end` → `main` via PR. `runs/*/eval.csv` committed so `plot.py` works from a clone; `plot.py` now refuses to run without them instead of blanking results/.

**Acceptance criteria:**
- [ ] Space builds and works from a logged-out browser
- [ ] README reproduces results from a fresh clone

**Verification:**
- [ ] Open Space in private window, click 3 spots
- [ ] Fresh venv: `pip install -r requirements.txt && python test_shaping.py && python plot.py`

**Dependencies:** 10
**Files:** `README.md`, Space config (README front-matter), `requirements.txt`
**Scope:** S

### Checkpoint D
- [ ] All acceptance criteria met
- [ ] Public link + README ready to share
