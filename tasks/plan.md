# Implementation Plan: Lunar Lander with Reward Shaping + Click-to-Land Web App

## Overview
Train RL agents to land Gymnasium's `LunarLander-v3` (source: `gymnasium/envs/box2d/lunar_lander.py`, Farama-Foundation/Gymnasium on GitHub) and show how reward shaping changes learning. Then ship a public web app: a visitor clicks anywhere in the sky, the rocket spawns there, and a trained agent lands it — with a dropdown to compare agents trained under different rewards.

Reward conditions (all scored on the same true score):

| Condition | Training reward |
|---|---|
| `default` | Env as shipped (it already contains potential-style shaping) |
| `sparse` | Built-in shaping stripped out: only ±100 terminal + fuel cost |
| `pbrs` | `sparse` + my own potential-based shaping `F = γΦ(s') − Φ(s)` |
| `hack` | `sparse` + a naive non-potential bonus (expected to cause hovering / reward hacking) |

Deliverables: learning curves, a "training progression" strip (rollouts at episode ~1 / ~500 / ~3000 next to the score curve with markers), landing-success heatmap over start positions, GIFs with a live score readout, README, and a live Hugging Face Space.

Inspiration: an Instagram reel explaining exactly this setup (+closer, +slowing down, +upright, −fuel, −time, ±100 land/crash). Its 3D rocket render is a stylized video, not the sim — out of scope; the sim and app stay 2D.

## Architecture Decisions
- **No fork of `lunar_lander.py`.** Built-in shaping is a pure function of the observation: `Φ_env(s) = −100·‖(x,y)‖ − 100·‖(vx,vy)‖ − 100·|θ| + 10·legL + 10·legR`, step reward = `Φ_env(s') − Φ_env(s)` − fuel, overridden to ±100 on termination. A wrapper subtracts it to get the sparse reward.
- **Custom start = reset option.** `env.reset(options={"start": (x, y)})` translates the lander body and both legs together after the normal reset, zeroes velocity, and recomputes the observation. No start given → random start in the allowed sky region (used for training).
- **Click = start position; landing pad stays fixed** at the centre. Moving the pad needs terrain changes + goal-conditioned obs → stretch goal, not planned.
- **Train on random starts.** A fixed-start agent won't generalize to clicked positions. Phase 2 compares rewards quickly on the fixed start; the final sweep trains on random starts, which is also where shaping matters most (far starts are hard with sparse reward).
- **Evaluation always uses the env's original reward.** `Monitor` wraps the base env (inside shaping wrappers). "Solved" = mean ≥ 200; success = episode ends at rest (+100).
- **PBRS:** shaping γ equals PPO's γ; `Φ(terminal) = 0` (Ng et al., 1999 policy invariance).
- **Algorithm:** Stable-Baselines3 PPO, discrete actions, RL-Zoo hyperparameters as start.
- **Web app: Gradio on Hugging Face Spaces, policy runs server-side in Python.** Click on a background image → `(x, y)` → run one episode (≤ 1000 steps, ~1 s on CPU) → return GIF + outcome. In-browser inference rejected: would require re-implementing Box2D physics in JS, and sim mismatch breaks the policy.

## Dependency Graph
```
env installs (box2d)
   └── reward wrappers (sparse → pbrs, hack)
          └── train.py (train + eval on true score)
                 ├── fixed-start comparisons (Phase 2)
                 └── custom/random start option
                        └── random-start sweep + heatmap
                               └── land(x, y, agent) → frames, outcome
                                      └── Gradio app → HF Space + README
```

## Task List (details in `tasks/todo.md`)

### Phase 1: Foundation
- [x] Task 1: Env runs; built-in heuristic lander scores ≥ 200
- [x] Task 2: Sparse-reward wrapper, verified by reward decomposition check
- [x] Task 3: End-to-end slice: train PPO on `default`, eval true score, save model + CSV

### Checkpoint A: pipeline works end to end

### Phase 2: Shaping experiments (fixed start, seed 0)
- [x] Task 4: `sparse` run
- [ ] Task 5: Potential-based shaping `pbrs` + telescoping check
- [ ] Task 6: Non-potential `hack` shaping; observe hovering

### Checkpoint B: each condition shows a distinct, explainable behavior

### Phase 3: Land from anywhere
- [ ] Task 7: Custom start position via `reset(options={"start": ...})`
- [ ] Task 8: Random-start sweep (conditions × seeds) + learning curves + success heatmap
- [ ] Task 9: `land(x, y, agent)` inference function

### Checkpoint C: chosen agents land from most of the sky

### Phase 4: Web app
- [ ] Task 10: Gradio click-to-land app (local)
- [ ] Task 11: Deploy to Hugging Face Spaces + README

### Checkpoint D: public link works for a stranger

## Risks and Mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| `box2d` won't build on Python 3.13 | High | Task 1 first; fall back to a 3.11/3.12 venv (HF Spaces lets you pin Python) |
| Teleporting lander breaks joints/physics | Med | Move body + legs by the same offset, zero velocities; Task 7 checks a resting drop from the click point |
| Random starts much harder to learn → agents crash from corners | Med | Heatmap shows where it fails; restrict clickable region to where success ≥ 80%, or train longer |
| Compute (4 conditions × 3 seeds × ~1–2M steps on CPU) | Med | Phase 2 at 1M (300k too short); sweep in background; drop `default`/seeds before dropping `pbrs` vs `sparse` |
| Public app abuse / slow free CPU | Low | Validate + clamp coords, cap episode at 1000 steps, Gradio queue (one run at a time) |
| `hack` doesn't visibly hack | Low | Per-step proximity bonus so hovering is clearly optimal |

## Open Questions
- Click sets the **start position** (planned). Also want the click to move the **landing pad**? (Stretch: needs terrain + goal-conditioned training.)
- Compute budget OK for ~1–2M steps × 12 runs on random starts, or cut to 2 seeds?
- Public HF Space under your account — fine?
