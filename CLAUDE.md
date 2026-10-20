# Rocket Land — Lunar Lander reward shaping + click-to-land web app

RL agents land Gymnasium's `LunarLander-v3`. The project compares reward designs
(`default`, `sparse`, `pbrs`, `hack`) on the same true score, trains agents to land
from any start point, and ships a Gradio app on Hugging Face Spaces where a visitor
clicks a spot in the sky and a chosen agent lands from there.

Plan: `tasks/plan.md` (decisions, risks). Task list + status: `tasks/todo.md`.
Read both before starting work; tick boxes in `todo.md` when a task's verification passes.

## Stack
- Python 3.11 or 3.12 (box2d wheels are unreliable on 3.13), venv in `.venv/`
- `gymnasium[box2d]`, `stable-baselines3` (PPO, discrete actions), `matplotlib`, `imageio`, `gradio`
- Windows dev machine: if `box2d` fails to build, `pip install swig` first, then reinstall `gymnasium[box2d]`
- Edit text files with the Edit tool or Python (`encoding="utf-8"`). PowerShell 5.1 `Get-Content`/`Set-Content` re-encodes UTF-8 as cp1252 and mangles the ≥ − Φ γ in `tasks/*.md`.

## Commands
- Setup: `python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt`
- Checks: `python test_shaping.py`
- Train: `python train.py --reward {default,sparse,pbrs,hack} --seed N --steps N [--random-start]`
- Plots/results: `python plot.py`
- App: `python app.py`

## Files
- `shaping.py`: reward wrappers (`SparseReward`, `PotentialShaping`, `NaiveBonus`)
- `env.py`: custom/random start position via `reset(options={"start": (x, y)})`
- `train.py`: train + periodic eval → `runs/{reward}_s{seed}/eval.csv`, checkpoints, `model.zip`
- `plot.py`: learning curves, success heatmaps, progression strip → `results/`
- `app.py`: `trajectory()` server function + one `gr.HTML` page (SVG scene + JS replay, design from `UI/designs/`)
- `test_shaping.py`: the single assert-based check file
- `models/`: the few final models the app ships (committed); `runs/` is scratch (gitignored)

## Rules that keep results correct
- **Never fork or edit `lunar_lander.py`.** Use the installed Gymnasium env + wrappers. The sparse reward comes from subtracting the env's own shaping `Φ_env(s) = −100‖(x,y)‖ − 100‖(vx,vy)‖ − 100|θ| + 10·legL + 10·legR`, not from a copied file.
- **Evaluate on the true score.** `Monitor` wraps the base env *inside* any shaping wrapper. Compare conditions only on that score and on success rate (episode ends at rest, +100).
- **PBRS:** shaping γ must equal PPO's γ. `Φ(terminal) = 0`.
- Any change to a reward wrapper keeps `test_shaping.py` passing (decomposition + telescoping asserts).
- Plots and tables are generated from saved CSVs only. No hand-typed numbers in the README.

## Conventions
- Few files, plain functions, no config framework: CLI flags with argparse.
- Seed everything (`--seed`), so any run can be repeated.
- App input is untrusted: clamp click coords into the allowed sky region, cap episodes at 1000 steps.

## Boundaries
- Ask before starting any training run longer than ~30 min, or a multi-run sweep. Measured on this machine: 1M steps ≈ 13 min solo; 2M ≈ 16–19 min each with 4 runs in parallel (`OMP_NUM_THREADS=2`).
- Don't commit `runs/` (except `runs/*/eval.csv`, the ~75 KB of inputs `plot.py` needs on a fresh clone), `.venv/`, or checkpoint dumps; only the app's models in `models/` (keep under ~50 MB total).
- Commits use the repo owner's own git identity (Triet Le). No Claude author or co-author lines.
- Don't add dependencies beyond the stack above without asking.
