import argparse
import csv
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from env import StartPosition
from shaping import NaiveBonus, PotentialShaping, SparseReward

# RL-Zoo PPO hyperparams for LunarLander, except gae_lambda 0.98 -> 0.9: at 0.98 pbrs washes out to sparse (runs/*_lam0.98)
HPARAMS = dict(n_steps=1024, batch_size=64, gae_lambda=0.9, gamma=0.999, n_epochs=4, ent_coef=0.01)
WRAPPERS = {
    "default": None,
    "sparse": SparseReward,
    "pbrs": lambda env: PotentialShaping(SparseReward(env), HPARAMS["gamma"]),  # shaping γ must equal PPO's γ
    "hack": lambda env: NaiveBonus(SparseReward(env)),
}
N_ENVS = 16
EVAL_EPISODES = 10


def evaluate(model, random_start=False, episodes=EVAL_EPISODES):
    """Deterministic episodes on the plain env: its reward is the true score. Fixed seeds so every run sees the same starts."""
    env = StartPosition(gym.make("LunarLander-v3"), random=random_start)
    scores, wins = [], 0
    for i in range(episodes):
        obs, _ = env.reset(seed=10_000 + i)
        total, done = 0.0, False
        while not done:
            obs, r, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
            total += r
            done = term or trunc
        scores.append(total)
        wins += term and r == 100  # +100 only when the lander comes to rest
    return float(np.mean(scores)), wins / episodes


class EvalCallback(BaseCallback):
    def __init__(self, out, random_start):
        super().__init__()
        self.out, self.random_start, self.episodes = out, random_start, 0

    def _on_step(self):
        self.episodes += sum("episode" in i for i in self.locals["infos"])  # Monitor adds "episode" at episode end
        return True

    # Once per policy update: rollout start = policy after the previous update (first = untrained), plus the final one.
    def _on_rollout_start(self):
        self._eval()

    def _on_training_end(self):
        self._eval()

    def _eval(self):
        row = [self.num_timesteps, self.episodes, *evaluate(self.model, self.random_start)]
        with open(f"{self.out}/eval.csv", "a", newline="") as f:
            csv.writer(f).writerow(row)
        self.model.save(f"{self.out}/ckpt_{self.episodes}.zip")
        print(row, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward", choices=WRAPPERS, default="default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--gae-lambda", type=float, default=HPARAMS["gae_lambda"])
    p.add_argument("--random-start", action="store_true", help="spawn anywhere in the allowed sky region (env.py)")
    a = p.parse_args()

    out = f"runs/{a.reward}_s{a.seed}" + (f"_lam{a.gae_lambda}" if a.gae_lambda != HPARAMS["gae_lambda"] else "") + ("_rand" * a.random_start)
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/eval.csv", "w", newline="") as f:
        csv.writer(f).writerow(["step", "episodes_so_far", "mean_true_score", "success_rate"])

    # make_vec_env puts Monitor directly on the base env, under wrapper_class -> Monitor logs the true score
    make = lambda: StartPosition(gym.make("LunarLander-v3"), random=a.random_start)
    env = make_vec_env(make, n_envs=N_ENVS, seed=a.seed, wrapper_class=WRAPPERS[a.reward])
    model = PPO("MlpPolicy", env, seed=a.seed, device="cpu", **{**HPARAMS, "gae_lambda": a.gae_lambda})
    model.learn(a.steps, callback=EvalCallback(out, a.random_start))
    model.save(f"{out}/model.zip")


if __name__ == "__main__":
    main()
