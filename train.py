import argparse
import csv
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from shaping import PotentialShaping, SparseReward

# RL-Zoo PPO hyperparams for LunarLander
HPARAMS = dict(n_steps=1024, batch_size=64, gae_lambda=0.98, gamma=0.999, n_epochs=4, ent_coef=0.01)
WRAPPERS = {
    "default": None,
    "sparse": SparseReward,
    "pbrs": lambda env: PotentialShaping(SparseReward(env), HPARAMS["gamma"]),  # shaping γ must equal PPO's γ
}
N_ENVS = 16
EVAL_EPISODES = 10


def evaluate(model, episodes=EVAL_EPISODES):
    """Deterministic episodes on the plain env: its reward is the true score. Fixed seeds so every run sees the same starts."""
    env = gym.make("LunarLander-v3")
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
    def __init__(self, out):
        super().__init__()
        self.out, self.episodes = out, 0

    def _on_step(self):
        self.episodes += sum("episode" in i for i in self.locals["infos"])  # Monitor adds "episode" at episode end
        return True

    # Once per policy update: rollout start = policy after the previous update (first = untrained), plus the final one.
    def _on_rollout_start(self):
        self._eval()

    def _on_training_end(self):
        self._eval()

    def _eval(self):
        row = [self.num_timesteps, self.episodes, *evaluate(self.model)]
        with open(f"{self.out}/eval.csv", "a", newline="") as f:
            csv.writer(f).writerow(row)
        self.model.save(f"{self.out}/ckpt_{self.episodes}.zip")
        print(row, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward", choices=WRAPPERS, default="default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    a = p.parse_args()

    out = f"runs/{a.reward}_s{a.seed}"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/eval.csv", "w", newline="") as f:
        csv.writer(f).writerow(["step", "episodes_so_far", "mean_true_score", "success_rate"])

    # make_vec_env puts Monitor directly on the base env, under wrapper_class -> Monitor logs the true score
    env = make_vec_env("LunarLander-v3", n_envs=N_ENVS, seed=a.seed, wrapper_class=WRAPPERS[a.reward])
    model = PPO("MlpPolicy", env, seed=a.seed, device="cpu", **HPARAMS)
    model.learn(a.steps, callback=EvalCallback(out))
    model.save(f"{out}/model.zip")


if __name__ == "__main__":
    main()
