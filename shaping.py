import gymnasium as gym
import numpy as np


class SparseReward(gym.Wrapper):
    """Strips LunarLander's built-in shaping: non-terminal reward = -fuel only, terminal ±100 untouched.

    Reads the env's own Φ_env(s) (`prev_shaping`, float64, set on every step incl. the one inside reset)
    instead of recomputing it from the float32 obs. Wrap a custom-start env only if it refreshes
    `prev_shaping` after teleporting, else the first step's subtraction is off.
    """

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.phi = self.env.unwrapped.prev_shaping
        return out

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        phi = self.env.unwrapped.prev_shaping
        if not terminated:
            r -= phi - self.phi
        self.phi = phi
        return obs, r, terminated, truncated, info


def potential(obs):
    """My Φ: closer to pad, slower, upright, legs down. Same terms as the env's, own weights (half the tilt weight)."""
    x, y, vx, vy, theta, _, leg_l, leg_r = obs
    return float(-100 * np.hypot(x, y) - 100 * np.hypot(vx, vy) - 50 * abs(theta) + 10 * (leg_l + leg_r))


class PotentialShaping(gym.Wrapper):
    """Adds F = γΦ(s') − Φ(s) (Ng et al. 1999). Φ(terminal) = 0; truncation isn't terminal, so it keeps Φ(s')."""

    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.phi = potential(obs)
        return obs, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        phi = 0.0 if terminated else potential(obs)
        r += self.gamma * phi - self.phi
        self.phi = phi
        return obs, r, terminated, truncated, info
