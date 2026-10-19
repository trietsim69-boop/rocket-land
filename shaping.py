import gymnasium as gym


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
