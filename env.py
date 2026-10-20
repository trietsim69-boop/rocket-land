import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import LEG_DOWN, SCALE, VIEWPORT_H, VIEWPORT_W

# Allowed start region in observation coords (pad at (0, 0)). x: |x| >= 1 ends the episode, keep a margin.
# y: terrain peaks at ~0.41, the default spawn (top of the viewport) is ~1.41.
X_RANGE, Y_RANGE = (-0.8, 0.8), (0.6, 1.4)


def clamp_start(x, y):
    """Untrusted (x, y) -> a point inside the allowed region. NaN -> centre of the range, ±inf -> the edge."""
    return tuple(
        float(np.clip(np.nan_to_num(v, nan=sum(r) / 2), *r)) for v, r in ((x, X_RANGE), (y, Y_RANGE))
    )


class StartPosition(gym.Wrapper):
    """`reset(options={"start": (x, y)})` spawns the lander at rest at (x, y), clamped into the allowed region.
    No start given: uniform random start if `random`, else the env's normal spawn."""

    def __init__(self, env, random=False):
        super().__init__(env)
        self.random = random

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        u = self.env.unwrapped
        start = (options or {}).get("start")
        if start is None and self.random:
            start = (u.np_random.uniform(*X_RANGE), u.np_random.uniform(*Y_RANGE))
        if start is None:
            return obs, info

        x, y = clamp_start(*start)
        W, H = VIEWPORT_W / SCALE, VIEWPORT_H / SCALE
        dx = x * W / 2 + W / 2 - u.lander.position.x
        dy = y * H / 2 + u.helipad_y + LEG_DOWN / SCALE - u.lander.position.y
        for body in [u.lander, *u.legs]:  # same offset for all three keeps the leg joints intact
            body.position = (body.position.x + dx, body.position.y + dy)
            body.linearVelocity = (0, 0)
            body.angularVelocity = 0
        # Same trick reset() itself ends with: one no-op step recomputes obs AND prev_shaping (SparseReward reads it).
        # Bypasses TimeLimit/Monitor, so it doesn't count as an episode step. Gravity moves y by ~6e-4 in that step.
        return u.step(0)[0], info
