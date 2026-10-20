import gymnasium as gym

import numpy as np

from env import X_RANGE, Y_RANGE, StartPosition
from shaping import NaiveBonus, PotentialShaping, SparseReward, potential

FUEL = {0: 0.0, 1: 0.03, 2: 0.3, 3: 0.03}  # discrete actions fire engines at full power


def test_sparse_decomposition(steps=5000):
    env = SparseReward(gym.make("LunarLander-v3"))
    env.reset(seed=0)
    env.action_space.seed(0)
    terminals = 0
    for _ in range(steps):
        a = int(env.action_space.sample())
        _, r, term, trunc, _ = env.step(a)
        if term:
            assert r in (-100, 100), r
            terminals += 1
        else:
            assert abs(r + FUEL[a]) < 1e-4, (a, r)
        if term or trunc:
            env.reset()
    assert terminals > 10  # random policy crashes often, so terminals were actually exercised


def test_monitor_sees_true_score():
    from stable_baselines3.common.monitor import Monitor

    wrapped, plain = SparseReward(Monitor(gym.make("LunarLander-v3"))), gym.make("LunarLander-v3")
    wrapped.reset(seed=1)
    plain.reset(seed=1)
    plain.action_space.seed(1)
    total, done = 0.0, False
    while not done:
        a = int(plain.action_space.sample())
        wrapped.step(a)
        _, r, term, trunc, _ = plain.step(a)
        total += r
        done = term or trunc
    assert abs(wrapped.env.get_episode_rewards()[0] - total) < 1e-6


def test_pbrs_telescopes(gamma=0.999):
    """Σ γ^t F_t = γ^T Φ(s_T) − Φ(s_0), with Φ(s_T) = 0 on termination. F = pbrs reward − sparse reward."""
    sparse = SparseReward(gym.make("LunarLander-v3"))
    pbrs = PotentialShaping(SparseReward(gym.make("LunarLander-v3")), gamma)
    obs0, _ = sparse.reset(seed=2)
    pbrs.reset(seed=2)
    sparse.action_space.seed(2)
    disc, t, done = 0.0, 0, False
    while not done:
        a = int(sparse.action_space.sample())
        _, r_s, term, trunc, _ = sparse.step(a)
        _, r_p, *_ = pbrs.step(a)
        disc += gamma**t * (r_p - r_s)
        t += 1
        done = term or trunc
    assert term  # covers Φ(terminal) = 0
    assert abs(disc - (gamma**t * 0.0 - potential(obs0))) < 1e-6, (disc, -potential(obs0))


def test_hack_bonus_only_near_pad(steps=3000):
    sparse = SparseReward(gym.make("LunarLander-v3"))
    hack = NaiveBonus(SparseReward(gym.make("LunarLander-v3")))
    sparse.reset(seed=3)
    hack.reset(seed=3)
    sparse.action_space.seed(3)
    near = 0
    for _ in range(steps):
        a = int(sparse.action_space.sample())
        obs, r_s, term, trunc, _ = sparse.step(a)
        _, r_h, *_ = hack.step(a)
        inside = np.hypot(obs[0], obs[1]) < hack.radius
        assert abs(r_h - r_s - 0.5 * inside) < 1e-9, (r_h, r_s, inside)
        near += inside
        if term or trunc:
            sparse.reset()
            hack.reset()
    assert near > 0  # bonus branch actually exercised


def test_start_position():
    env = SparseReward(StartPosition(gym.make("LunarLander-v3")))
    u = env.unwrapped
    rng = np.random.default_rng(4)
    for i in range(10):
        start = (rng.uniform(*X_RANGE), rng.uniform(*Y_RANGE))
        obs, _ = env.reset(seed=i, options={"start": start})
        assert np.abs(obs[:2] - start).max() < 1e-3, (obs[:2], start)
        assert np.abs(obs[2:4]).max() < 0.05  # at rest apart from one frame of gravity (vy ≈ −0.026; default spawn ≈ −0.5)
        # No-op drop: legs stay attached all the way down, and SparseReward's Φ was refreshed by the teleport
        done, first = False, True
        while not done:
            _, r, term, trunc, _ = env.step(0)
            if first and not term:
                assert abs(r) < 1e-4, r  # stale prev_shaping would show up here as a big shaping diff
            first = False
            if not (u.game_over or any(leg.ground_contact for leg in u.legs)):  # crash impact stretches joints in the stock env too
                for leg in u.legs:
                    a, b = leg.joint.anchorA, leg.joint.anchorB
                    assert np.hypot(a.x - b.x, a.y - b.y) < 1e-3, "leg detached mid-air"
            done = term or trunc
    for bad in [(5, -3), (-1e9, 1e9), (float("nan"), float("inf"))]:  # clamps, never raises
        obs, _ = env.reset(seed=0, options={"start": bad})
        assert X_RANGE[0] - 1e-3 <= obs[0] <= X_RANGE[1] + 1e-3 and Y_RANGE[0] - 1e-3 <= obs[1] <= Y_RANGE[1] + 1e-3
    rand = StartPosition(gym.make("LunarLander-v3"), random=True)
    a, b = rand.reset(seed=7)[0][:2], rand.reset(seed=7)[0][:2]
    assert np.allclose(a, b) and X_RANGE[0] <= a[0] <= X_RANGE[1]  # random start is seeded + in region


def test_click_mapping():
    from gymnasium.envs.box2d.lunar_lander import SCALE, VIEWPORT_H

    from app import CLICK_X, clamp_click, pixel_to_obs

    env = StartPosition(gym.make("LunarLander-v3"))
    for start in [(0.0, 1.0), (-0.6, 0.6), (0.55, 1.4)]:
        obs, _ = env.reset(seed=0, options={"start": start})
        pos = env.unwrapped.lander.position
        assert np.allclose(pixel_to_obs(pos.x * SCALE, VIEWPORT_H - pos.y * SCALE), obs[:2], atol=1e-3)
    x, y, clamped = clamp_click(5, -3)
    assert clamped and CLICK_X[0] <= x <= CLICK_X[1] and Y_RANGE[0] <= y <= Y_RANGE[1]
    assert not clamp_click(0.2, 1.0)[2]


def test_trajectory():
    """Replaying the trajectory's actions on a fresh env ends on the same step with the same true score."""
    from app import APP_SEED, obs_to_pixel, trajectory

    px, py = obs_to_pixel(-0.4, 1.2)
    tr = trajectory({"px": px, "py": py, "agent": "default"})
    assert np.allclose(tr["start"], (-0.4, 1.2), atol=0.01) and not tr["clamped"]
    env = StartPosition(gym.make("LunarLander-v3"))
    env.reset(seed=APP_SEED, options={"start": tuple(tr["start"])})
    total, done, steps = 0.0, False, 0
    for state in tr["states"][1:]:
        assert not done
        _, r, term, trunc, _ = env.step(state[5])
        total, done, steps = total + r, term or trunc, steps + 1
    assert done and steps == len(tr["states"]) - 1
    assert abs(total - tr["score"]) < 0.06 and abs(tr["states"][-1][6] - tr["score"]) < 0.06


if __name__ == "__main__":
    test_sparse_decomposition()
    test_monitor_sees_true_score()
    test_pbrs_telescopes()
    test_hack_bonus_only_near_pad()
    test_start_position()
    test_click_mapping()
    test_trajectory()
    print("ok")
