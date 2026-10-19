import gymnasium as gym

import numpy as np

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
        inside = np.hypot(obs[0], obs[1]) < 0.2
        assert abs(r_h - r_s - 0.5 * inside) < 1e-9, (r_h, r_s, inside)
        near += inside
        if term or trunc:
            sparse.reset()
            hack.reset()
    assert near > 0  # bonus branch actually exercised


if __name__ == "__main__":
    test_sparse_decomposition()
    test_monitor_sees_true_score()
    test_pbrs_telescopes()
    test_hack_bonus_only_near_pad()
    print("ok")
