import gymnasium as gym

from shaping import SparseReward

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


if __name__ == "__main__":
    test_sparse_decomposition()
    test_monitor_sees_true_score()
    print("ok")
