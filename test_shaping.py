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


if __name__ == "__main__":
    test_sparse_decomposition()
    print("ok")
