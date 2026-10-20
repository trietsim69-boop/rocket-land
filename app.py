"""Click a spot in the sky, pick an agent, watch it land from there. `python app.py` -> http://localhost:7860"""
import tempfile

import gradio as gr
import gymnasium as gym
import imageio
import numpy as np
from gymnasium.envs.box2d.lunar_lander import LEG_DOWN, SCALE, VIEWPORT_H, VIEWPORT_W
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from env import Y_RANGE, StartPosition, clamp_start

REWARDS = {
    "pbrs": "potential-based shaping (sparse + γΦ(s') − Φ(s))",
    "default": "the env's built-in reward",
    "sparse": "only ±100 at the end + fuel cost",
    "hack": "sparse + a naive 'stay near the pad' bonus",
}
MODELS = {r: PPO.load(f"models/{r}.zip", device="cpu") for r in REWARDS}  # best seed per reward, see tasks/todo.md
CLICK_X = (-0.6, 0.6)  # Checkpoint C: narrower than training (±0.8), where the shipped agents land reliably
APP_SEED = 0  # fixed terrain, so every episode matches the background image
FONT = ImageFont.load_default(size=22)
W, H = VIEWPORT_W / SCALE, VIEWPORT_H / SCALE
PAD_Y = H / 4 + LEG_DOWN / SCALE  # world y of obs y = 0 (helipad_y + leg offset, as in the env's obs)


def pixel_to_obs(px, py):
    """Render pixel (origin top-left, 600×400) -> observation (x, y), pad at (0, 0)."""
    return px / (VIEWPORT_W / 2) - 1, ((VIEWPORT_H - py) / SCALE - PAD_Y) / (H / 2)


def obs_to_pixel(x, y):
    return (x + 1) * VIEWPORT_W / 2, VIEWPORT_H - (y * H / 2 + PAD_Y) * SCALE


def clamp_click(x, y):
    """Untrusted click -> (x, y) inside the click region, plus whether it had to move."""
    cx, cy = clamp_start(x, y)
    cx = float(np.clip(cx, *CLICK_X))
    return cx, cy, not (np.isclose(cx, x) and np.isclose(cy, y))


def make_env():
    return StartPosition(gym.make("LunarLander-v3", render_mode="rgb_array"))  # TimeLimit caps episodes at 1000 steps


def background():
    env = make_env()
    env.reset(seed=APP_SEED)
    u = env.unwrapped
    for body in [u.lander, *u.legs]:  # empty sky: park the lander off-screen for this one render
        body.position = (-100, -100)
    img = Image.fromarray(env.render())
    (x0, y0), (x1, y1) = obs_to_pixel(CLICK_X[0], Y_RANGE[1]), obs_to_pixel(CLICK_X[1], Y_RANGE[0])
    ImageDraw.Draw(img).rectangle((x0, y0, x1, y1), outline=(90, 90, 90), width=2)
    return img


def land(x, y, agent):
    """One deterministic episode from (x, y) -> (gif_path, outcome, true_score, (x, y) actually used, clamped)."""
    x, y, clamped = clamp_click(x, y)
    env, model = make_env(), MODELS[agent]
    obs, _ = env.reset(seed=APP_SEED, options={"start": (x, y)})
    frames, score, done = [], 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(int(model.predict(obs, deterministic=True)[0]))
        score += r  # plain env reward = the true score
        done = term or trunc
        frames.append((env.render(), score))
    outcome = "landed" if term and r == 100 else "crashed" if term else "timeout"

    gif = []
    for k, (frame, s) in enumerate(frames):
        if k % 2 and k != len(frames) - 1:  # every 2nd frame at 40 ms ≈ real time (50 FPS sim), always keep the last
            continue
        img = Image.fromarray(frame)
        ImageDraw.Draw(img).text((12, 10), f"SCORE {s:+.0f}", font=FONT, fill=(60, 200, 90) if s >= 0 else (230, 70, 70))
        gif.append(np.asarray(img))
    gif += [gif[-1]] * 25  # hold the final frame ~1 s before looping
    path = tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name
    imageio.mimsave(path, gif, duration=40, loop=0)
    return path, outcome, score, (x, y), clamped


def run(agent, click):
    if agent not in MODELS:  # dropdown values can be forged through the API
        raise gr.Error(f"unknown agent {agent!r}")
    path, outcome, score, (x, y), clamped = land(*click, agent)
    text = f"**{agent}**: {outcome}, true score **{score:+.0f}**, from ({x:.2f}, {y:.2f})"
    if clamped:
        text += "  \n_Your click was outside the sky region (grey box), so it was moved to the nearest point inside._"
    return path, text


def on_click(agent, evt: gr.SelectData):
    click = pixel_to_obs(*evt.index[:2])
    return *run(agent, click), click


def on_agent_change(agent, click):
    return run(agent, click) if click else (gr.skip(), gr.skip())


with gr.Blocks(title="Rocket Land") as demo:
    gr.Markdown("# Rocket Land\nClick anywhere in the grey box to drop the lander there. The agent tries to land on the pad between the flags.")
    agent = gr.Dropdown([(f"{r}: {d}", r) for r, d in REWARDS.items()], value="pbrs", label="Agent (training reward)")
    click = gr.State(None)
    with gr.Row():
        sky = gr.Image(background(), interactive=False, show_label=False, buttons=[])
        replay = gr.Image(show_label=False, buttons=[])
    result = gr.Markdown()
    sky.select(on_click, [agent], [replay, result, click])
    agent.change(on_agent_change, [agent, click], [replay, result])

demo.queue(default_concurrency_limit=1)  # one episode at a time on a small CPU box

if __name__ == "__main__":
    demo.launch()
