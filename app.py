"""Click a spot in the sky, pick an agent, watch it land from there. `python app.py` -> http://localhost:7860
The server runs the real episode and returns a trajectory; the page (gr.HTML, design from UI/designs/) draws it as SVG."""
import csv
import random
import threading

import gradio as gr
import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import LEG_DOWN, SCALE, VIEWPORT_H, VIEWPORT_W
from stable_baselines3 import PPO

from env import Y_RANGE, StartPosition, clamp_start

REWARDS = {
    "pbrs": "Potential-based shaping: sparse + γΦ(s′) − Φ(s)",
    "default": "The env’s built-in reward",
    "sparse": "Only ±100 at the end, plus fuel cost",
    "hack": "Sparse + a naive “stay near the pad” bonus",
}
MODELS = {r: PPO.load(f"models/{r}.zip", device="cpu") for r in REWARDS}
# Run each models/{r}.zip was copied from (best seed in the click region, Checkpoint C in tasks/todo.md)
REWARD_RUNS = {"default": "runs/default_s1_rand", "sparse": "runs/sparse_s2_rand", "pbrs": "runs/pbrs_s2_rand", "hack": "runs/hack_s1_rand"}
CLICK_X = (-0.6, 0.6)  # Checkpoint C: narrower than training (±0.8), where the shipped agents land reliably
APP_SEED = 0  # fixed terrain: every episode uses the terrain drawn on the page
W, H = VIEWPORT_W / SCALE, VIEWPORT_H / SCALE
PAD_Y = H / 4 + LEG_DOWN / SCALE  # world y of obs y = 0 (helipad_y + leg offset, as in the env's obs)
FUEL = {0: 0.0, 1: 0.03, 2: 0.3, 3: 0.03}  # env's fuel cost per discrete action
# ponytail: one episode at a time for the whole app (server functions skip Gradio's queue); per-session if it ever queues up
LOCK = threading.Lock()


def pixel_to_obs(px, py):
    """Render pixel (origin top-left, 600×400) -> observation (x, y), pad at (0, 0)."""
    return px / (VIEWPORT_W / 2) - 1, ((VIEWPORT_H - py) / SCALE - PAD_Y) / (H / 2)


def obs_to_pixel(x, y):
    return (x + 1) * VIEWPORT_W / 2, VIEWPORT_H - (y * H / 2 + PAD_Y) * SCALE


def to_pixel(wx, wy):
    """World (Box2D) -> render pixel, origin top-left."""
    return round(float(wx) * SCALE, 1), round(VIEWPORT_H - float(wy) * SCALE, 1)


def clamp_click(x, y):
    """Untrusted click -> (x, y) inside the click region, plus whether it had to move."""
    cx, cy = clamp_start(x, y)
    cx = float(np.clip(cx, *CLICK_X))
    return cx, cy, not (np.isclose(cx, x) and np.isclose(cy, y))


def make_env():
    return StartPosition(gym.make("LunarLander-v3"))  # TimeLimit caps episodes at 1000 steps


def terrain():
    """Ground polyline + pad in render pixels for APP_SEED (the terrain of every episode)."""
    env = make_env()
    env.reset(seed=APP_SEED)
    u = env.unwrapped
    ground = [to_pixel(*p[0]) for p in u.sky_polys] + [to_pixel(*u.sky_polys[-1][1])]
    (x1, y), (x2, _) = to_pixel(u.helipad_x1, u.helipad_y), to_pixel(u.helipad_x2, u.helipad_y)
    return {"ground": ground, "pad": {"x1": x1, "x2": x2, "y": y}}


def zone_success(reward):
    """Landing rate of the shipped model inside the click region, from plot.py's saved heatmap CSV."""
    with open(f"results/heatmap_{reward}.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["run"] == REWARD_RUNS[reward] and abs(float(r["x"])) <= CLICK_X[1]]
    return sum(int(r["landed"]) for r in rows) / sum(int(r["episodes"]) for r in rows)


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")  # clamp_start turns NaN into the centre of the region


def trajectory(req):
    """Server function for the page JS: {px, py, agent} (click in scene pixels) -> one episode as per-step states.
    State = [px, py, angle, legL, legR, action, score, alt, speed, fuel]; states[0] is the start, then one per step."""
    agent = req.get("agent") if isinstance(req, dict) else None
    if agent not in MODELS:  # untrusted: anything can be POSTed to a server function
        raise gr.Error(f"unknown agent {agent!r}")
    x, y, clamped = clamp_click(*pixel_to_obs(_num(req.get("px")), _num(req.get("py"))))
    with LOCK:
        env, model = make_env(), MODELS[agent]
        obs, _ = env.reset(seed=APP_SEED, options={"start": (x, y)})
        u, a, r, term, score, fuel, done = env.unwrapped, 0, 0.0, False, 0.0, 0.0, False
        states = []
        while True:
            states.append([*to_pixel(*u.lander.position), round(u.lander.angle, 3), int(obs[6]), int(obs[7]), a,
                           round(score, 1), round(float(obs[1]), 2), round(float(np.hypot(obs[2], obs[3])), 2), round(fuel, 1)])
            if done:
                break
            a = int(model.predict(obs, deterministic=True)[0])
            obs, r, term, trunc, _ = env.step(a)
            score += r  # plain env reward = the true score
            fuel += FUEL[a]
            done = term or trunc
    outcome = "landed" if term and r == 100 else "crashed" if term else "timeout"
    return {"states": states, "outcome": outcome, "score": round(score, 1), "start": [round(x, 2), round(y, 2)], "clamped": bool(clamped)}


# ---------- page (markup copied from UI/designs/*.html) ----------

ROCKET = """
<g id="flame"><path d="M-8,8 Q0,44 8,8 Z" fill="#FF8A3D" opacity="0.9"/><path d="M-5,8 Q0,30 5,8 Z" fill="#FFD27A"/><path d="M-2.5,8 Q0,18 2.5,8 Z" fill="#FFF6DA"/></g>
<g id="puff-r"><path d="M17,-12 Q34,-9 17,-5 Z" fill="#9FE3FF" opacity="0.9"/><path d="M17,-10.5 Q26,-8.5 17,-6.5 Z" fill="#E6F8FF"/></g>
<g id="puff-l"><path d="M-17,-12 Q-34,-9 -17,-5 Z" fill="#9FE3FF" opacity="0.9"/><path d="M-17,-10.5 Q-26,-8.5 -17,-6.5 Z" fill="#E6F8FF"/></g>
<line x1="-11" y1="5" x2="-26" y2="15" stroke="#9AA6BD" stroke-width="3" stroke-linecap="round"/>
<line x1="11" y1="5" x2="26" y2="15" stroke="#9AA6BD" stroke-width="3" stroke-linecap="round"/>
<rect id="foot-l" x="-31" y="15" width="10" height="3" rx="1.5" fill="#C9D2E3"/>
<rect id="foot-r" x="21" y="15" width="10" height="3" rx="1.5" fill="#C9D2E3"/>
<polygon points="-14,-17 -17,0 -17,10 17,10 17,0 14,-17" fill="#E8ECF4" stroke="#9AA6BD" stroke-width="1.5"/>
<rect x="-17" y="2" width="34" height="3" fill="#FF8A3D"/>
<circle cx="0" cy="-6" r="5" fill="#1B2A4A" stroke="#5CC8FF" stroke-width="1.5"/>"""

GHOST = """<g fill="none" stroke="#5CC8FF" stroke-width="1.5" stroke-dasharray="3 3" opacity="0.75">
<line x1="-11" y1="5" x2="-26" y2="15"/><line x1="11" y1="5" x2="26" y2="15"/>
<polygon points="-14,-17 -17,0 -17,10 17,10 17,0 14,-17"/><circle cx="0" cy="-6" r="5"/></g>"""

LOGO = """<svg width="44" height="44" viewBox="-30 -30 60 60" aria-hidden="true"><rect x="-30" y="-30" width="60" height="60" rx="14" fill="#0F1630" stroke="#223058"/>
<path d="M-5,8 Q0,26 5,8 Z" fill="#FFB547"/><line x1="-8" y1="5" x2="-16" y2="16" stroke="#9AA6BD" stroke-width="2.5" stroke-linecap="round"/>
<line x1="8" y1="5" x2="16" y2="16" stroke="#9AA6BD" stroke-width="2.5" stroke-linecap="round"/>
<polygon points="-9,-12 -12,0 -12,7 12,7 12,0 9,-12" fill="#E8ECF4"/><circle cx="0" cy="-4" r="3.5" fill="#1B2A4A" stroke="#5CC8FF" stroke-width="1.2"/></svg>"""


def page():
    t, rng = terrain(), random.Random(1)
    ground = " ".join(f"{x},{y}" for x, y in t["ground"])
    pad = t["pad"]
    stars = "".join(f'<circle cx="{rng.uniform(0, 600):.0f}" cy="{rng.uniform(0, 230):.0f}" r="{rng.uniform(0.5, 1.2):.1f}" opacity="{rng.uniform(0.4, 0.9):.1f}"/>' for _ in range(40))
    (zx0, zy0), (zx1, zy1) = obs_to_pixel(CLICK_X[0], Y_RANGE[1]), obs_to_pixel(CLICK_X[1], Y_RANGE[0])
    flags = "".join(f'<line x1="{x}" y1="{pad["y"]}" x2="{x}" y2="{pad["y"] - 36}" stroke="#C9D2E3" stroke-width="1.5"/>'
                    f'<polygon points="{x},{pad["y"] - 36} {x + 14},{pad["y"] - 32} {x},{pad["y"] - 28}" fill="#FFB547"/>'
                    f'<circle cx="{x}" cy="{pad["y"] - 38}" r="1.8" fill="#FF6B6B"/>' for x in (pad["x1"], pad["x2"]))
    cards = "".join(f"""<label class="card"><input type="radio" name="agent" value="{r}"{" checked" if r == "pbrs" else ""}>
<span class="card-text"><span class="card-name">{r}</span><span class="card-desc">{d}</span>
<span class="card-stat">{zone_success(r):.0%} land in zone</span></span></label>""" for r, d in REWARDS.items())
    return f"""<div class="rl">
<header><div class="brand">{LOGO}<div><h1>Rocket Land</h1><p>Click the sky, pick a reward, and watch a trained agent land.</p></div></div></header>
<fieldset><legend>Agent · training reward</legend><div class="cards">{cards}</div></fieldset>
<div class="scene-wrap"><svg class="scene" viewBox="0 0 600 400" role="img" aria-label="Lunar scene: click inside the launch zone to drop the rocket there">
<rect width="600" height="400" fill="#070B18"/><g fill="#FFFFFF">{stars}</g>
<circle cx="526" cy="78" r="26" fill="#161E3D"/><circle cx="518" cy="70" r="6" fill="#1D2650"/><circle cx="534" cy="88" r="4" fill="#1D2650"/>
<rect id="zone" x="{zx0:.0f}" y="{zy0:.0f}" width="{zx1 - zx0:.0f}" height="{zy1 - zy0:.0f}" rx="6" fill="#5CC8FF" fill-opacity="0.04" stroke="#5CC8FF" stroke-opacity="0.4" stroke-dasharray="4 4"/>
<text x="{zx0 + 8:.0f}" y="{zy0 + 14:.0f}" font-family="JetBrains Mono, monospace" font-size="8" fill="#7FB4DA" letter-spacing="1">LAUNCH ZONE · CLICK TO DROP</text>
<polygon points="0,400 {ground} 600,400" fill="#1E2440"/><polyline points="{ground}" fill="none" stroke="#4A5580" stroke-width="1.5" stroke-linejoin="round"/>
<rect x="{pad["x1"]}" y="{pad["y"] - 2}" width="{pad["x2"] - pad["x1"]}" height="4" rx="1" fill="#5CC8FF"/>{flags}
<polyline id="trail" points="" fill="none" stroke="#5CC8FF" stroke-opacity="0.55" stroke-width="1.2" stroke-dasharray="1.5 5" stroke-linecap="round"/>
<g id="ghost" style="display:none">{GHOST}</g>
<g id="burst" style="display:none"><circle r="30" fill="#FF6B6B" opacity="0.14"/><circle r="18" fill="#FF8A3D" opacity="0.5"/><circle r="9" fill="#FFD27A"/>
<g fill="#FFB547"><circle cx="-30" cy="-22" r="2.2"/><circle cx="28" cy="-28" r="1.8"/><circle cx="38" cy="-6" r="2"/><circle cx="-38" cy="-2" r="1.6"/><circle cx="8" cy="-38" r="1.4"/></g></g>
<g id="rocket" style="display:none">{ROCKET}</g>
<g font-family="JetBrains Mono, monospace"><rect x="10" y="10" width="100" height="90" rx="6" fill="#0F1630" fill-opacity="0.9" stroke="#223058"/>
<text x="18" y="24" font-size="7" fill="#9AA6C4" letter-spacing="1">SCORE</text><text id="h-score" x="18" y="44" font-size="20" font-weight="600" fill="#9AA6C4">—</text>
<text x="18" y="60" font-size="7" fill="#9AA6C4">ALT</text><text id="h-alt" x="60" y="60" font-size="7" fill="#E6EAF5">—</text>
<text x="18" y="71" font-size="7" fill="#9AA6C4">SPEED</text><text id="h-speed" x="60" y="71" font-size="7" fill="#E6EAF5">—</text>
<text x="18" y="82" font-size="7" fill="#9AA6C4">TILT</text><text id="h-tilt" x="60" y="82" font-size="7" fill="#E6EAF5">—</text>
<text x="18" y="93" font-size="7" fill="#9AA6C4">FUEL</text><text id="h-fuel" x="60" y="93" font-size="7" fill="#E6EAF5">—</text>
<text id="h-step" x="590" y="20" font-size="8" fill="#9AA6C4" text-anchor="end"></text></g>
<g id="card" style="display:none" font-family="JetBrains Mono, monospace" text-anchor="middle"><rect x="216" y="40" width="168" height="62" rx="10" stroke-width="1.5"/>
<text id="card-title" x="300" y="64" font-weight="700" font-size="14" letter-spacing="2"></text><text id="card-score" x="300" y="92" font-weight="600" font-size="20" fill="#E6EAF5"></text></g>
</svg></div>
<div class="bar"><p id="status" aria-live="polite">Click inside the launch zone to drop the rocket.</p>
<div class="controls"><button type="button" id="replay" disabled>
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 3v6h6"/></svg>Replay</button>
<div class="speed" role="radiogroup" aria-label="Playback speed"><label><input type="radio" name="speed" value="1" checked>1×</label><label><input type="radio" name="speed" value="2">2×</label></div></div></div>
</div>"""


CSS = """
.rl{color:#E6EAF5;font-family:'Space Grotesk',system-ui,sans-serif;display:flex;flex-direction:column;gap:24px;max-width:1200px;margin:0 auto;padding:8px 0}
.rl header .brand{display:flex;align-items:center;gap:14px}
.rl h1{margin:0;font-size:28px;font-weight:700;letter-spacing:-0.01em;color:#E6EAF5}
.rl header p{margin:0;font-size:15px;color:#9AA6C4}
.rl fieldset{border:0;margin:0;padding:0}
.rl legend{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#9AA6C4;padding:0;margin-bottom:10px}
.rl .cards{display:flex;flex-wrap:wrap;gap:12px}
.rl .card{flex:1 1 220px;display:flex;gap:12px;padding:16px;border-radius:12px;border:1px solid #223058;background:#0F1630;cursor:pointer}
.rl .card:has(input:checked){border:1.5px solid #5CC8FF;background:#122045}
.rl .card:has(input:focus-visible){outline:2px solid #FFB547;outline-offset:3px}
.rl .card input{appearance:auto;-webkit-appearance:radio;border-radius:50%;accent-color:#5CC8FF;width:18px;height:18px;margin:2px 0 0 0;flex-shrink:0}
.rl .card-text{display:flex;flex-direction:column;gap:4px}
.rl .card-name{font-weight:600;font-size:17px}
.rl .card-desc{font-size:13px;color:#9AA6C4;line-height:1.4}
.rl .card-stat{font-family:'JetBrains Mono',monospace;font-size:12px;color:#9AA6C4;margin-top:4px}
.rl .card:has(input:checked) .card-stat{color:#5CC8FF}
.rl .scene-wrap{border-radius:16px;overflow:hidden;border:1px solid #223058;background:#070B18;width:100%;max-width:max(480px, calc((100vh - 360px) * 1.5));margin:0 auto}  /* header + cards + bar ≈ 360 px: keep the controls above the fold */
.rl .scene{display:block;width:100%;height:auto;cursor:crosshair}
.rl .bar{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:16px}
.rl #status{margin:0;font-size:16px}
.rl #status strong{color:#5CC8FF}
.rl .controls{display:flex;gap:10px;align-items:center}
.rl #replay{display:flex;align-items:center;gap:8px;min-height:44px;padding:0 18px;border-radius:10px;border:1px solid #223058;background:#0F1630;color:#E6EAF5;font:500 15px 'Space Grotesk',sans-serif;cursor:pointer}
.rl #replay:disabled{border-color:#1A2444;background:#0B1126;color:#6F7C9C;cursor:default}
.rl .speed{display:flex;border:1px solid #223058;border-radius:10px;overflow:hidden}
.rl .speed label{position:relative;min-height:44px;min-width:52px;display:flex;align-items:center;justify-content:center;background:#0F1630;color:#E6EAF5;font:600 15px 'JetBrains Mono',monospace;cursor:pointer}
.rl .speed label:has(input:checked){background:#5CC8FF;color:#070B18}
.rl .speed label:has(input:focus-visible){outline:2px solid #FFB547;outline-offset:-2px}
.rl .speed input{position:absolute;opacity:0;width:1px;height:1px}
"""

PAGE_CSS = "body, .gradio-container, .main, .contain {background:#070B18 !important} footer {opacity:.4}"
FONTS = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">"""

JS = r"""
const $ = s => element.querySelector(s);
const svg = $('svg.scene'), rocket = $('#rocket'), ghost = $('#ghost'), trail = $('#trail'), burst = $('#burst');
const card = $('#card'), status = $('#status'), replayBtn = $('#replay'), zone = $('#zone');
const z = ['x', 'y', 'width', 'height'].map(k => +zone.getAttribute(k));
const ENDINGS = {landed: ['LANDED', '#0F2A1F', '#4ADE80'], crashed: ['CRASHED', '#2E1218', '#FF6B6B'], timeout: ['OUT OF TIME', '#1E1838', '#C4B5FD']};
const ENGINE = ['Engines off.', 'Right thruster.', 'Main engine on.', 'Left thruster.'];
let run = null, click = null, agentName = 'pbrs', speed = 1, token = 0, raf = 0, t0 = 0, last = -1;

const show = (el, on) => { el.style.display = on ? '' : 'none'; };
const signed = v => (v >= 0 ? '+' : '−') + Math.abs(Math.round(v));
const place = (g, x, y, deg = 0) => g.setAttribute('transform', `translate(${x} ${y}) rotate(${deg})`);
const agent = () => element.querySelector('input[name=agent]:checked').value;

function draw(i) {
  const s = run.states[i], deg = -s[2] * 180 / Math.PI;  // Box2D angle is counter-clockwise, SVG rotate is clockwise
  place(rocket, s[0], s[1], deg);
  show($('#flame'), s[5] === 2); show($('#puff-r'), s[5] === 1); show($('#puff-l'), s[5] === 3);
  $('#foot-l').setAttribute('fill', s[3] ? '#4ADE80' : '#C9D2E3');
  $('#foot-r').setAttribute('fill', s[4] ? '#4ADE80' : '#C9D2E3');
  trail.setAttribute('points', run.states.slice(0, i + 1).map(p => p[0] + ',' + p[1]).join(' '));
  const sc = $('#h-score'); sc.textContent = signed(s[6]); sc.setAttribute('fill', s[6] >= 0 ? '#4ADE80' : '#FF6B6B');
  $('#h-alt').textContent = s[7].toFixed(2);
  $('#h-speed').textContent = s[8].toFixed(2); $('#h-speed').setAttribute('fill', s[8] > 1 ? '#FFB547' : '#E6EAF5');  // amber = fast
  $('#h-tilt').textContent = Math.round(Math.abs(deg)) + '°'; $('#h-tilt').setAttribute('fill', Math.abs(deg) > 20 ? '#FFB547' : '#E6EAF5');
  $('#h-fuel').textContent = s[9].toFixed(1);
  $('#h-step').textContent = `STEP ${i} / 1000`;
  if (i < run.states.length - 1) status.innerHTML = `<strong>${agentName}</strong> is descending from (${run.start[0]}, ${run.start[1]}). ${ENGINE[s[5]]}`;
}

function finish() {
  const s = run.states[run.states.length - 1], [title, bg, fg] = ENDINGS[run.outcome];
  card.querySelector('rect').setAttribute('fill', bg); card.querySelector('rect').setAttribute('stroke', fg);
  $('#card-title').textContent = title; $('#card-title').setAttribute('fill', fg);
  $('#card-score').textContent = signed(run.score);
  show(card, true);
  if (run.outcome === 'crashed') { place(burst, s[0], s[1]); show(burst, true); }
  rocket.style.opacity = run.outcome === 'timeout' ? 0.5 : 1;
  const verb = {landed: 'landed', crashed: 'crashed', timeout: 'ran out of time (1000 steps)'}[run.outcome];
  status.innerHTML = `<strong>${agentName}</strong> ${verb} from (${run.start[0]}, ${run.start[1]}), true score ${signed(run.score)}.` +
    (run.clamped ? ' Your click was outside the launch zone, so it started from the nearest point inside.' : '');
}

function tick(now) {  // time-based: the sim runs at 50 steps/s, so step i is shown at i * 20 ms / speed
  const i = Math.min(run.states.length - 1, Math.floor((now - t0) * speed / 20));
  if (i !== last) { draw(i); last = i; }
  if (i < run.states.length - 1) raf = requestAnimationFrame(tick); else finish();
}

function play() {
  cancelAnimationFrame(raf);
  show(card, false); show(burst, false); show(ghost, false); show(rocket, true); rocket.style.opacity = 1;
  last = -1; t0 = performance.now(); replayBtn.disabled = false;
  raf = requestAnimationFrame(tick);
}

async function launch(px, py) {
  const my = ++token;
  click = [px, py]; agentName = agent();
  cancelAnimationFrame(raf);
  show(rocket, false); show(card, false); show(burst, false); trail.setAttribute('points', '');
  place(ghost, Math.min(Math.max(px, z[0]), z[0] + z[2]), Math.min(Math.max(py, z[1]), z[1] + z[3])); show(ghost, true);
  status.textContent = 'Fuelling… running the episode on the server.';
  try {
    const res = await server.trajectory({px, py, agent: agentName});
    if (my !== token) return;  // a newer click or agent change won
    run = res; play();
  } catch (e) {
    if (my === token) status.textContent = 'Something went wrong running that episode. Try another spot.';
  }
}

svg.addEventListener('click', e => {
  const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse());  // works at any displayed size
  launch(p.x, p.y);
});
element.querySelectorAll('input[name=agent]').forEach(r => r.addEventListener('change', () => { if (click) launch(...click); }));
element.querySelectorAll('input[name=speed]').forEach(r => r.addEventListener('change', () => {
  const now = performance.now(), i = Math.max(last, 0);
  speed = +r.value; t0 = now - i * 20 / speed;  // keep the current step when switching speed mid-replay
}));
replayBtn.addEventListener('click', () => { if (run) play(); });
"""

with gr.Blocks(title="Rocket Land") as demo:
    gr.HTML(page(), css_template=CSS, js_on_load=JS, server_functions=[trajectory], container=False, padding=False, apply_default_css=False)

if __name__ == "__main__":
    demo.launch(css=PAGE_CSS, head=FONTS)
