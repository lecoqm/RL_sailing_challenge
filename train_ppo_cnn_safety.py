#!/usr/bin/env python3
"""
PPO/CNN + safety shield for the stable_v2_RL_sailing_challenge.

Run from the root of the official repository:
    python train_ppo_cnn_safety.py --total-steps 1000000 --n-envs 16

Then export a Codabench-compatible NumPy-only agent:
    python export_policy_to_agent.py --checkpoint runs/ppo_cnn_safety/best_model.pt --out my_agent.py

Local evaluation:
    cp my_agent.py src/agents/my_agent.py
    cd src
    python evaluate_submission.py agents/my_agent.py --seeds 1 --num-seeds 20
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Make official src imports work when this file is launched from repo root.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from env_sailing import SailingEnv  # type: ignore
from wind_scenarios import WIND_SCENARIOS  # type: ignore

GRID = 128
IMG = 32
BLOCK = GRID // IMG
WORLD_SIZE = GRID * GRID
WIND_SIZE = WORLD_SIZE * 2
GOAL = np.array([64.0, 127.0], dtype=np.float32)
ACTIONS = np.array(
    [
        [0, 1],    # 0 N
        [1, 1],    # 1 NE
        [1, 0],    # 2 E
        [1, -1],   # 3 SE
        [0, -1],   # 4 S
        [-1, -1],  # 5 SW
        [-1, 0],   # 6 W
        [-1, 1],   # 7 NW
        [0, 0],    # 8 stay
    ],
    dtype=np.float32,
)


def parse_observation(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pos, vel, wind_field, world_map from the official flat observation."""
    obs = np.asarray(obs, dtype=np.float32)
    pos = obs[0:2].astype(np.float32)
    vel = obs[2:4].astype(np.float32)
    wind = obs[6 : 6 + WIND_SIZE].reshape(GRID, GRID, 2).astype(np.float32)
    world = obs[6 + WIND_SIZE : 6 + WIND_SIZE + WORLD_SIZE].reshape(GRID, GRID).astype(np.float32)
    return pos, vel, wind, world


def block_mean_2d(x: np.ndarray) -> np.ndarray:
    return x.reshape(IMG, BLOCK, IMG, BLOCK).mean(axis=(1, 3))


def block_max_2d(x: np.ndarray) -> np.ndarray:
    return x.reshape(IMG, BLOCK, IMG, BLOCK).max(axis=(1, 3))


def preprocess_one(obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build CNN image channels and vector features from one official observation."""
    pos, vel, wind, world = parse_observation(obs)
    x, y = float(pos[0]), float(pos[1])
    vx, vy = float(vel[0]), float(vel[1])
    wx, wy = float(obs[4]), float(obs[5])

    wind_x = np.clip(block_mean_2d(wind[:, :, 0]) / 10.0, -2.0, 2.0)
    wind_y = np.clip(block_mean_2d(wind[:, :, 1]) / 10.0, -2.0, 2.0)
    obstacle = block_max_2d(world)

    boat = np.zeros((IMG, IMG), dtype=np.float32)
    bx = int(np.clip(x / BLOCK, 0, IMG - 1))
    by = int(np.clip(y / BLOCK, 0, IMG - 1))
    boat[by, bx] = 1.0

    goal = np.zeros((IMG, IMG), dtype=np.float32)
    goal[IMG - 1, max(0, IMG // 2 - 1) : min(IMG, IMG // 2 + 2)] = 1.0

    image = np.stack([wind_x, wind_y, obstacle, boat, goal], axis=0).astype(np.float32)

    to_goal = GOAL - pos
    dist = float(np.linalg.norm(to_goal) + 1e-6)
    angle = math.atan2(float(to_goal[1]), float(to_goal[0]))
    vec = np.array(
        [
            x / 127.0,
            y / 127.0,
            vx / 8.0,
            vy / 8.0,
            wx / 10.0,
            wy / 10.0,
            to_goal[0] / 127.0,
            to_goal[1] / 127.0,
            dist / 180.0,
            math.sin(angle),
            math.cos(angle),
            (x - 64.0) / 64.0,
        ],
        dtype=np.float32,
    )
    return image, vec


def preprocess_batch(obs_batch: List[np.ndarray] | np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs, vecs = zip(*(preprocess_one(obs) for obs in obs_batch))
    img_t = torch.as_tensor(np.stack(imgs), dtype=torch.float32, device=device)
    vec_t = torch.as_tensor(np.stack(vecs), dtype=torch.float32, device=device)
    return img_t, vec_t


def sailing_efficiency(direction: np.ndarray, wind: np.ndarray) -> float:
    wind_norm = np.linalg.norm(wind)
    direction_norm = np.linalg.norm(direction)
    if wind_norm < 1e-10 or direction_norm < 1e-10:
        # Official env maps stay direction to a default direction before efficiency.
        direction = np.array([1.0, 0.0], dtype=np.float32)
        direction_norm = 1.0
    wind_direction = wind / max(wind_norm, 1e-10)
    boat_direction = direction / max(direction_norm, 1e-10)
    wind_from = -wind_direction
    wind_angle = math.acos(float(np.clip(np.dot(wind_from, boat_direction), -1.0, 1.0)))
    if wind_angle < math.pi / 4:
        eff = 0.05
    elif wind_angle < math.pi / 2:
        eff = 0.5 + 0.5 * (wind_angle - math.pi / 4) / (math.pi / 4)
    elif wind_angle < 3 * math.pi / 4:
        eff = 1.0
    else:
        eff = 1.0 - 0.5 * (wind_angle - 3 * math.pi / 4) / (math.pi / 4)
        eff = max(0.5, eff)
    return float(eff)


class SafetyShield:
    """One-step physics shield. It returns the proposed action if safe, else a safe alternative."""

    def __init__(self, boat_performance: float = 0.4, max_speed: float = 8.0, inertia_factor: float = 0.3):
        self.boat_performance = boat_performance
        self.max_speed = max_speed
        self.inertia_factor = inertia_factor

    def simulate(self, obs: np.ndarray, action: int) -> Tuple[np.ndarray, np.ndarray, bool, float]:
        pos, vel, _wind_field, world = parse_observation(obs)
        wind = obs[4:6].astype(np.float32)
        direction = ACTIONS[int(action)]
        wind_norm = float(np.linalg.norm(wind))
        eff = sailing_efficiency(direction, wind)
        theoretical = direction * eff * wind_norm * self.boat_performance
        speed = float(np.linalg.norm(theoretical))
        if speed > self.max_speed:
            theoretical = theoretical / speed * self.max_speed
        new_vel = theoretical + self.inertia_factor * (vel - theoretical)
        speed = float(np.linalg.norm(new_vel))
        if speed > self.max_speed:
            new_vel = new_vel / speed * self.max_speed
        new_vel_i = np.where(new_vel < 0, np.ceil(new_vel), np.floor(new_vel)).astype(np.int32)
        new_pos = np.clip(pos.astype(np.int32) + new_vel_i, [0, 0], [GRID - 1, GRID - 1]).astype(np.int32)
        unsafe = self._hits_obstacle(pos.astype(np.int32), new_pos, world)
        score = self._score_action(pos, vel, new_pos.astype(np.float32), new_vel_i.astype(np.float32), world, unsafe)
        return new_pos.astype(np.float32), new_vel_i.astype(np.float32), unsafe, score

    def _hits_obstacle(self, old_pos: np.ndarray, new_pos: np.ndarray, world: np.ndarray) -> bool:
        x1, y1 = int(new_pos[0]), int(new_pos[1])
        if world[y1, x1] > 0.5:
            return True
        # More conservative than the official env: sample the segment and reject crossings.
        dx = int(new_pos[0] - old_pos[0])
        dy = int(new_pos[1] - old_pos[1])
        n = max(abs(dx), abs(dy), 1)
        for k in range(1, n + 1):
            x = int(round(old_pos[0] + dx * k / n))
            y = int(round(old_pos[1] + dy * k / n))
            if 0 <= x < GRID and 0 <= y < GRID and world[y, x] > 0.5:
                return True
        return False

    def _local_obstacle_penalty(self, pos: np.ndarray, world: np.ndarray, radius: int = 4) -> float:
        x, y = int(pos[0]), int(pos[1])
        x0, x1 = max(0, x - radius), min(GRID, x + radius + 1)
        y0, y1 = max(0, y - radius), min(GRID, y + radius + 1)
        patch = world[y0:y1, x0:x1]
        if patch.max() < 0.5:
            return 0.0
        ys, xs = np.where(patch > 0.5)
        if len(xs) == 0:
            return 0.0
        d2 = (xs + x0 - x) ** 2 + (ys + y0 - y) ** 2
        d = math.sqrt(float(np.min(d2)))
        return max(0.0, radius - d) / radius

    def _score_action(
        self,
        old_pos: np.ndarray,
        old_vel: np.ndarray,
        new_pos: np.ndarray,
        new_vel: np.ndarray,
        world: np.ndarray,
        unsafe: bool,
    ) -> float:
        if unsafe:
            return -1e9
        old_dist = float(np.linalg.norm(GOAL - old_pos))
        new_dist = float(np.linalg.norm(GOAL - new_pos))
        progress = old_dist - new_dist
        dy = float(new_pos[1] - old_pos[1])
        center_pen = abs(float(new_pos[0]) - 64.0) / 64.0
        obstacle_pen = self._local_obstacle_penalty(new_pos, world, radius=5)
        return 1.8 * progress + 0.35 * dy + 0.10 * float(new_vel[1]) - 0.05 * center_pen - 0.60 * obstacle_pen

    def filter(self, obs: np.ndarray, proposed_action: int) -> int:
        proposed_action = int(proposed_action)
        _, _, unsafe, _ = self.simulate(obs, proposed_action)
        if not unsafe:
            return proposed_action
        best_a, best_score = 8, -1e18
        for a in range(9):
            _, _, u, s = self.simulate(obs, a)
            if not u and s > best_score:
                best_a, best_score = a, s
        return int(best_a)


class PolicyNet(nn.Module):
    def __init__(self, image_channels: int = 5, vector_dim: int = 12, n_actions: int = 9):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, image_channels, IMG, IMG)).shape[1]
        self.vec_net = nn.Sequential(nn.Linear(vector_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(n_flat + 64, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

        # Mildly conservative initialization for PPO.
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, image: torch.Tensor, vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_img = self.cnn(image)
        z_vec = self.vec_net(vec)
        z = self.trunk(torch.cat([z_img, z_vec], dim=1))
        return self.actor(z), self.critic(z).squeeze(-1)

    def get_action_and_value(
        self, image: torch.Tensor, vec: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(image, vec)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


@dataclass
class PPOConfig:
    total_steps: int = 1_000_000
    n_envs: int = 16
    rollout_steps: int = 256
    learning_rate: float = 2.5e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ppo_epochs: int = 4
    minibatch_size: int = 512
    clip_coef: float = 0.20
    ent_coef: float = 0.01
    vf_coef: float = 0.50
    max_grad_norm: float = 0.50
    seed: int = 1
    eval_interval: int = 50_000
    eval_seeds: int = 20
    use_safety: bool = True
    out_dir: str = "runs/ppo_cnn_safety"


class SailingSlot:
    def __init__(self, seed: int, use_safety: bool = True):
        self.rng = np.random.default_rng(seed)
        self.use_safety = use_safety
        self.shield = SafetyShield()
        self.env: Optional[SailingEnv] = None
        self.obs: Optional[np.ndarray] = None
        self.prev_y = 0.0
        self.reset()

    def _new_env(self) -> SailingEnv:
        name = str(self.rng.choice(list(WIND_SCENARIOS.keys())))
        scenario = copy.deepcopy(WIND_SCENARIOS[name])
        # Small domain randomization to reduce overfit to the three public scenarios.
        scenario["wind_init_params"]["base_max_rotation_angle_degree"] = float(
            scenario["wind_init_params"].get("base_max_rotation_angle_degree", 10) + self.rng.uniform(-4, 6)
        )
        scenario["wind_evol_params"]["mean_rotation_angle_degree"] = float(
            scenario["wind_evol_params"].get("mean_rotation_angle_degree", 3) + self.rng.uniform(-0.8, 0.8)
        )
        scenario["wind_evol_params"]["std_rotation_angle_degree"] = float(
            max(0.1, scenario["wind_evol_params"].get("std_rotation_angle_degree", 0.8) + self.rng.uniform(-0.25, 0.35))
        )
        return SailingEnv(
            wind_init_params=scenario["wind_init_params"],
            wind_evol_params=scenario["wind_evol_params"],
            render_mode=None,
            max_horizon=500,
        )

    def reset(self) -> np.ndarray:
        self.env = self._new_env()
        seed = int(self.rng.integers(0, 2**31 - 1))
        obs, _ = self.env.reset(seed=seed)
        self.obs = obs
        self.prev_y = float(obs[1])
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        assert self.env is not None and self.obs is not None
        old_obs = self.obs
        action = int(action)
        if self.use_safety:
            action = self.shield.filter(old_obs, action)
        next_obs, official_reward, terminated, truncated, info = self.env.step(action)
        done = bool(terminated or truncated)
        shaped = self.shape_reward(old_obs, next_obs, float(official_reward), info, done)
        if done:
            next_obs = self.reset()
        else:
            self.obs = next_obs
        return next_obs, shaped, done, info

    def shape_reward(self, old_obs: np.ndarray, new_obs: np.ndarray, official: float, info: Dict, done: bool) -> float:
        old_pos = old_obs[0:2].astype(np.float32)
        new_pos = new_obs[0:2].astype(np.float32)
        old_d = float(np.linalg.norm(GOAL - old_pos))
        new_d = float(np.linalg.norm(GOAL - new_pos))
        progress = old_d - new_d
        dy = float(new_pos[1] - old_pos[1])
        vy = float(new_obs[3])
        reward = 0.08 * progress + 0.015 * dy + 0.006 * vy - 0.006
        if new_pos[1] < old_pos[1]:
            reward -= 0.03
        # Penalize getting close to the island.
        _, _, _, world = parse_observation(new_obs)
        near_pen = self.shield._local_obstacle_penalty(new_pos, world, radius=5)
        reward -= 0.05 * near_pen
        if official > 0:
            reward += 10.0
        if done and official <= 0 and bool(info.get("is_stuck", False)):
            reward -= 4.0
        return float(np.clip(reward, -5.0, 12.0))


def evaluate_policy(model: PolicyNet, device: torch.device, n_seeds: int, use_safety: bool = True) -> Dict[str, float]:
    shield = SafetyShield()
    successes: List[bool] = []
    rewards: List[float] = []
    steps_all: List[int] = []
    model.eval()
    with torch.no_grad():
        for scenario_name, scenario in WIND_SCENARIOS.items():
            for seed in range(1, n_seeds + 1):
                env = SailingEnv(
                    wind_init_params=copy.deepcopy(scenario["wind_init_params"]),
                    wind_evol_params=copy.deepcopy(scenario["wind_evol_params"]),
                    max_horizon=500,
                    render_mode=None,
                )
                obs, _ = env.reset(seed=seed)
                total_disc = 0.0
                success = False
                for step in range(500):
                    img, vec = preprocess_batch([obs], device)
                    logits, _ = model(img, vec)
                    action = int(torch.argmax(logits, dim=1).item())
                    if use_safety:
                        action = shield.filter(obs, action)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_disc += float(reward) * (0.995 ** step)
                    if terminated or truncated:
                        success = bool(reward > 0 or info.get("distance_to_goal", 999) < 1.5)
                        steps_all.append(step + 1)
                        break
                else:
                    steps_all.append(500)
                successes.append(success)
                rewards.append(total_disc)
    model.train()
    return {
        "success_rate": float(np.mean(successes)),
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps_all)),
    }


def train(cfg: PPOConfig) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    envs = [SailingSlot(cfg.seed * 1000 + i, use_safety=cfg.use_safety) for i in range(cfg.n_envs)]
    obs = [e.obs for e in envs]
    assert all(o is not None for o in obs)
    obs = [o for o in obs if o is not None]

    model = PolicyNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    num_updates = cfg.total_steps // (cfg.n_envs * cfg.rollout_steps)
    global_step = 0
    best_score = -1e18

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1.0) / max(1, num_updates)
        optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        obs_buf = list(obs)
        img_buf, vec_buf, action_buf, logprob_buf, reward_buf, done_buf, value_buf = [], [], [], [], [], [], []

        for _ in range(cfg.rollout_steps):
            global_step += cfg.n_envs
            img_t, vec_t = preprocess_batch(obs, device)
            with torch.no_grad():
                actions, logprobs, _, values = model.get_action_and_value(img_t, vec_t)
            actions_np = actions.cpu().numpy()

            next_obs, rewards, dones = [], [], []
            for i, env in enumerate(envs):
                no, r, d, _info = env.step(int(actions_np[i]))
                next_obs.append(no)
                rewards.append(r)
                dones.append(d)

            img_buf.append(img_t.cpu().numpy())
            vec_buf.append(vec_t.cpu().numpy())
            action_buf.append(actions_np)
            logprob_buf.append(logprobs.cpu().numpy())
            reward_buf.append(np.array(rewards, dtype=np.float32))
            done_buf.append(np.array(dones, dtype=np.float32))
            value_buf.append(values.cpu().numpy())
            obs = next_obs

        with torch.no_grad():
            next_img, next_vec = preprocess_batch(obs, device)
            _, next_values = model.forward(next_img, next_vec)
            next_values_np = next_values.cpu().numpy()

        rewards_np = np.asarray(reward_buf, dtype=np.float32)        # T,N
        dones_np = np.asarray(done_buf, dtype=np.float32)            # T,N
        values_np = np.asarray(value_buf, dtype=np.float32)          # T,N
        advantages = np.zeros_like(rewards_np, dtype=np.float32)
        lastgaelam = np.zeros(cfg.n_envs, dtype=np.float32)
        for t in reversed(range(cfg.rollout_steps)):
            if t == cfg.rollout_steps - 1:
                next_nonterminal = 1.0 - dones_np[t]
                next_values_t = next_values_np
            else:
                next_nonterminal = 1.0 - dones_np[t + 1]
                next_values_t = values_np[t + 1]
            delta = rewards_np[t] + cfg.gamma * next_values_t * next_nonterminal - values_np[t]
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values_np

        b_img = torch.as_tensor(np.asarray(img_buf).reshape(-1, 5, IMG, IMG), dtype=torch.float32, device=device)
        b_vec = torch.as_tensor(np.asarray(vec_buf).reshape(-1, 12), dtype=torch.float32, device=device)
        b_actions = torch.as_tensor(np.asarray(action_buf).reshape(-1), dtype=torch.long, device=device)
        b_logprobs = torch.as_tensor(np.asarray(logprob_buf).reshape(-1), dtype=torch.float32, device=device)
        b_advantages = torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
        b_returns = torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device)
        b_values = torch.as_tensor(values_np.reshape(-1), dtype=torch.float32, device=device)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        batch_size = cfg.n_envs * cfg.rollout_steps
        inds = np.arange(batch_size)

        pg_losses, v_losses, entropies = [], [], []
        for _epoch in range(cfg.ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, cfg.minibatch_size):
                mb = inds[start : start + cfg.minibatch_size]
                _, newlogprob, entropy, newvalue = model.get_action_and_value(b_img[mb], b_vec[mb], b_actions[mb])
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb]) ** 2
                v_clipped = b_values[mb] + torch.clamp(newvalue - b_values[mb], -cfg.clip_coef, cfg.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[mb]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()

                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropies.append(float(entropy_loss.item()))

        if update == 1 or global_step % 10_000 < cfg.n_envs * cfg.rollout_steps:
            print(
                f"step={global_step:>9} update={update:>4}/{num_updates} "
                f"pg={np.mean(pg_losses):+.4f} vf={np.mean(v_losses):.4f} ent={np.mean(entropies):.3f}",
                flush=True,
            )

        if update == num_updates or global_step % cfg.eval_interval < cfg.n_envs * cfg.rollout_steps:
            metrics = evaluate_policy(model, device, n_seeds=cfg.eval_seeds, use_safety=cfg.use_safety)
            score = metrics["mean_reward"] + 20.0 * metrics["success_rate"]
            print(f"EVAL step={global_step}: {metrics}", flush=True)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "metrics": metrics,
            }
            torch.save(ckpt, out_dir / "last_model.pt")
            if score > best_score:
                best_score = score
                torch.save(ckpt, out_dir / "best_model.pt")
                print(f"saved best_model.pt score={best_score:.3f}", flush=True)


def parse_args() -> PPOConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--rollout-steps", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-interval", type=int, default=50_000)
    p.add_argument("--eval-seeds", type=int, default=20)
    p.add_argument("--out-dir", type=str, default="runs/ppo_cnn_safety")
    p.add_argument("--no-safety", action="store_true")
    a = p.parse_args()
    return PPOConfig(
        total_steps=a.total_steps,
        n_envs=a.n_envs,
        rollout_steps=a.rollout_steps,
        learning_rate=a.learning_rate,
        seed=a.seed,
        eval_interval=a.eval_interval,
        eval_seeds=a.eval_seeds,
        out_dir=a.out_dir,
        use_safety=not a.no_safety,
    )


if __name__ == "__main__":
    train(parse_args())
