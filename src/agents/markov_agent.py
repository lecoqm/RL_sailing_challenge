"""
Markov-wind MPC agent for the RL Sailing Challenge.

Idea:
- keep the previous full wind field in memory;
- estimate the Markov transition of the wind as a global rotation;
- predict the next wind fields over a short horizon;
- run a small beam-search MPC using the same sailing physics as the environment;
- add a safety shield around the island.

This is a model-based RL/control agent: it uses the Markov state transition observed online
instead of training a neural network offline.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    # Codabench-style import, if available.
    from evaluator.base_agent import BaseAgent  # type: ignore
except Exception:
    # Local repository import.
    from agents.base_agent import BaseAgent  # type: ignore


class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()

        # Environment constants from env_sailing.py
        self.boat_performance = 0.4
        self.max_speed = 8.0
        self.inertia_factor = 0.3

        # MPC parameters. Keep these modest: act() is called at every step.
        self.horizon = 5
        self.beam_width = 28
        self.safety_radius = 4

        # Memory used to exploit the Markov wind process.
        self.prev_wind_field: Optional[np.ndarray] = None
        self.wind_rotation_estimate = 0.0
        self.side: Optional[str] = None  # "left" or "right" while bypassing the island.

        # Cached maps.
        self._inflated_world: Optional[np.ndarray] = None
        self._world_shape: Optional[Tuple[int, int]] = None
        self._world_sum: Optional[float] = None
        self._obstacle_bounds: Optional[Tuple[int, int, int, int]] = None

        self.actions = np.array(
            [
                [0, 1],    # 0: North
                [1, 1],    # 1: North-East
                [1, 0],    # 2: East
                [1, -1],   # 3: South-East
                [0, -1],   # 4: South
                [-1, -1],  # 5: South-West
                [-1, 0],   # 6: West
                [-1, 1],   # 7: North-West
                [0, 0],    # 8: Stay
            ],
            dtype=np.float32,
        )

    def reset(self) -> None:
        self.prev_wind_field = None
        self.wind_rotation_estimate = 0.0
        self.side = None

    def seed(self, seed: Optional[int] = None) -> None:
        self.np_random = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> int:
        pos, vel, wind_field, world = self._parse_observation(observation)
        h, w = world.shape
        goal = np.array([w // 2, h - 1], dtype=np.float32)

        self._update_cached_maps(world)
        inflated = self._inflated_world if self._inflated_world is not None else world > 0.5

        # Estimate and exploit the Markov wind transition.
        self._update_wind_transition_estimate(wind_field)
        predicted_winds = self._predict_wind_fields(wind_field, self.horizon)

        target = self._select_target(pos, goal, world, inflated, predicted_winds)
        action = self._mpc_action(pos, vel, goal, target, world, inflated, predicted_winds)

        # Store current wind at the end, so next call can estimate wind_t -> wind_t+1.
        self.prev_wind_field = wind_field.copy()
        return int(action)

    # -------------------------------------------------------------------------
    # Observation and wind-transition model
    # -------------------------------------------------------------------------

    def _parse_observation(self, observation: np.ndarray):
        obs = np.asarray(observation, dtype=np.float32)
        rest = obs.size - 6
        # Observation = 6 + 2*n*n + n*n = 6 + 3*n*n
        n = int(round(math.sqrt(rest / 3)))
        if 6 + 3 * n * n != obs.size:
            # Fallback for the official environment size.
            n = 128

        pos = np.array([obs[0], obs[1]], dtype=np.float32)
        vel = np.array([obs[2], obs[3]], dtype=np.float32)
        wind_flat_end = 6 + 2 * n * n
        wind_field = obs[6:wind_flat_end].reshape(n, n, 2)
        world = obs[wind_flat_end:wind_flat_end + n * n].reshape(n, n)
        return pos, vel, wind_field, world

    def _update_wind_transition_estimate(self, wind_field: np.ndarray) -> None:
        """Estimate the global rotation angle between previous and current wind fields."""
        if self.prev_wind_field is None or self.prev_wind_field.shape != wind_field.shape:
            # The public environment mean is 0.5 degrees per step. Use it as prior.
            self.wind_rotation_estimate = math.radians(0.5)
            return

        a = self.prev_wind_field.reshape(-1, 2).astype(np.float64)
        b = wind_field.reshape(-1, 2).astype(np.float64)

        dot = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
        cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

        # Robust global rotation estimate: atan2 of summed cross/dot.
        delta = math.atan2(float(np.sum(cross)), float(np.sum(dot)) + 1e-12)

        # Smooth a little; the environment noise is small, but hidden scenarios may differ.
        self.wind_rotation_estimate = 0.65 * self.wind_rotation_estimate + 0.35 * delta

    def _predict_wind_fields(self, wind_field: np.ndarray, horizon: int) -> List[np.ndarray]:
        """Predict wind fields assuming the Markov transition remains a global rotation."""
        fields = []
        for k in range(horizon):
            theta = self.wind_rotation_estimate * k
            c, s = math.cos(theta), math.sin(theta)
            pred = np.empty_like(wind_field)
            x = wind_field[:, :, 0]
            y = wind_field[:, :, 1]
            pred[:, :, 0] = x * c - y * s
            pred[:, :, 1] = x * s + y * c
            fields.append(pred)
        return fields

    # -------------------------------------------------------------------------
    # Global target selection: left/right bypass of the island
    # -------------------------------------------------------------------------

    def _update_cached_maps(self, world: np.ndarray) -> None:
        world_sum = float(np.sum(world))
        if self._world_shape == world.shape and self._world_sum == world_sum:
            return

        self._world_shape = world.shape
        self._world_sum = world_sum
        self._inflated_world = self._inflate_world(world, self.safety_radius)

        ys, xs = np.where(world > 0.5)
        if len(xs) == 0:
            self._obstacle_bounds = None
        else:
            self._obstacle_bounds = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))

    def _inflate_world(self, world: np.ndarray, radius: int) -> np.ndarray:
        blocked = world > 0.5
        inflated = blocked.copy()
        h, w = blocked.shape

        obstacle_y, obstacle_x = np.where(blocked)
        if len(obstacle_x) == 0:
            return inflated

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                yy = obstacle_y + dy
                xx = obstacle_x + dx
                mask = (0 <= yy) & (yy < h) & (0 <= xx) & (xx < w)
                inflated[yy[mask], xx[mask]] = True
        return inflated

    def _select_target(
        self,
        pos: np.ndarray,
        goal: np.ndarray,
        world: np.ndarray,
        inflated: np.ndarray,
        predicted_winds: List[np.ndarray],
    ) -> np.ndarray:
        """Pick a waypoint. Around the island, commit to either the left or right corridor."""
        if self._obstacle_bounds is None:
            return goal

        x_min, x_max, y_min, y_max = self._obstacle_bounds
        h, w = world.shape

        # If above the obstacle, go directly to goal.
        if pos[1] > y_max + 8:
            self.side = None
            return goal

        # If the direct line is not blocked, avoid unnecessary detours.
        if not self._line_hits_map(pos, goal, inflated):
            return goal

        left_wp = np.array([max(4, x_min - 12), min(h - 2, y_max + 12)], dtype=np.float32)
        right_wp = np.array([min(w - 5, x_max + 12), min(h - 2, y_max + 12)], dtype=np.float32)

        # Commit to a side while below / next to the island. This prevents dithering.
        if self.side == "left":
            return left_wp
        if self.side == "right":
            return right_wp

        left_score = self._corridor_score(pos, left_wp, predicted_winds[0], inflated)
        right_score = self._corridor_score(pos, right_wp, predicted_winds[0], inflated)
        self.side = "left" if left_score >= right_score else "right"
        return left_wp if self.side == "left" else right_wp

    def _corridor_score(
        self,
        pos: np.ndarray,
        waypoint: np.ndarray,
        wind_field: np.ndarray,
        inflated: np.ndarray,
    ) -> float:
        dist = np.linalg.norm(waypoint - pos)
        if dist < 1e-6:
            return 1e6

        direction = (waypoint - pos) / dist
        samples = 9
        total_eff = 0.0
        collision_penalty = 0.0

        for t in np.linspace(0.0, 1.0, samples):
            p = pos * (1.0 - t) + waypoint * t
            x, y = self._clip_xy(p, inflated.shape)
            if inflated[y, x]:
                collision_penalty += 100.0
            wind = wind_field[y, x]
            total_eff += self._sailing_efficiency(direction, wind)

        return 5.0 * (total_eff / samples) - 0.03 * dist - collision_penalty

    # -------------------------------------------------------------------------
    # MPC / approximate finite-horizon value maximization
    # -------------------------------------------------------------------------

    def _mpc_action(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        goal: np.ndarray,
        target: np.ndarray,
        world: np.ndarray,
        inflated: np.ndarray,
        predicted_winds: List[np.ndarray],
    ) -> int:
        # Each candidate is (score, pos, vel, first_action)
        candidates: List[Tuple[float, np.ndarray, np.ndarray, Optional[int]]] = [
            (0.0, pos.astype(np.float32), vel.astype(np.float32), None)
        ]

        for depth in range(self.horizon):
            new_candidates: List[Tuple[float, np.ndarray, np.ndarray, Optional[int]]] = []
            wind_field = predicted_winds[min(depth, len(predicted_winds) - 1)]

            for score, cpos, cvel, first_action in candidates:
                for action in range(9):
                    step_score, npos, nvel, bad = self._score_one_step(
                        cpos, cvel, action, goal, target, world, inflated, wind_field
                    )
                    if bad:
                        # Keep it technically possible but very unlikely.
                        step_score -= 1000.0

                    fa = action if first_action is None else first_action
                    # Discount future terms slightly, as in finite-horizon RL.
                    total = score + (0.92 ** depth) * step_score
                    new_candidates.append((total, npos, nvel, fa))

            # Beam pruning.
            new_candidates.sort(key=lambda z: z[0] + self._terminal_value(z[1], goal, target), reverse=True)
            candidates = new_candidates[: self.beam_width]

        if not candidates:
            return self._safe_greedy_action(pos, vel, goal, target, world, inflated, predicted_winds[0])

        best = max(candidates, key=lambda z: z[0] + self._terminal_value(z[1], goal, target))
        if best[3] is None:
            return self._safe_greedy_action(pos, vel, goal, target, world, inflated, predicted_winds[0])
        return int(best[3])

    def _score_one_step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        action: int,
        goal: np.ndarray,
        target: np.ndarray,
        world: np.ndarray,
        inflated: np.ndarray,
        wind_field: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray, bool]:
        h, w = world.shape
        old_goal_dist = float(np.linalg.norm(goal - pos))
        old_target_dist = float(np.linalg.norm(target - pos))

        npos, nvel, clipped = self._simulate_step(pos, vel, action, wind_field, (h, w))
        x, y = self._clip_xy(npos, world.shape)

        collision = bool(world[y, x] > 0.5)
        near_obstacle = bool(inflated[y, x])
        line_risk = self._line_hits_map(pos, npos, inflated)

        new_goal_dist = float(np.linalg.norm(goal - npos))
        new_target_dist = float(np.linalg.norm(target - npos))

        progress_target = old_target_dist - new_target_dist
        progress_goal = old_goal_dist - new_goal_dist

        to_target = target - pos
        to_target_norm = float(np.linalg.norm(to_target))
        if to_target_norm > 1e-6:
            unit_target = to_target / to_target_norm
        else:
            unit_target = np.array([0.0, 1.0], dtype=np.float32)

        vmg = float(np.dot(nvel, unit_target))
        wind = wind_field[self._clip_xy(pos, world.shape)[1], self._clip_xy(pos, world.shape)[0]]
        eff = self._sailing_efficiency(self.actions[action], wind)

        reward = 0.0
        reward += 5.0 * progress_target
        reward += 1.2 * progress_goal
        reward += 0.25 * vmg
        reward += 0.15 * eff

        # Penalize time and unstable behavior.
        reward -= 0.08
        if action == 8:
            reward -= 0.7
        if npos[1] < pos[1] - 1.5:
            reward -= 0.5
        if clipped:
            reward -= 6.0
        if near_obstacle:
            reward -= 18.0
        if line_risk:
            reward -= 22.0
        if collision:
            reward -= 500.0

        if new_goal_dist < 1.5:
            reward += 300.0

        bad = collision or line_risk
        return reward, npos, nvel, bad

    def _terminal_value(self, pos: np.ndarray, goal: np.ndarray, target: np.ndarray) -> float:
        return -0.50 * float(np.linalg.norm(target - pos)) - 0.06 * float(np.linalg.norm(goal - pos))

    def _safe_greedy_action(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        goal: np.ndarray,
        target: np.ndarray,
        world: np.ndarray,
        inflated: np.ndarray,
        wind_field: np.ndarray,
    ) -> int:
        best_action = 0
        best_score = -1e18
        for action in range(9):
            score, _, _, bad = self._score_one_step(pos, vel, action, goal, target, world, inflated, wind_field)
            if bad:
                score -= 1000.0
            if score > best_score:
                best_score = score
                best_action = action
        return int(best_action)

    # -------------------------------------------------------------------------
    # Physics copied from env_sailing.py / sailing_physics.py
    # -------------------------------------------------------------------------

    def _simulate_step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        action: int,
        wind_field: np.ndarray,
        shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        h, w = shape
        x, y = self._clip_xy(pos, shape)
        wind = wind_field[y, x]
        direction = self.actions[action]

        new_vel = self._calculate_new_velocity(vel, wind, direction)
        new_vel = np.where(new_vel < 0, np.ceil(new_vel), np.floor(new_vel)).astype(np.float32)

        raw_pos = pos + new_vel
        new_pos = np.clip(raw_pos, [0, 0], [w - 1, h - 1]).astype(np.float32)
        clipped = bool(np.any(np.abs(raw_pos - new_pos) > 1e-6))
        return new_pos, new_vel, clipped

    def _calculate_new_velocity(self, current_velocity: np.ndarray, wind: np.ndarray, direction: np.ndarray) -> np.ndarray:
        wind_norm = float(np.linalg.norm(wind))
        if wind_norm <= 1e-10:
            return self.inertia_factor * current_velocity

        wind_normalized = wind / wind_norm
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-10:
            direction_normalized = np.array([1.0, 0.0], dtype=np.float32)
        else:
            direction_normalized = direction / direction_norm

        efficiency = self._sailing_efficiency(direction_normalized, wind_normalized)
        theoretical_velocity = direction * efficiency * wind_norm * self.boat_performance

        speed = float(np.linalg.norm(theoretical_velocity))
        if speed > self.max_speed:
            theoretical_velocity = theoretical_velocity / speed * self.max_speed

        new_velocity = theoretical_velocity + self.inertia_factor * (current_velocity - theoretical_velocity)
        speed = float(np.linalg.norm(new_velocity))
        if speed > self.max_speed:
            new_velocity = new_velocity / speed * self.max_speed
        return new_velocity.astype(np.float32)

    def _sailing_efficiency(self, boat_direction: np.ndarray, wind_direction: np.ndarray) -> float:
        boat_norm = float(np.linalg.norm(boat_direction))
        wind_norm = float(np.linalg.norm(wind_direction))
        if boat_norm < 1e-10 or wind_norm < 1e-10:
            return 0.0

        b = boat_direction / boat_norm
        w = wind_direction / wind_norm
        wind_from = -w
        wind_angle = math.acos(float(np.clip(np.dot(wind_from, b), -1.0, 1.0)))

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

    # -------------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------------

    def _clip_xy(self, p: np.ndarray, shape: Tuple[int, int]) -> Tuple[int, int]:
        h, w = shape
        x = int(np.clip(round(float(p[0])), 0, w - 1))
        y = int(np.clip(round(float(p[1])), 0, h - 1))
        return x, y

    def _line_hits_map(self, a: np.ndarray, b: np.ndarray, blocked: np.ndarray) -> bool:
        diff = b - a
        steps = int(max(abs(float(diff[0])), abs(float(diff[1])), 1.0)) + 1
        for t in np.linspace(0.0, 1.0, steps):
            p = a * (1.0 - t) + b * t
            x, y = self._clip_xy(p, blocked.shape)
            if blocked[y, x]:
                return True
        return False
