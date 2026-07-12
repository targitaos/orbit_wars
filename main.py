import inspect
import math
from pathlib import Path

import numpy as np
import torch
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
from scipy.optimize import brentq
from torch import nn

SOLAR_X = 50
SOLAR_Y = 50
SOLAR_RADIUS = 10
ORBIT_RADIUS = 50
MAX_SPEED = 6.0
COMET_MARKER = -99

MAX_PLANETS = 30
_PLANET_FEATURES = 7  # [is_mine, is_enemy, ships, production, x, y, dist_from_sun]

_ACTION_LOW = np.array([0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
_ACTION_HIGH = np.array([5.0, 5.0, 5.0, 5.0, 0.9], dtype=np.float32)


def trajectory_calculation(
    sender: Planet,
    target: Planet,
    v: float,
    angular_speed: float,
) -> tuple[float, float]:
    rp = math.sqrt((SOLAR_X - target.x) ** 2 + (SOLAR_Y - target.y) ** 2)
    theta_p = math.atan2(target.y - SOLAR_Y, target.x - SOLAR_X)

    dx, dy = sender.x - SOLAR_X, sender.y - SOLAR_Y
    rs = math.hypot(dx, dy)
    theta_s = math.atan2(dy, dx)

    omega_p = angular_speed if rp + target.radius < ORBIT_RADIUS else 0

    # By the triangle inequality, (v*t)^2 > (rp+rs)^2 guarantees f(t) > 0,
    # so the solution must lie within t < (rp+rs)/v.
    t_max = (rp + rs) / v + 1.0

    def f(t):
        return (v * t) ** 2 - (
            rp**2 + rs**2 - 2 * rp * rs * np.cos(omega_p * t + theta_p - theta_s)
        )

    t_grid = np.linspace(1e-6, t_max, 10_000)
    signs = np.sign(f(t_grid))
    idx = np.where(np.diff(signs))[0]
    if len(idx) == 0:
        return None, None

    t_sol = brentq(f, t_grid[idx[0]], t_grid[idx[0] + 1])

    dx = rp * np.cos(omega_p * t_sol + theta_p) - rs * np.cos(theta_s)
    dy = rp * np.sin(omega_p * t_sol + theta_p) - rs * np.sin(theta_s)
    return np.arctan2(dy, dx), t_sol


def fleet_speed(ships: int, max_speed: float = MAX_SPEED) -> float:
    return 1.0 + (max_speed - 1.0) * (np.log(ships) / np.log(1000)) ** 1.5


def trajectory_crosses_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - SOLAR_X, y1 - SOLAR_Y
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - SOLAR_RADIUS**2
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False
    sqrt_disc = math.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


class AgentHelper:
    def __init__(self, episode_steps: int = 500):
        self.episode_steps = episode_steps
        self._fleet_ledger: dict[tuple[int, int], int] = {}

    def _compute_moves(self, obs, w_dist, w_allegiance, w_ships, w_prod, send_ratio) -> list:
        step = obs.step
        player = obs.player

        if step == 0:
            self._fleet_ledger.clear()
        else:
            stale = [k for k, cutoff in self._fleet_ledger.items() if cutoff <= step]
            for k in stale:
                del self._fleet_ledger[k]

        planets = [Planet(*p) for p in obs.planets]
        my_planets = [p for p in planets if p.owner == player]

        comet_ids = {
            planets[i].id
            for i, ip in enumerate(obs.initial_planets)
            if ip[2] == COMET_MARKER and ip[3] == COMET_MARKER
        }
        targets = [p for p in planets if p.owner != player and p.id not in comet_ids]
        if not targets:
            return []

        def required(t: Planet) -> int:
            return t.ships + 1 if t.owner == -1 else max(t.ships + 1, 15)

        committed: dict[int, int] = {}

        moves = []
        for sender in my_planets:
            available = [t for t in targets if committed.get(t.id, 0) < required(t)]
            if not available:
                available = targets

            best = max(
                available,
                key=lambda t: self._score(sender, t, w_dist, w_allegiance, w_ships, w_prod),
            )

            if step < self._fleet_ledger.get((sender.id, best.id), 0):
                continue

            needed = max(0, required(best) - committed.get(best.id, 0))
            ships = max(needed, int(sender.ships * send_ratio))

            if sender.ships < ships:
                continue

            resolved = self._resolve_launch(sender, best, ships, obs.angular_velocity)
            if resolved is None:
                continue
            angle, delta_t = resolved

            self._fleet_ledger[(sender.id, best.id)] = step + math.ceil(delta_t)
            committed[best.id] = committed.get(best.id, 0) + ships
            moves.append([sender.id, angle, ships])

        return moves

    def _resolve_launch(
        self,
        sender: Planet,
        target: Planet,
        ships: int,
        angular_velocity: float,
    ) -> tuple[float, float] | None:
        v = fleet_speed(ships)
        angle, delta_t = trajectory_calculation(sender, target, v=v, angular_speed=angular_velocity)
        if angle is None:
            return None

        ix = sender.x + v * delta_t * math.cos(angle)
        iy = sender.y + v * delta_t * math.sin(angle)
        if trajectory_crosses_sun(sender.x, sender.y, ix, iy):
            return None

        return angle, delta_t

    def _score(
        self, sender: Planet, target: Planet, w_dist, w_allegiance, w_ships, w_prod
    ) -> float:
        dist = math.hypot(sender.x - target.x, sender.y - target.y)
        dist_factor = 1.0 / (1.0 + dist * w_dist * 0.1)
        # Neutral targets are the baseline; w_allegiance scales how much
        # enemy-owned targets are preferred (>1) or avoided (<1) relative to them.
        allegiance_factor = 1.0 if target.owner == -1 else w_allegiance
        ship_factor = 1.0 / (1.0 + target.ships * w_ships * 0.01)
        prod_factor = 1.0 + target.production * w_prod * 0.1
        return dist_factor * ship_factor * prod_factor * allegiance_factor

    def _extract_obs(self, obs) -> np.ndarray:
        planets = [Planet(*p) for p in obs.planets]
        player = obs.player

        my_ships = sum(p.ships for p in planets if p.owner == player)
        enemy_ships = sum(p.ships for p in planets if p.owner not in {-1, player})
        my_count = sum(1 for p in planets if p.owner == player)
        enemy_count = sum(1 for p in planets if p.owner not in {-1, player})
        neutral_count = sum(1 for p in planets if p.owner == -1)

        global_feat = np.array(
            [
                obs.step / self.episode_steps,
                min(my_ships / 500.0, 1.0),
                min(enemy_ships / 500.0, 1.0),
                min(my_count / MAX_PLANETS, 1.0),
                min(enemy_count / MAX_PLANETS, 1.0),
                min(neutral_count / MAX_PLANETS, 1.0),
            ],
            dtype=np.float32,
        )

        planet_feat = np.zeros(MAX_PLANETS * _PLANET_FEATURES, dtype=np.float32)
        for i, p in enumerate(planets[:MAX_PLANETS]):
            base = i * _PLANET_FEATURES
            planet_feat[base + 0] = 1.0 if p.owner == player else 0.0
            planet_feat[base + 1] = 1.0 if p.owner not in {-1, player} else 0.0
            planet_feat[base + 2] = min(p.ships / 200.0, 1.0)
            planet_feat[base + 3] = min(p.production / 10.0, 1.0)
            planet_feat[base + 4] = p.x / 100.0
            planet_feat[base + 5] = p.y / 100.0
            planet_feat[base + 6] = min(math.hypot(p.x - SOLAR_X, p.y - SOLAR_Y) / 100.0, 1.0)

        return np.concatenate([global_feat, planet_feat])


class ModelModule(nn.Module):
    def __init__(self):
        super().__init__()
        obs_size = 6 + MAX_PLANETS * _PLANET_FEATURES
        self.policy_net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.action_net = nn.Linear(64, 5)

    def forward(self, x):
        return self.action_net(self.policy_net(x))

    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self(x).squeeze(0).numpy()
        return np.clip(action, _ACTION_LOW, _ACTION_HIGH)


helper = AgentHelper()
model = ModelModule()
_weights_path = (
    Path(inspect.getfile(inspect.currentframe())).parent / 'rl' / 'models' / 'policy_weights_v2.pt'
)
model.load_state_dict(torch.load(_weights_path, map_location='cpu'))
model.eval()


def agent(obs):
    gym_obs = helper._extract_obs(obs)
    action = model.predict(gym_obs)
    return helper._compute_moves(obs, *action)
