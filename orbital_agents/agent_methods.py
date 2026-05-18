import math
import os
from pathlib import Path

# import kagglehub
import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
from scipy.optimize import brentq

from orbital_agents.v1_first import first_agent

SOLAR_X = 50
SOLAR_Y = 50
SOLAR_RADIUS = 10
ORBIT_RADIUS = 50
MAX_SPEED = 6.0
COMET_MARKER = -99


class TargetModule:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def nearest_target(self, sender: Planet, targets: list[Planet]) -> Planet | None:
        """Prioritize closer targets and those with fewer ships."""
        target = None
        min_dist = float('inf')
        for t in targets:
            dist = math.sqrt((sender.x - t.x) ** 2 + (sender.y - t.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                target = t
        return target

    def simple_weighting_targeting(self, sender: Planet, targets: list[Planet]) -> float:
        """Target planets with higher production and fewer ships, while also considering distance."""
        target = None
        max_weight = float('-inf')
        for t in targets:
            weight = self.simple_weighting(sender, t)
            if weight > max_weight:
                max_weight = weight
                target = t
        return target

    def threat_assessment(self, sender: Planet, all_planets: list[Planet]) -> float:
        """Net threat score for `sender` from all other planets.

        Enemy planets  → positive contribution  (ships / distance)
        Own planets    → negative contribution  (support)
        Neutral planets (owner == -1) → no contribution
        Higher score = more threatened; negative = well supported.
        """
        threat = 0.0
        for planet in all_planets:
            if planet.id == sender.id:
                continue
            dist = max(math.sqrt((sender.x - planet.x) ** 2 + (sender.y - planet.y) ** 2), 1.0)
            if planet.owner == self.player_id:
                threat -= planet.ships / dist**2
            elif planet.owner != -1:
                threat += planet.ships / dist**2
        return threat

    def top_n_targets(self, sender: Planet, targets: list[Planet], n: int) -> list[Planet]:
        """Return the top n targets ranked by simple weighting."""
        return sorted(targets, key=lambda t: self.simple_weighting(sender, t), reverse=True)[:n]

    def simple_weighting(self, sender: Planet, target: Planet) -> float:
        """Calculate a simple weight for a target based on distance, and produciton and number of ships."""
        dist_factor = math.sqrt((sender.x - target.x) ** 2 + (sender.y - target.y) ** 2) * 5
        ship_factor = 1 / (1 + target.ships)  # more ships → lower weight
        production_factor = 1 + target.production / 10  # more production → higher weight
        return (
            ship_factor * production_factor / dist_factor
        )  # closer, higher production and fewer ships → higher weight

    def simple_weighting_w_threats(self, sender: Planet, target: Planet) -> float:
        """Calculate a simple weight for a target based on distance, and produciton and number of ships."""
        dist_factor = math.sqrt((sender.x - target.x) ** 2 + (sender.y - target.y) ** 2) * 5
        ship_factor = 1 / (1 + target.ships)  # more ships → lower weight
        production_factor = 1 + target.production / 10  # more production → higher weight
        return (
            ship_factor * production_factor / dist_factor
        )  # closer, higher production and fewer ships → higher weight


def trajectory_calculation(
    sender: Planet,
    target: Planet,
    v: float,  # fleet speed
    angular_speed: float,
) -> tuple[float, float]:

    solar_x = SOLAR_X
    solar_y = SOLAR_Y

    target_radius = target.radius
    rp = math.sqrt((solar_x - target.x) ** 2 + (solar_y - target.y) ** 2)
    theta_p = math.atan2(target.y - solar_y, target.x - solar_x)

    dx, dy = sender.x - solar_x, sender.y - solar_y
    rs = math.hypot(dx, dy)
    theta_s = math.atan2(dy, dx)

    omega_p = angular_speed if rp + target_radius < ORBIT_RADIUS else 0

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
        # print('No solution found for trajectory calculation')
        return None, None

    t_sol = brentq(f, t_grid[idx[0]], t_grid[idx[0] + 1])

    dx = rp * np.cos(omega_p * t_sol + theta_p) - rs * np.cos(theta_s)
    dy = rp * np.sin(omega_p * t_sol + theta_p) - rs * np.sin(theta_s)
    theta_i = np.arctan2(dy, dx)
    return theta_i, t_sol


def fleet_speed(ships: int, max_speed: float = MAX_SPEED) -> float:
    return 1.0 + (max_speed - 1.0) * (np.log(ships) / np.log(1000)) ** 1.5


def trajectory_crosses_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    """Return True if the line segment (x1,y1)→(x2,y2) passes through the sun disk."""
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
