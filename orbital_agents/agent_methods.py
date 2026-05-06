import math
import os
from pathlib import Path

# import kagglehub
import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
from scipy.optimize import brentq

from orbital_agents.first import first_agent

SOLAR_X = 50
SOLAR_Y = 50
SOLAR_RADIUS = 10
ORBIT_RADIUS = 50
MAX_SPEED = 6.0
COMET_MARKER = -99


def trajectory_calculation(
    sender: Planet,
    target: Planet,
    # ships_needed: int,
    v: float,  # fleet speed
    angular_speed: float,
    t_max: float = 100.0,
) -> tuple[float, float]:

    solar_x = SOLAR_X
    solar_y = SOLAR_Y

    target_radius = target.radius  # physical radius of orbiting object
    sender_radius = sender.radius  # physical radius of planet object

    rp = math.sqrt((solar_x - target.x) ** 2 + (solar_y - target.y) ** 2)  # orbital radius
    theta_p = math.atan2(target.y - solar_y, target.x - solar_x)  # initial angle of orbiting object

    # rs = math.sqrt(
    #     (sender.x - solar_x + sender_radius * np.cos(theta_p)) ** 2
    #     + (sender.y - solar_y + sender_radius * np.sin(theta_p)) ** 2,
    # )  # starting radius from sun for linear object
    # theta_s = math.atan2(sender.y - solar_y, sender.x - solar_x)

    dx, dy = sender.x - solar_x, sender.y - solar_y
    rs = math.hypot(dx, dy)
    theta_s = math.atan2(dy, dx)

    omega_p = angular_speed if rp + target_radius < ORBIT_RADIUS else 0

    # fleet_startpos = np.array([sender.x, sender.y])  # starting position of linear object

    def f(t):
        return (v * t) ** 2 - (
            rp**2 + rs**2 - 2 * rp * rs * np.cos((omega_p) * t + theta_p - theta_s)
        )

    # Find a bracket where f changes sign
    t_grid = np.linspace(1e-6, t_max, 10_000)
    signs = np.sign(f(t_grid))
    idx = np.where(np.diff(signs))[0]
    if len(idx) == 0:
        print('No solution found for trajectory calculation')
        return None, None  # no solution found

    # Take the first (earliest) intercept
    t_sol = brentq(f, t_grid[idx[0]], t_grid[idx[0] + 1])

    # Recover theta_i from the original equations
    dx = rp * np.cos(omega_p * t_sol + theta_p) - rs * np.cos(theta_s)
    dy = rp * np.sin(omega_p * t_sol + theta_p) - rs * np.sin(theta_s)
    theta_i = np.arctan2(dy, dx)
    # theta_i = 3.087482441904048 # hardcoded for testing
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
