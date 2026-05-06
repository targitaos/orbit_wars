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


def solar_afraid(obs: dict) -> list:
    print(f'--- NEW STEP: {obs.step} ---')
    moves = []
    player = obs.get('player', 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get('planets', []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    # Separate our planets from targets
    my_planets = [p for p in planets if p.owner == player]
    targets_comets = [
        p
        for p, ip in zip(planets, obs.initial_planets, strict=True)
        if ip[2] == COMET_MARKER and ip[3] == COMET_MARKER
    ]
    comet_set = set(targets_comets)
    targets_all = [p for p in planets if p.owner != player and p not in comet_set]
    targets_opponent = [p for p in planets if p.owner not in {-1, player}]
    targets_neutral = [p for p in planets if p.owner == -1]
    if not targets_all:
        return moves  # issue; assumes no danger from inflight fleets

    for sender in my_planets:
        # Find the nearest planet we don't own
        nearest = None
        min_dist = float('inf')
        for t in targets_all:
            dist = math.sqrt((sender.x - t.x) ** 2 + (sender.y - t.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = t

        if nearest is None:
            continue

        # How many ships do we need? Target's garrison + 1
        ships_needed = max(nearest.ships + 1, 15)
        # ships_needed = nearest.ships + 1

        v = fleet_speed(ships_needed)  # speed of linear object
        # Only send if we have enough
        if sender.ships >= ships_needed:
            # Calculate angle from our planet to the target
            angle, delta_t = trajectory_calculation(
                sender,
                nearest,
                # ships_needed,
                v=v,
                angular_speed=obs.angular_velocity,
                # t_max=50.0,
            )
            if angle is None:
                continue  # no valid trajectory
            ix = sender.x + v * delta_t * math.cos(angle)
            iy = sender.y + v * delta_t * math.sin(angle)
            if trajectory_crosses_sun(sender.x, sender.y, ix, iy):
                print(
                    f'Player {player}: Trajectory from {sender.id} crosses the sun — send cancelled'
                )
                continue
            moves.append([sender.id, angle, ships_needed])
            print(f'Player {player}: Sending fleet from {sender.id} to {nearest.id}')
            print(f'  Ships: {ships_needed}, against target with {nearest.ships} ships')
            print(f'  Angle: {(angle / np.pi):.2f}π, ETA: {delta_t:.2f} steps')
            print(
                f'  Sender pos: ({sender.x:.2f}, {sender.y:.2f}), Target pos: ({nearest.x:.2f}, {nearest.y:.2f})',
            )
            print(f'  Travel distance: {delta_t * v:.2f} ')
            print('---------------------------------')

    return moves


if __name__ == '__main__':
    env = make('orbit_wars', debug=True)
    env.run([solar_afraid, first_agent])

    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f'Player {i}: reward={s.reward}, status={s.status}')

    html = env.render(mode='html', width=800, height=600)
    with Path.open('replay.html', 'w') as f:
        f.write(html)
