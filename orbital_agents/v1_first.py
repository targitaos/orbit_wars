import math
import os
from pathlib import Path

# import kagglehub
import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
from scipy.optimize import brentq

SOLAR_X = 50
SOLAR_Y = 50
ORBIT_RADIUS = 50
MAX_SPEED = 6.0
COMET_MARKER = -99


def trajectory_calculation(
    sender: Planet,
    target: Planet,
    ships_needed: int,
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

    def fleet_speed(ships: int, max_speed: float = MAX_SPEED) -> float:
        return 1.0 + (max_speed - 1.0) * (np.log(ships) / np.log(1000)) ** 1.5

    v = fleet_speed(ships_needed)

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
        print('No solution found for trajectory calculation')
        return None, None

    t_sol = brentq(f, t_grid[idx[0]], t_grid[idx[0] + 1])

    dx = rp * np.cos(omega_p * t_sol + theta_p) - rs * np.cos(theta_s)
    dy = rp * np.sin(omega_p * t_sol + theta_p) - rs * np.sin(theta_s)
    theta_i = np.arctan2(dy, dx)
    return theta_i, t_sol


def first_agent(obs: dict) -> list:
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
    # targets_opponent = [p for p in planets if p.owner not in {-1, player}]
    # targets_neutral = [p for p in planets if p.owner == -1]
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
        # nearest = planets[12]  # TODO: Remove hardcoded target
        ships_needed = max(nearest.ships + 1, 15)
        # ships_needed = nearest.ships + 1

        # Only send if we have enough
        if sender.ships >= ships_needed:
            # Calculate angle from our planet to the target
            # angle = math.atan2(nearest.y - sender.y, nearest.x - sender.x)
            angle, delta_t = trajectory_calculation(
                sender,
                nearest,
                ships_needed,
                angular_speed=obs.angular_velocity,
                # t_max=50.0,
            )
            # if nearest.x**2 + nearest.y**2 < ORBIT_RADIUS**2:
            # print('Target is MOVING')
            # if obs.step == 36:
            #     angle = 3.087482441904048  # hardcoded for testing
            if angle is None:
                continue  # no valid trajectory
            moves.append([sender.id, angle, ships_needed])
            print(f'Player {player}: Sending fleet from {sender.id} to {nearest.id}')
            print(f'  Ships: {ships_needed}, against target with {nearest.ships} ships')
            print(f'  Angle: {(angle / np.pi):.2f}π, ETA: {delta_t:.2f} steps')
            print(
                f'  Sender pos: ({sender.x:.2f}, {sender.y:.2f}), Target pos: ({nearest.x:.2f}, {nearest.y:.2f})',
            )
            print('---------------------------------')

    return moves


if __name__ == '__main__':
    # Test it against the random agent
    env = make('orbit_wars', debug=True)
    env.run([first_agent, 'random', 'random', 'random'])

    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f'Player {i}: reward={s.reward}, status={s.status}')

    html = env.render(mode='html', width=800, height=600)
    with Path.open('replay.html', 'w') as f:
        f.write(html)
