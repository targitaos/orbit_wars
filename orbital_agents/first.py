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


def trajectory_calculation(
    mine: Planet,
    target: Planet,
    ships_needed: int,
    angular_speed: float,
    t_max: float = 100.0,
) -> tuple[float, float]:

    solar_x = SOLAR_X
    solar_y = SOLAR_Y

    target_radius = target.radius  # physical radius of orbiting object
    sender_radius = mine.radius  # physical radius of planet object

    rp = math.sqrt((solar_x - target.x) ** 2 + (solar_y - target.y) ** 2)  # orbital radius
    theta_p = math.acos((solar_x - target.x) / rp)  # initial angle of orbiting object

    rs = math.sqrt(
        (mine.x - solar_x + sender_radius * np.cos(theta_p)) ** 2
        + (mine.y - solar_y + sender_radius * np.sin(theta_p)) ** 2,
    )  # starting radius from sun for linear object
    # FIXME:
    theta_s = math.atan2(target.y - solar_y, target.x - solar_x)
    theta_p

    omega = (
        angular_speed if rp + target_radius < ORBIT_RADIUS else 0
    )  # angular speed of orbiting object (rad/s)

    def fleet_speed(ships: int, max_speed: float = MAX_SPEED) -> float:
        return 1.0 + (max_speed - 1.0) * (np.log(ships) / np.log(1000)) ** 1.5

    v = fleet_speed(ships_needed)  # speed of linear object
    # fleet_startpos = np.array([mine.x, mine.y])  # starting position of linear object

    def f(t):
        return (v * t) ** 2 - (rp**2 + rs**2 - 2 * rp * rs * np.cos(omega * t + theta_p - theta_s))

    # Find a bracket where f changes sign
    t_grid = np.linspace(1e-6, t_max, 10000)
    signs = np.sign(f(t_grid))
    idx = np.where(np.diff(signs))[0]
    if len(idx) == 0:
        print('No solution found for trajectory calculation')
        return None, None  # no solution found

    # Take the first (earliest) intercept
    t_sol = brentq(f, t_grid[idx[0]], t_grid[idx[0] + 1])

    # Recover theta_i from the original equations
    dx = rp * np.cos(omega * t_sol + theta_p) - rs * np.cos(theta_s)
    dy = rp * np.sin(omega * t_sol + theta_p) - rs * np.sin(theta_s)
    theta_i = np.arctan2(dy, dx)

    return theta_i, t_sol
    # return alpha_lead, t_star
    # print(f'Collision time : {t_star:.4f} s')
    # print(np.cos(alpha_lead), np.sin(alpha_lead), np.cos(alpha_trail), np.sin(alpha_trail))
    # print(f'Leading  angle : {np.degrees(alpha_lead):.2f}°')
    # print(f'Trailing angle : {np.degrees(alpha_trail):.2f}°')


def first_agent(obs: dict) -> list:
    moves = []
    player = obs.get('player', 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get('planets', []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    # Separate our planets from targets
    my_planets = [p for p in planets if p.owner == player]
    targets_all = [p for p in planets if p.owner != player]
    # targets_opponent = [p for p in planets if p.owner not in {-1, player}]
    # targets_neutral = [p for p in planets if p.owner == -1]

    if not targets_all:
        return moves  # issue; assumes no danger from inflight fleets

    for mine in my_planets:
        # Find the nearest planet we don't own
        nearest = None
        min_dist = float('inf')
        for t in targets_all:
            dist = math.sqrt((mine.x - t.x) ** 2 + (mine.y - t.y) ** 2)
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
        if mine.ships >= ships_needed:
            # Calculate angle from our planet to the target
            # angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            angle, delta_t = trajectory_calculation(
                mine,
                nearest,
                ships_needed,
                angular_speed=obs.angular_velocity,
                # t_max=50.0,
            )
            # if nearest.x**2 + nearest.y**2 < ORBIT_RADIUS**2:
            # print('Target is MOVING')
            if angle is None:
                continue  # no valid trajectory
            moves.append([mine.id, angle, ships_needed])
            print(f'Player {player}: Sending fleet from {mine.id} to {nearest.id}')
            print(f'  Ships: {ships_needed}, against target with {nearest.ships} ships')
            print(f'  Angle: {(angle / np.pi):.2f}π, ETA: {delta_t:.2f} steps')
            print(
                f'  Mine pos: ({mine.x:.2f}, {mine.y:.2f}), Target pos: ({nearest.x:.2f}, {nearest.y:.2f})',
            )
            print('---------------------------------')

    return moves


if __name__ == '__main__':
    # Test it against the random agent
    env = make('orbit_wars', debug=True)
    env.run([first_agent, 'random'])

    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f'Player {i}: reward={s.reward}, status={s.status}')

    html = env.render(mode='html', width=800, height=600)
    with Path.open('replay.html', 'w') as f:
        f.write(html)

    # import matplotlib.pyplot as plt

    # t = np.linspace(0, 50, 500)
    # rp = 75
    # rs = 81
    # omega = 0.25
    # theta_p0 = np.pi / 4
    # theta_s = np.pi / 3
    # v = 4
    # xp = rp * np.cos(omega * t + theta_p0)

    # plt.figure(figsize=(6, 6))
    # plt.plot(t, xp)
    # for theta_i in [0.1, 0.5, 1.0]:
    #     xs = rs * np.cos(theta_s) + v * t * np.cos(theta_i) * t
    #     plt.plot(t, xs, label=f'theta_i={theta_i:.1f}')
    # # xs = rs * np.cos(theta_s) + v * t * np.cos(theta_i) * t
    # plt.xlim(0, 5)
    # plt.ylim(-100, 100)
    # plt.legend()
    # plt.show()
