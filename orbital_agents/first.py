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
) -> tuple[float, float]:
    solar_x = 50
    solar_y = 50
    # phi0 = (np.pi - 1) / 2  # initial angle of orbiting object
    r = math.sqrt((solar_x - target.x) ** 2 + (solar_y - target.y) ** 2)  # orbital radius
    phi0 = math.acos((target.x - solar_x) / r)  # initial angle of orbiting object
    target_radius = target.radius  # physical radius of orbiting object

    omega = (
        angular_speed if r + target_radius < ORBIT_RADIUS else 0
    )  # angular speed of orbiting object (rad/s)

    def fleet_speed(ships: int, max_speed: float = MAX_SPEED) -> float:
        return 1.0 + (max_speed - 1.0) * (np.log(ships) / np.log(1000)) ** 1.5

    def fleet_pos(t: float) -> np.ndarray:
        return r * np.array([np.cos(omega * t + phi0), np.sin(omega * t + phi0)])

    def fleet_travel_distance(t: float) -> float:
        return np.linalg.norm(fleet_pos(t) - fleet_startpos)

    def phi(t: float) -> float:
        d = fleet_pos(t) - fleet_startpos
        return np.arctan2(d[1], d[0])

    def cos_arg(t: float) -> float:
        return (v**2 * t**2 + fleet_travel_distance(t) ** 2 - target_radius**2) / (
            2 * v * t * fleet_travel_distance(t)
        )

    v = fleet_speed(ships_needed)  # speed of linear object
    fleet_startpos = np.array([mine.x, mine.y])  # starting position of linear object

    # --- Find first valid collision time ---
    t_vals = np.linspace(0.01, 50, 50_000)
    g = np.array([abs(cos_arg(t)) - 1 for t in t_vals])

    t_star = None
    for i in range(len(g) - 1):
        if g[i] > 0 and g[i + 1] <= 0:
            t_star = brentq(lambda t: abs(cos_arg(t)) - 1, t_vals[i], t_vals[i + 1])
            break

    if t_star is None:
        print('No collision possible with these parameters')
        return None, None
    c = np.clip(cos_arg(t_star), -1, 1)
    # Two solutions: leading (+) and trailing (-) intercept
    alpha_lead = phi(t_star) + np.arccos(c)
    alpha_trail = phi(t_star) - np.arccos(c)

    return alpha_lead, t_star
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
        nearest = planets[12]  # TODO: Remove hardcoded target
        ships_needed = max(nearest.ships + 1, 20)
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
            )
            # if nearest.x**2 + nearest.y**2 < ORBIT_RADIUS**2:
            # print('Target is MOVING')
            if angle is None:
                continue  # no valid trajectory
            moves.append([mine.id, angle, ships_needed])
            # print(moves)
            print(f'Player {player}: Sending fleet from {mine.id} to {nearest.id}')
            # print(
            #     f'  Ships: {ships_needed}, Angle: {math.degrees(angle):.2f}°, ETA: {delta_t:.2f} steps'
            # )
            print(
                f'  Ships: {ships_needed}, Angle: {(angle / np.pi):.2f}π, ETA: {delta_t:.2f} steps'
            )
            print(
                f'  Mine pos: ({mine.x:.2f}, {mine.y:.2f}), Target pos: ({nearest.x:.2f}, {nearest.y:.2f})'
            )

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
