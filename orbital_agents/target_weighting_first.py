import math
import os
from pathlib import Path

# import kagglehub
import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
from scipy.optimize import brentq

from orbital_agents.agent_methods import fleet_speed, trajectory_calculation, trajectory_crosses_sun
from orbital_agents.first import first_agent

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
        target = None
        min_dist = float('inf')
        for t in targets:
            dist = math.sqrt((sender.x - t.x) ** 2 + (sender.y - t.y) ** 2)
            if dist < min_dist:
                min_dist = dist
                target = t
        return target

    # for sender in my_planets:
    #     # Find the nearest planet we don't own
    #     target = None
    #     min_dist = float('inf')
    #     for t in targets_all:
    #         dist = math.sqrt((sender.x - t.x) ** 2 + (sender.y - t.y) ** 2)
    #         if dist < min_dist:
    #             min_dist = dist
    #             target = t

    #     if target is None:
    #         continue


def agent_target_weighting(obs: dict) -> list:
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

    target_module = TargetModule(player)
    for sender in my_planets:
        target = target_module.nearest_target(sender, targets_all)
        if target is None:
            continue
        # # Find the nearest planet we don't own
        # target = None
        # min_dist = float('inf')
        # for t in targets_all:
        #     dist = math.sqrt((sender.x - t.x) ** 2 + (sender.y - t.y) ** 2)
        #     if dist < min_dist:
        #         min_dist = dist
        #         target = t

        # if target is None:
        #     continue

        # How many ships do we need? Target's garrison + 1
        ships_needed = max(target.ships + 1, 15)

        v = fleet_speed(ships_needed)  # speed of linear object
        # Only send if we have enough
        if sender.ships >= ships_needed:
            # Calculate angle from our planet to the target
            angle, delta_t = trajectory_calculation(
                sender,
                target,
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
            print(f'Player {player}: Sending fleet from {sender.id} to {target.id}')
            print(f'  Ships: {ships_needed}, against target with {target.ships} ships')
            print(f'  Angle: {(angle / np.pi):.2f}π, ETA: {delta_t:.2f} steps')
            print(
                f'  Sender pos: ({sender.x:.2f}, {sender.y:.2f}), Target pos: ({target.x:.2f}, {target.y:.2f})',
            )
            print(f'  Travel distance: {delta_t * v:.2f} ')
            print('---------------------------------')

    return moves


if __name__ == '__main__':
    env = make('orbit_wars', debug=True)
    env.run([agent_target_weighting, first_agent])

    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f'Player {i}: reward={s.reward}, status={s.status}')

    html = env.render(mode='html', width=800, height=600)
    with Path.open('replay.html', 'w') as f:
        f.write(html)
