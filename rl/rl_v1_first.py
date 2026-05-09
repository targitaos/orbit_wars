import math
import os
from pathlib import Path

# import kagglehub
import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
from scipy.optimize import brentq
from stable_baselines3 import PPO

from orbital_agents.agent_methods import (
    TargetModule,
    fleet_speed,
    trajectory_calculation,
    trajectory_crosses_sun,
)
from orbital_agents.v1_first import first_agent
from orbital_agents.v2_solar_afraid import solar_afraid
from orbital_agents.v3_target_weighting_first import agent_target_weighting
from rl.rl_env import OrbitWarsEnv

SOLAR_X = 50
SOLAR_Y = 50
SOLAR_RADIUS = 10
ORBIT_RADIUS = 50
MAX_SPEED = 6.0
COMET_MARKER = -99

fleet_ledger: dict[tuple[int, int], int] = {}  # {(sender_id, target_id): arrival_step}

# Load model and create a helper env just for its obs/move conversion methods
model = PPO.load('rl/models/best/best_model')  # or 'rl/models/ppo_v1_final'
helper = OrbitWarsEnv()


def rl_agent(obs):
    gym_obs = helper._extract_obs(obs)  # kaggle obs → flat numpy vector
    action, _ = model.predict(gym_obs, deterministic=True)  # vector → 4 weights
    return helper._compute_moves(obs, *action)  # weights → fleet moves


if __name__ == '__main__':
    env = make('orbit_wars', debug=True)
    env.run([rl_agent, agent_target_weighting])

    html = env.render(mode='html', width=800, height=600)
    with open('replay_rl.html', 'w') as f:
        f.write(html)


# def rl_agent(obs: dict) -> list:
#     """Purpose is to ensure fleets are not spamming planets, and thus conserve resources."""
#     step = obs.step
#     if step == 0:
#         fleet_ledger.clear()
#     else:
#         expired = [key for key, cutoff in fleet_ledger.items() if cutoff <= step]
#         for key in expired:
#             del fleet_ledger[key]

#     print(f'--- NEW STEP: {step} ---')
#     moves = []
#     player = obs.get('player', 0) if isinstance(obs, dict) else obs.player
#     raw_planets = obs.get('planets', []) if isinstance(obs, dict) else obs.planets
#     planets = [Planet(*p) for p in raw_planets]

#     # Separate our planets from targets
#     my_planets = [p for p in planets if p.owner == player]
#     targets_comets = [
#         p
#         for p, ip in zip(planets, obs.initial_planets, strict=True)
#         if ip[2] == COMET_MARKER and ip[3] == COMET_MARKER
#     ]
#     comet_set = set(targets_comets)
#     targets_all = [p for p in planets if p.owner != player and p not in comet_set]
#     # targets_opponent = [p for p in planets if p.owner not in {-1, player}]
#     # targets_neutral = [p for p in planets if p.owner == -1]
#     if not targets_all:
#         return moves  # issue; assumes no danger from inflight fleets

#     target_module = TargetModule(player)
#     for sender in my_planets:
#         ranked_targets = target_module.top_n_targets(sender, targets_all, n=3)
#         available_ships = sender.ships

#         for target in ranked_targets:
#             if step < fleet_ledger.get((sender.id, target.id), 0):
#                 print(
#                     f'Player {player}: Fleet from {sender.id} to {target.id} already in flight — skipping'
#                 )
#                 continue

#             ships_needed = target.ships + 1 if target.owner == -1 else max(target.ships + 1, 15)
#             if available_ships < ships_needed:
#                 continue

#             v = fleet_speed(ships_needed)
#             angle, delta_t = trajectory_calculation(
#                 sender,
#                 target,
#                 v=v,
#                 angular_speed=obs.angular_velocity,
#             )
#             if angle is None:
#                 continue
#             ix = sender.x + v * delta_t * math.cos(angle)
#             iy = sender.y + v * delta_t * math.sin(angle)
#             if trajectory_crosses_sun(sender.x, sender.y, ix, iy):
#                 print(
#                     f'Player {player}: Trajectory from {sender.id} crosses the sun — send cancelled'
#                 )
#                 continue

#             fleet_ledger[(sender.id, target.id)] = step + math.ceil(delta_t)
#             available_ships -= ships_needed
#             moves.append([sender.id, angle, ships_needed])
#             print(f'Player {player}: Sending fleet from {sender.id} to {target.id}')
#             print(f'  Ships: {ships_needed}, against target with {target.ships} ships')
#             print(f'  Angle: {(angle / np.pi):.2f}π, ETA: {delta_t:.2f} steps')
#             print(
#                 f'  Sender pos: ({sender.x:.2f}, {sender.y:.2f}), Target pos: ({target.x:.2f}, {target.y:.2f})'
#             )
#             print(f'  Travel distance: {delta_t * v:.2f} ')
#             print('---------------------------------')

#     return moves


# if __name__ == '__main__':
#     env = make('orbit_wars', debug=True)
#     env.run([rl_agent, agent_target_weighting])

#     final = env.steps[-1]
#     for i, s in enumerate(final):
#         print(f'Player {i}: reward={s.reward}, status={s.status}')

#     html = env.render(mode='html', width=800, height=600)
#     with Path.open('replay.html', 'w') as f:
#         f.write(html)
