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
model = PPO.load('rl/models/ppo_64x64x64')
# model = PPO.load('rl/models/best/best_model_3x64')  # TODO: change this to your trained model
# opponent_model = PPO.load('rl/models/best/best_model')
# opponent_model = PPO.load('rl/models/ppo_default_10k')
helper = OrbitWarsEnv()


def rl_agent(obs):
    gym_obs = helper._extract_obs(obs)  # kaggle obs → flat numpy vector
    action, _ = model.predict(gym_obs, deterministic=True)  # vector → 4 weights
    return helper._compute_moves(obs, *action)  # weights → fleet moves


def rl_opponent(obs):
    gym_obs = helper._extract_obs(obs)  # kaggle obs → flat numpy vector
    action, _ = opponent_model.predict(gym_obs, deterministic=True)  # vector → 4 weights
    return helper._compute_moves(obs, *action)  # weights → fleet moves


if __name__ == '__main__':
    env = make('orbit_wars', debug=True)
    env.run([rl_agent, agent_target_weighting])
    # env.run([agent_target_weighting, rl_agent])
    # env.run([rl_agent, rl_opponent])

    html = env.render(mode='html', width=800, height=600)
    with open('replay_rl.html', 'w') as f:
        f.write(html)
