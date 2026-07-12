from stable_baselines3 import PPO

from rl.rl_env import OrbitWarsEnv

model = PPO.load('rl/models/best_model')
helper = OrbitWarsEnv()


def agent(obs):
    gym_obs = helper._extract_obs(obs)  # kaggle obs → flat numpy vector
    action, _ = model.predict(gym_obs, deterministic=True)  # vector → 4 weights
    return helper._compute_moves(obs, *action)  # weights → fleet moves
