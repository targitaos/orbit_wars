"""RL training for orbit_wars using PPO.

The big picture
---------------
RL works by repeating this loop thousands of times:
  1. Look at the game state (observation)
  2. Choose an action (our 4 weights)
  3. The heuristic agent uses those weights to send fleets
  4. The game advances one step
  5. We get a reward (small shaping reward each step, big +1/-1 at the end)
  6. After collecting enough experience, update the neural network to prefer
     actions that led to higher rewards

The neural network (the "policy") is just a function: observation -> action.
It starts random and slowly improves through trial and error.

We never hand-code strategy here. The policy learns on its own which weight
combination wins most often against the opponent.
"""

import argparse
import time

# from html import parser
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from orbital_agents.v3_target_weighting_first import agent_target_weighting
from rl.rl_env import OrbitWarsEnv

# torch.save(model.state_dict(), "/kaggle/working/model.pt")
# How many total game steps to train for.
# Each orbit_wars episode is 500 steps, so this is roughly 3000 episodes.
# Expect the first ~500 episodes to look completely random — that's normal.


def parse_arguments():

    parser = argparse.ArgumentParser(description='Train a PPO agent for Orbit Wars.')

    parser.add_argument(
        '--architecture',
        type=int,
        nargs='+',
        default=None,
        help='Architecture of the neural network as a list of layer sizes, e.g. --architecture 64 64 (default: None, which uses stable-baselines3 defaults of [64, 64])',
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=100_000,
        help='Total number of training timesteps (default: 100_000)',
    )
    parser.add_argument(
        '--episode_steps',
        type=int,
        default=500,
        help='Number of steps per episode (default: 500). This means by default we train for 200 episodes (100_000 / 500).',
    )
    parser.add_argument(
        '--eval_freq',
        type=int,
        default=10_000,
        help='Evaluate every N training steps (default: 10000)',
    )
    parser.add_argument(
        '--n_eval_episodes',
        type=int,
        default=50,
        help='Number of episodes to average over during evaluation (default: 50)',
    )
    parser.add_argument(
        '--n_eval_episodes_final',
        type=int,
        default=200,
        help='Number of episodes to average over during final evaluation (default: 200)',
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=500,
        help='Number of steps to collect before each PPO update (default: 500)',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=50,
        help='Batch size for PPO updates (default: 50)',
    )
    parser.add_argument(
        '--models_dir',
        type=Path,
        default=Path('rl/models'),
        help='Directory to save trained models (default: rl/models)',
    )
    parser.add_argument(
        '--logs_dir',
        type=Path,
        default=Path('rl/logs'),
        help='Directory to save training logs (default: rl/logs)',
    )
    return parser.parse_args()


def make_env() -> Monitor:
    """Create one environment instance wrapped in Monitor.

    Monitor is a thin wrapper that records how much reward was earned each
    episode and how long it lasted. stable-baselines3 reads those logs to
    print progress updates (ep_rew_mean, ep_len_mean).
    """
    args = parse_arguments()
    env = OrbitWarsEnv(opponent_agent=agent_target_weighting, episode_steps=args.episode_steps)
    return Monitor(env)


def main() -> None:
    start_time = time.time()
    args = parse_arguments()

    (args.models_dir).mkdir(parents=True, exist_ok=True)
    (args.logs_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Sanity-check the environment
    # ------------------------------------------------------------------
    # check_env runs a handful of episodes and verifies that the env obeys
    # the gymnasium contract (correct shapes, dtypes, reward is a float, etc.).
    # Comment this out once everything is working — it slows startup.
    print('Checking environment...')
    check_env(OrbitWarsEnv(opponent_agent=agent_target_weighting), warn=True)
    print('Environment OK.\n')

    train_env = make_env()
    eval_env = make_env()

    # ------------------------------------------------------------------
    # Step 2: Set up evaluation callback
    # ------------------------------------------------------------------
    # Every eval_freq training steps, EvalCallback pauses training, plays
    # n_eval_episodes full games, and records the mean reward.
    # If the mean reward is the best seen so far, the model is saved.
    # This is how you track whether training is actually making progress.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.models_dir / 'best'),
        log_path=str(args.logs_dir / 'eval'),
        eval_freq=args.eval_freq,  # evaluate every N steps
        n_eval_episodes=args.n_eval_episodes,  # average over N episodes to reduce variance
        deterministic=True,  # use the greedy action (no random sampling) when evaluating
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Step 3: Create the PPO model
    # ------------------------------------------------------------------
    # PPO (Proximal Policy Optimization) is the standard first choice for
    # continuous action spaces. The "proximal" part means each gradient update
    # is clipped so the policy can't change too drastically in one step —
    # this makes training much more stable than older methods.
    # TODO: Find if n_steps "overshoots" the episode, start with new episode every time.
    # MlpPolicy = a small feed-forward neural net (two hidden layers of 64
    # neurons by default). The net takes the observation vector as input
    # and outputs the 4 action weights.
    import torch

    if args.architecture is not None:
        policy_kwargs = {
            # 'activation_fn': torch.nn.ReLU,
            'net_arch': {'pi': args.architecture, 'vf': args.architecture},
        }
    else:
        policy_kwargs = None

    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        # n_steps: steps collected before each gradient update.
        n_steps=500,
        # batch_size: PPO splits n_steps into mini-batches. Must divide n_steps.
        batch_size=args.batch_size,
        # n_epochs: how many gradient steps to take on each collected batch.
        # More epochs = more learning per episode, but risks overfit to that batch.
        n_epochs=5,
        learning_rate=3e-4,
        # gamma: discount factor. How much future rewards are worth now.
        # gamma=0.99 means a reward 30 steps away is worth 0.99^30 ≈ 0.74 today.
        gamma=0.99,
        # ent_coef: entropy bonus. Rewards the policy for staying uncertain.
        # Without this, the policy collapses to one action too early, before
        # it has explored enough of the action space.
        ent_coef=0.01,
        tensorboard_log=str(args.logs_dir),
        verbose=0,
        policy_kwargs=policy_kwargs,
    )
    print(f'Model device: {model.device}')
    # ------------------------------------------------------------------
    # Step 4: Train
    # ------------------------------------------------------------------
    # To watch training live, open a terminal and run:
    #   tensorboard --logdir rl/logs
    # then open http://localhost:6006 in a browser.
    # Key metrics to watch:
    #   ep_rew_mean        — average reward per episode (should trend upward)
    #   ep_len_mean        — average episode length (30 = game ran to completion)
    #   train/entropy_loss — if this collapses to 0 the policy stopped exploring
    print('Starting training...')
    print('Watch progress: tensorboard --logdir rl/logs\n')

    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        tb_log_name='ppo_v2',
        progress_bar=True,
    )
    final_filename = (
        f'ppo_{"x".join(map(str, args.architecture))}' if args.architecture else 'ppo_default'
    )
    final_path = str(args.models_dir / final_filename)
    model.save(final_path)
    print(f'\nFinal model saved to {final_path}.zip')

    # ------------------------------------------------------------------
    # Step 5: Quick evaluation of the trained model
    # ------------------------------------------------------------------
    # evaluate_policy plays n_eval_episodes full episodes with the greedy
    # policy and returns (mean_reward, std_reward).
    # mean_reward > 0 means the agent wins more than it loses on average
    # (since win=+1, loss=-1, small shaping rewards are ~0.01 per step total).
    print('\nEvaluating final model against agent_target_weighting...')
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.n_eval_episodes_final,
        deterministic=True,
    )
    print(f'Mean reward: {mean_reward:.3f} ± {std_reward:.3f}')
    print('(>0 = winning more than losing, <0 = losing more than winning)')
    end_time = time.time()
    print(f'Total training time: {(end_time - start_time) / 60:.2f} min')


if __name__ == '__main__':
    main()
    # python -m rl.rl_train --timesteps 1_200
    # uv run tensorboard --logdir rl/logs
    # uv run python -m rl.rl_train --timesteps 10_000
