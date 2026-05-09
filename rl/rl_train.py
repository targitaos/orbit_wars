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

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from orbital_agents.v3_target_weighting_first import agent_target_weighting
from rl.rl_env import OrbitWarsEnv

MODELS_DIR = Path('rl/models')
LOGS_DIR = Path('rl/logs')

# How many total game steps to train for.
# Each orbit_wars episode is 30 steps, so this is roughly 3000 episodes.
# Expect the first ~500 episodes to look completely random — that's normal.
TOTAL_TIMESTEPS = 90_000
# TOTAL_TIMESTEPS = 1_000


def make_env() -> Monitor:
    """
    Create one environment instance wrapped in Monitor.

    Monitor is a thin wrapper that records how much reward was earned each
    episode and how long it lasted. stable-baselines3 reads those logs to
    print progress updates (ep_rew_mean, ep_len_mean).
    """
    env = OrbitWarsEnv(opponent_agent=agent_target_weighting, episode_steps=30)
    return Monitor(env)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

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
        best_model_save_path=str(MODELS_DIR / 'best'),
        log_path=str(LOGS_DIR / 'eval'),
        eval_freq=3_000,  # evaluate every 3000 steps = ~100 episodes
        n_eval_episodes=30,  # average over 30 episodes to reduce variance
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
    #
    # MlpPolicy = a small feed-forward neural net (two hidden layers of 64
    # neurons by default). The net takes the observation vector as input
    # and outputs the 4 action weights.
    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        # n_steps: steps collected before each gradient update.
        # 30 steps/episode * 10 = 300, so each update uses ~10 full episodes.
        n_steps=300,
        # batch_size: PPO splits n_steps into mini-batches. Must divide n_steps.
        batch_size=60,
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
        tensorboard_log=str(LOGS_DIR),
        verbose=1,
    )

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
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        tb_log_name='ppo_v1',
        progress_bar=True,
    )

    final_path = str(MODELS_DIR / 'ppo_v1_final')
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
        n_eval_episodes=50,
        deterministic=True,
    )
    print(f'Mean reward: {mean_reward:.3f} ± {std_reward:.3f}')
    print('(>0 = winning more than losing, <0 = losing more than winning)')


if __name__ == '__main__':
    main()
