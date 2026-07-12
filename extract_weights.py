import torch
from stable_baselines3 import PPO

model = PPO.load('rl/models/ppo_2x128_2x64')
state = model.policy.state_dict()

weights = {
    'policy_net.0.weight': state['mlp_extractor.policy_net.0.weight'],
    'policy_net.0.bias': state['mlp_extractor.policy_net.0.bias'],
    'policy_net.2.weight': state['mlp_extractor.policy_net.2.weight'],
    'policy_net.2.bias': state['mlp_extractor.policy_net.2.bias'],
    'policy_net.4.weight': state['mlp_extractor.policy_net.4.weight'],
    'policy_net.4.bias': state['mlp_extractor.policy_net.4.bias'],
    'policy_net.6.weight': state['mlp_extractor.policy_net.6.weight'],
    'policy_net.6.bias': state['mlp_extractor.policy_net.6.bias'],
    'action_net.weight': state['action_net.weight'],
    'action_net.bias': state['action_net.bias'],
}
torch.save(weights, 'rl/models/policy_weights_v2.pt')
print('Saved rl/models/policy_weights_v2.pt')
