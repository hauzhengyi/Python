# ENVIRONMENT CLASS

import gym
import torch

class LunarLanderEnvManager():
    
    def __init__(self, device):
        self.device = device
        self.env = gym.make('LunarLander-v2')
        self.current_state = self.env.reset()
        self.done = False
        
    def reset(self):
        self.current_state = self.env.reset()

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n
    
    def take_action(self, action):
        self.current_state, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)
    
    def get_state(self):
        self.current_state = torch.from_numpy(self.current_state)
        return self.current_state.unsqueeze(0).to(self.device)

    def num_states_available(self):
        return self.env.observation_space.shape[0]
    
    def if_done(self):
        return torch.tensor(int(self.done)).unsqueeze(-1).to(self.device)