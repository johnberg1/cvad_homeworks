import numpy as np
import torch
import torch.nn as nn
from func_timeout import FunctionTimedOut, func_timeout
from utils.rl_utils import generate_noisy_action_tensor

from .off_policy import BaseOffPolicy
import sys

class TD3(BaseOffPolicy):
    def _compute_q_loss(self, data):
        """Compute q loss for given batch of data."""
        # Your code here
        gamma = 0.99
        criterion = nn.MSELoss()
        state, cmd, action, reward, new_state, cmd2, d = data
        state = torch.vstack(state)
        new_state = torch.vstack(new_state)
        state = torch.transpose(state, 0,1).float()
        new_state = torch.transpose(new_state, 0,1).float()
        state, cmd, action, reward, new_state, cmd2, d = state.cuda(), cmd.cuda(), action.cuda(), reward.cuda(), new_state.cuda(), cmd2.cuda(), d.cuda().unsqueeze(1)



        q1 = self.q_nets[0]
        q2 = self.q_nets[1]
        
        q1_target = self.target_q_nets[0]
        q2_target = self.target_q_nets[1]

        q1_out = q1(state, action)
        q2_out = q2(state, action)

        # with torch.no_grad():
        act = self.policy(state)
        noise = torch.clamp(torch.randn_like(act) * 0.1, -0.5, 0.5) # add the noise to the action, the clamp limits are taken from spinningup
        act = torch.clamp(act + noise, -1.0, 1.0)

        q1_target_out = q1_target(new_state, act)
        q2_target_out = q2_target(new_state, act)
        min_target = torch.minimum(q1_target_out, q2_target_out)
        target = reward + gamma* (1.0 - d.float()) *min_target
        return (q1_out,q2_out), criterion(q1_out, target) + criterion(q2_out, target)

            

    def _compute_p_loss(self, data):
        """Compute policy loss for given batch of data."""
        # Your code here
        state, cmd, action, reward, new_state, cmd2, d = data
        state = torch.vstack(state)
        new_state = torch.vstack(new_state)
        state = torch.transpose(state, 0,1).float()
        new_state = torch.transpose(state, 0,1).float()
        state, cmd, action, reward, new_state, cmd2, d = state.cuda(), cmd.cuda(), action.cuda(), reward.cuda(), new_state.cuda(), cmd2.cuda(), d.cuda()

        q1 = self.q_nets[0]
        act = self.policy(state)
        q = q1(state, act)
        return torch.mean(-q)


    def _extract_features(self, state):
        """Extract whatever features you wish to give as input to policy and q networks."""
        # Your code here
        features = torch.tensor([state["command"],
                                 state["route_dist"],
                                 state["route_angle"],
                                 state["lane_dist"],
                                 state["lane_angle"],
                                 float(state["tl_state"]),
                                 state["tl_dist"],
                                 float(state["is_junction"]),
                                 state["waypoint_dist"],
                                 state["waypoint_angle"]])
        return features

    def _take_step(self, state, action, test=False):
        try:
            if test:            
                action_dict = {
                "throttle": np.clip(action[0].item(), 0, 1),
                "brake": abs(np.clip(action[0].item(), -1, 0)),
                "steer": np.clip(action[1].item(), -1, 1),
                }
                
            else:
                action_dict = {
                "throttle": np.clip(action[0,0].item(), 0, 1),
                "brake": abs(np.clip(action[0,0].item(), -1, 0)),
                "steer": np.clip(action[0,1].item(), -1, 1),
                }
            new_state, reward_dict, is_terminal = func_timeout(
                20, self.env.step, (action_dict,))
        except FunctionTimedOut:
            print("\nEnv.step did not return.")
            raise
        return new_state, reward_dict, is_terminal

    def _collect_data(self, state):
        """Take one step and put data into the replay buffer."""
        features = self._extract_features(state).cuda().float()
        if self.step >= self.config["exploration_steps"]:
            action = self.policy(features) # , [state["command"]]
            action = action.unsqueeze(0)
            action = generate_noisy_action_tensor(
                action, self.config["action_space"], self.config["policy_noise"], 1.0)
        else:
            action = self._explorer.generate_action(state)
        if self.step <= self.config["augment_steps"]:
            action = self._augmenter.augment_action(action, state)

        # Take step
        new_state, reward_dict, is_terminal = self._take_step(state, action)

        new_features = self._extract_features(state)

        # Prepare everything for storage
        stored_features = [f.detach().cpu().squeeze(0) for f in features]
        stored_command = state["command"]
        stored_action = action.detach().cpu().squeeze(0)
        stored_reward = torch.tensor([reward_dict["reward"]], dtype=torch.float)
        stored_new_features = [f.detach().cpu().squeeze(0) for f in new_features]
        stored_new_command = new_state["command"]
        stored_is_terminal = bool(is_terminal)

        self._replay_buffer.append(
            (stored_features, stored_command, stored_action, stored_reward,
             stored_new_features, stored_new_command, stored_is_terminal)
        )
        self._visualizer.visualize(new_state, stored_action, reward_dict)
        return reward_dict, new_state, is_terminal
