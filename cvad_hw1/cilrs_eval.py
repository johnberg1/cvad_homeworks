import os
from torchvision import transforms
import yaml
import torch

from carla_env.env import Env


class Evaluator():
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.agent = self.load_agent()

    def load_agent(self,):
        # Your code here
        model = torch.load("cilrs_model.ckpt")
        model.cuda().eval()
        return model

    def generate_action(self, rgb, command, speed):
        # Your code here
        rgb = transforms.ToTensor()(rgb).unsqueeze(0).cuda().float()
        command = torch.tensor(command).unsqueeze(0).unsqueeze(0).cuda().int()
        speed = torch.tensor(speed).unsqueeze(0).unsqueeze(0).cuda().float()
        with torch.no_grad():
            actions, _ = self.agent(rgb, command, speed)
        throttle, brake, steer = actions[0,1], actions[0,2], actions[0,0]
        throttle = torch.clamp(throttle, 0.0, 1.0).cpu().detach().item()
        brake = torch.clamp(brake, 0.0, 1.0).cpu().detach().item()
        steer = torch.clamp(steer, -1.0, 1.0).cpu().detach().item()
        return throttle, steer, brake


    def take_step(self, state):
        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        throttle, steer, brake = self.generate_action(rgb, command, speed)
        action = {
            "throttle": throttle,
            "brake": brake,
            "steer": steer
        }
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):
        terminal_histogram = {}
        for i in range(num_trials):
            state, _, is_terminal = self.env.reset()
            for i in range(5000):
                if is_terminal:
                    break
                state, is_terminal = self.take_step(state)
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0)+1)
        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/100")


def main():
    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        evaluator = Evaluator(env, config)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
