import torch.nn as nn
from torchvision.models import resnet18
import torch

class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        # resnet18 perception module
        super().__init__()
        self.perception_module = resnet18(pretrained=True)
        num_features = self.perception_module.fc.in_features
        self.perception_module.fc = nn.Linear(num_features, 256)
        
        # speed input
        self.speed_in = nn.Sequential(nn.Linear(1,128),nn.ReLU(),nn.Linear(128,128))
        
        # speed prediction
        self.speed_prediction = nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
        
        # conditional_module, we have 4 branches for 4 different commands
        branches = [nn.Sequential(nn.Linear(384,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,3)) for i in range(4)]
        self.branches = nn.ModuleList(branches)
        

    def forward(self, img, command, speed):
        # extract perceptual features from resnet
        img_features = self.perception_module(img)
        
        # predict the speed
        predicted_speed = self.speed_prediction(img_features)
        
        # pass the speed input through the mlp
        speed_features = self.speed_in(speed)
        
        # concat the perceptual and speed features
        cond_input = torch.hstack((img_features,speed_features))
        # select the branch based on the command, output (steer, throttle, brake)
        predicted_actions = []
        bs = img.shape[0]
        for i in range(bs):
            cmd = command[i,:]
            predicted_action = self.branches[cmd](cond_input[i,:])
            predicted_actions.append(predicted_action)
        predicted_actions = torch.vstack(predicted_actions)
        return predicted_actions, predicted_speed
    
        
