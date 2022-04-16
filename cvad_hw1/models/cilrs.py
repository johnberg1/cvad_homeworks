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
        self.speed_in = nn.Sequential(nn.Linear(1,128),nn.ReLU(),nn.Dropout(0.25),nn.Linear(128,128))
        
        # speed prediction
        self.speed_prediction = nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.25),nn.Linear(256,1))
        
        # conditional_module, we have 4 branches for 4 different commands
        throttle = [nn.Sequential(nn.Linear(384,256),nn.ReLU(),nn.Dropout(0.25), nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.25),nn.Linear(256,1),nn.Sigmoid()) for i in range(4)]
        self.throttle = nn.ModuleList(throttle)
        
        brake = [nn.Sequential(nn.Linear(384,256),nn.ReLU(),nn.Dropout(0.25),nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.25),nn.Linear(256,1),nn.Sigmoid()) for i in range(4)]
        self.brake = nn.ModuleList(brake)
        
        steer = [nn.Sequential(nn.Linear(384,256),nn.ReLU(),nn.Dropout(0.25),nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.25),nn.Linear(256,1),nn.Tanh()) for i in range(4)]
        self.steer = nn.ModuleList(steer)
        

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
        throttles, brakes, steers = [], [], []
        bs = img.shape[0]
        for i in range(bs):
            cmd = command[i,:]
            pred_throttle = self.throttle[cmd](cond_input[i,:])
            pred_brake = self.brake[cmd](cond_input[i,:])
            pred_steer = self.steer[cmd](cond_input[i,:])
            throttles.append(pred_throttle)
            brakes.append(pred_brake)
            steers.append(pred_steer)
        throttles, brakes, steers = torch.vstack(throttles), torch.vstack(brakes), torch.vstack(steers)
        predicted_actions = torch.cat((steers, throttles, brakes),1)
        return predicted_actions, predicted_speed
    
        
