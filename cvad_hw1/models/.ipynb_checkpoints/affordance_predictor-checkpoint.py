import torch.nn as nn
from torchvision.models import resnet18
import torch


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self):
        super().__init__()
        
        self.perception_module = resnet18(pretrained=True)
        num_features = self.perception_module.fc.in_features
        self.perception_module.fc = nn.Linear(num_features, 512)
        
        self.aff1 = nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,256), nn.ReLU(),nn.Linear(256,1), nn.Sigmoid())
        
        branches2 = [nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1)) for i in range(4)]
        self.aff2 = nn.ModuleList(branches2)
        
        branches3 = [nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1)) for i in range(4)]
        self.aff3 = nn.ModuleList(branches3)
        
        branches4 = [nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1)) for i in range(4)]
        self.aff4 = nn.ModuleList(branches4)

    def forward(self, img, command):
        # extract perceptual features from resnet
        img_features = self.perception_module(img)
        
        # predict the discrete affordance (traffic stop), unconditionally
        aff1 = self.aff1(img_features)
        
        # predict the rest of the affordances, conditionally
        aff2, aff3, aff4 = [],[],[]
        bs = img.shape[0]
        for i in range(bs):
            cmd = command[i,:]
            aff2_pred = self.aff2[cmd](img_features[i,:])
            aff3_pred = self.aff3[cmd](img_features[i,:])
            aff4_pred = self.aff4[cmd](img_features[i,:])
            aff2.append(aff2_pred)
            aff3.append(aff3_pred)
            aff4.append(aff4_pred)
        aff2, aff3, aff4 = torch.vstack(aff2), torch.vstack(aff3), torch.vstack(aff4)
        return aff1, aff2, aff3, aff4
        
        
