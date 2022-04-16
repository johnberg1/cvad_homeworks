from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import torchvision.transforms as transforms
import json

class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root):
        self.data_root = data_root
        # Your code here
        images_root = os.path.join(data_root, 'rgb')
        measurements_root = os.path.join(data_root, 'measurements')
        self.images_path = sorted(read_root(images_root))
        self.measurementes_path = sorted(read_root(measurements_root))
        self.transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.images_path)
        
    def __getitem__(self, index):
        """Return RGB images and measurements"""
        # Your code here
        im_path = self.images_path[index]
        m_path = self.measurementes_path[index]
        image = Image.open(im_path)
        image = self.transforms(image)
        with open(m_path) as json_file:
            measurements = json.load(json_file)
            command = torch.tensor([measurements["command"]])
            speed = torch.tensor([measurements["speed"]])
            actions = torch.tensor([measurements["steer"], measurements["throttle"], measurements["brake"]])
        return image, command, speed, actions

    
def read_root(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images