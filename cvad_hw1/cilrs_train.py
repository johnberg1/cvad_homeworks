import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from expert_dataset import ExpertDataset
from models.cilrs import CILRS

def validate(model, dataloader, criterion):
    """Validate model performance on the validation dataset"""
    # Your code here
    model.eval()
    losses = 0.0
    for batch in dataloader:
        image, command, speed, actions = batch
        image, command, speed, actions = image.cuda(), command.cuda(), speed.cuda(), actions.cuda()
        with torch.no_grad():
            pred_actions, pred_speed = model(image, command, speed)
            actions_loss = criterion(pred_actions, actions)
            speed_loss = criterion(pred_speed, speed)
        loss = actions_loss + speed_loss
        losses += loss.item() * image.shape[0]
        
    avg_loss = losses / len(dataloader.dataset)
    print('Validation loss: ', avg_loss)
    return avg_loss

def train(model, dataloader, optimizer, criterion):
    """Train model on the training dataset for one epoch"""
    # Your code here
    model.train()
    losses = []
    print('dataset length', len(dataloader))
    for batch in dataloader:
        optimizer.zero_grad()
        image, command, speed, actions = batch
        image, command, speed, actions = image.cuda(), command.cuda(), speed.cuda(), actions.cuda()
        pred_actions, pred_speed = model(image, command, speed)
        actions_loss = criterion(pred_actions, actions)
        speed_loss = criterion(pred_speed, speed)
        loss = actions_loss + speed_loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print('Train loss: ', loss.item())
    return losses


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    plt.figure()
    n_epochs = len(val_loss) - 1
    x_train = np.linspace(0, n_epochs, len(train_loss))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, train_loss, label='train loss')
    plt.plot(x_test, val_loss, label='test loss')
    plt.legend()
    plt.title('Training Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if not os.path.exists(os.path.dirname('cilrs_results/')):
        os.makedirs(os.path.dirname('cilrs_results/'))
    plt.tight_layout()
    plt.savefig('cilrs_results/training_plot.png')


def main():
    # Change these paths to the correct paths in your downloaded expert dataset   
    train_root = '/userfiles/eozsuer16/expert_data/train'
    val_root = '/userfiles/eozsuer16/expert_data/val'
    print('Initializing the model...')
    model = CILRS().cuda()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 128
    lr = 2e-4 # using the default lr from the paper
    save_path = "cilrs_model.ckpt"
    
    print('Preparing the dataloaders...')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    train_losses = []
    val_losses = []
    print('Starting CILRS training...')
    for i in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion)
        train_losses.extend(train_loss)
        val_losses.append(validate(model, val_loader, criterion))
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
