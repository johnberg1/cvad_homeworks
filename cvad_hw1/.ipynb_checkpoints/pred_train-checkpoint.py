import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor


def validate(model, dataloader, disc_criterion, cont_criterion):
    """Validate model performance on the validation dataset"""
    # Your code here
    model.eval()
    losses = 0.0
    for batch in dataloader:
        image, command, _, _, affordances = batch
        image, command, affordances = image.cuda(), command.cuda(), affordances.cuda()
        aff1_gt, aff2_gt, aff3_gt, aff4_gt = torch.chunk(affordances, 4, dim=1)
        aff2_gt /= 45.0 # To normalize the ground truth tl distance, to inverse this, we can multiply the output of the model with 45.0 when necessary
        with torch.no_grad():
            aff1, aff2, aff3, aff4 = model(image, command)
            aff1_loss = disc_criterion(aff1, aff1_gt)
            aff2_loss = cont_criterion(aff2, aff2_gt)
            aff3_loss = cont_criterion(aff3, aff3_gt)
            aff4_loss = cont_criterion(aff4, aff4_gt)
            loss = aff1_loss + aff2_loss + aff3_loss + aff4_loss
        losses += loss.item() * image.shape[0]
        
    avg_loss = losses / len(dataloader.dataset)
    print('Validation loss: ', avg_loss)
    return avg_loss


def train(model, dataloader, optimizer, disc_criterion, cont_criterion):
    """Train model on the training dataset for one epoch"""
    # Your code here
    model.train()
    aff1_losses, aff2_losses, aff3_losses, aff4_losses, total_losses = [],[],[],[],[]
    print('dataset length', len(dataloader))
    for batch in dataloader:
        optimizer.zero_grad()
        image, command, _, _, affordances = batch
        image, command, affordances = image.cuda(), command.cuda(), affordances.cuda()
        aff1_gt, aff2_gt, aff3_gt, aff4_gt = torch.chunk(affordances, 4, dim=1)
        aff2_gt /= 45.0 # To normalize the ground truth tl distance, to inverse this, we can multiply the output of the model with 45.0 when necessary
        aff1, aff2, aff3, aff4 = model(image, command)
        aff1_loss = disc_criterion(aff1, aff1_gt)
        aff2_loss = cont_criterion(aff2, aff2_gt)
        aff3_loss = cont_criterion(aff3, aff3_gt)
        aff4_loss = cont_criterion(aff4, aff4_gt)
        loss = aff1_loss + aff2_loss + aff3_loss + aff4_loss
        loss.backward()
        optimizer.step()
        aff1_losses.append(aff1_loss.item())
        aff2_losses.append(aff2_loss.item())
        aff3_losses.append(aff3_loss.item())
        aff4_losses.append(aff4_loss.item())
        total_losses.append(loss.item())
        print('Train loss: ', loss.item())
    losses = {"aff1": aff1_losses, "aff2": aff2_losses, "aff3": aff3_losses, "aff4": aff4_losses, "total": total_losses}
    return losses


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    plt.figure()
    total_train_loss = train_loss["total"] 
    
    n_epochs = len(val_loss) - 1
    x_train = np.linspace(0, n_epochs, len(total_train_loss))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, total_train_loss, label='train loss')
    plt.plot(x_test, val_loss, label='test loss')
    plt.legend()
    plt.title('Training Plot')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if not os.path.exists(os.path.dirname('pred_results/')):
        os.makedirs(os.path.dirname('pred_results/'))
    plt.tight_layout()
    plt.savefig('pred_results/training_plot.png')
    
    plt.figure()
    aff1_loss = train_loss["aff1"]
    aff2_loss = train_loss["aff2"] 
    aff3_loss = train_loss["aff3"] 
    aff4_loss = train_loss["aff4"] 
    plt.plot(x_train, aff1_loss, label='tl_state')
    plt.plot(x_train, aff2_loss, label='tl_dist')
    plt.plot(x_train, aff3_loss, label='lane_dist')
    plt.plot(x_train, aff4_loss, label='route_angle')
    plt.legend()
    plt.title('Training Plot for Affordances')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('pred_results/affordance_plot.png')


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = '/userfiles/eozsuer16/expert_data/train'
    val_root = '/userfiles/eozsuer16/expert_data/val'
    print('Initializing the model...')
    model = AffordancePredictor().cuda()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 256
    test_batch_size = 64
    lr = 5e-5 # using the default lr from the paper
    save_path = "pred_model.ckpt"

    print('Preparing the dataloaders...')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    disc_criterion = nn.BCELoss()
    cont_criterion = nn.MSELoss()
    
    aff1_losses, aff2_losses, aff3_losses, aff4_losses, total_losses = [],[],[],[],[]
    train_losses = {"aff1": aff1_losses, "aff2": aff2_losses, "aff3": aff3_losses, "aff4": aff4_losses, "total": total_losses}
    val_losses = []
    print('Starting AffordancePredictor training...')
    for i in range(num_epochs):
        train_loss_dict = train(model, train_loader, optimizer, disc_criterion, cont_criterion)
        for key,val in train_loss_dict.items():
            train_losses[key].extend(val)
        val_losses.append(validate(model, val_loader, disc_criterion, cont_criterion))
        plot_losses(train_losses, val_losses)
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
