import argparse
import os
import random
import time

import torch
import torch.nn as nn
from torchvision import transforms

from model import CANNet
from dataset import ShanghaiTechPartA

def make_args_parser():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # fill parser with information about program arguments
    parser.add_argument('-r', '--root', nargs='+', type=str,
                        default='/Users/pantelis/Downloads/archive/ShanghaiTech/part_A',
                        help='define the root path to dataset')
    parser.add_argument('-d', '--device', nargs='+', type=str,
                        choices=['cuda', 'cpu'],
                        default='cpu',
                        help='define the device to train/test the model')
    parser.add_argument('-l', '--learning_rate', nargs='+', type=float,
                        default=1e-7,
                        help='define the learning rate of the model')
    parser.add_argument('-b', '--batch_size', nargs='+', type=int,
                        default=1,
                        help='define the batch size')
    parser.add_argument('-m', '--momentum', nargs='+', type=float,
                        default=0.95,
                        help='define the momentum hyperparameter')
    parser.add_argument('-e', '--epochs', nargs='+', type=float,
                        default=20000,
                        help='define the number of epochs')
    # return an ArgumentParser object
    return parser.parse_args()

def print_args(args):
    print("Running with the following configuration")
    # get the __dict__ attribute of args using vars() function
    args_map = vars(args)
    for key in args_map:
        print('\t', key, '-->', args_map[key])
    # add one more empty line for better output
    print()


if __name__== "__main__":
    # Parse arguments
    args = make_args_parser()
    print_args(args)
    # Initialize model, loss function and optimizer
    seed = time.time()
    device = torch.device(args.device)
    torch.cuda.manual_seed(seed)
    model = CANNet().to(args.device)
    criterion = nn.MSELoss(size_average=False).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(),args.learning_rate,
                                momentum=args.momentum, weight_decay=0)
    print("Model loaded")
    print(model)

    # Load train dataset
    train_root = os.path.join(args.root, 'train_data', 'images')
    train_loader = torch.utils.data.DataLoader(
        ShanghaiTechPartA(train_root,
            shuffle=True,
            transform=transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
        ), downsample=8),
        batch_size=args.batch_size
    )

    # Create checkpoints folder
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    # Train and save the Model
    epochs = []
    train_erors = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for i, (img, density_map) in enumerate(train_loader):
            img = img.to(args.device)
            density_map = density_map.to(args.device)
            # forward propagation
            est_density_map = model(img)
            # calculate loss
            loss = criterion(est_density_map, density_map)
            epoch_loss += loss.item()
            # back propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch: [{}][{}/{}]\t'
                  'Loss: {:.4f}\t'
                  .format(epoch, i, len(train_loader), loss.item()))
        print('Epoch [{}] completed.\t Avg Loss: {}'.format(epoch, epoch_loss / len(train_loader)))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss / len(train_loader))
        torch.save(model.state_dict(),'./checkpoints/epoch_'+str(epoch)+".pth")
