import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.cm as CM

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
    parser.add_argument('-c', '--checkpoint', nargs='+', type=str,
                        help='define the model\'s checkpoint')
    parser.add_argument('-i', '--index', nargs='+', type=int,
                        help='define a random image index')
    # return an ArgumentParser object
    return parser.parse_args()

def predict_density_map(test_root, checkpoint_path, device, index):
    model = CANNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    test_loader = torch.utils.data.DataLoader(
        ShanghaiTechPartA(test_root,
            transform=transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
        ), downsample=8),
        batch_size=args.batch_size
    )
    model.eval()
    for i, (img, density_map) in enumerate(test_loader):
        if i == index:
            img = img.to(device)
            density_map = density_map.to(device)
            est_density_map = model(img).detach()
            est_density_map = est_density_map.squeeze(0).squeeze(0).cpu().numpy()
            plt.imshow(est_density_map, cmap=CM.jet)
            break

if __name__ == "__main__":
    args = make_args_parser()
    test_root = os.path.join(args.root, 'test_data', 'images')
    predict_density_map(test_root, args.checkpoint, args.device, args.index)
