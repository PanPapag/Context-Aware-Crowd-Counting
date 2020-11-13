import cv2
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ShanghaiTechPartA(Dataset):
    def __init__(self, root, shuffle=False, transform=None, downsample=1):
        self.root = root
        self.shuffle = shuffle
        self.transform = transform
        self.downsample = downsample

        self.image_names = [filename for filename in os.listdir(self.root)]
        self.n_samples = len(self.image_names)

        if self.shuffle:
            random.shuffle(self.image_names)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_name = self.image_names[index]
        # Read image and normalize its pixels to [0,1]
        img = plt.imread(os.path.join(self.root,img_name)) / 255
        # Expand grayscale image to three channel.
        if len(img.shape) == 2:
            img = img[:,:,np.newaxis]
            img = np.concatenate((img,img,img),2)

        # Read ground truth density-map
        density_map = np.load(os.path.join(self.root.replace('images','density_maps'),img_name.replace('.jpg','.npy')))

        # Downsample image and density-map to match model's input
        if self.downsample >1:
            rows = int(img.shape[0] // self.downsample)
            cols = int(img.shape[1] // self.downsample)
            img = cv2.resize(img,(cols*self.downsample, rows*self.downsample))
            img = img.transpose((2,0,1)) # convert to order (channel,rows,cols)
            density_map = cv2.resize(density_map, (cols,rows))
            density_map = density_map[np.newaxis,:,:] * self.downsample * self.downsample
            # transform image and density_map to tensors
            img_tensor = torch.tensor(img, dtype=torch.float)
            density_map_tensor = torch.tensor(density_map, dtype=torch.float)
            # Apply any other transformation
            if self.transform is not None:
                img_tensor = self.transform(img_tensor)

        return img_tensor, density_map_tensor


# Test code
if __name__== "__main__":
    root = '/Users/pantelis/Downloads/archive/ShanghaiTech/part_A/train_data/images'
    dataset = ShanghaiTechPartA(root,
                                transform=transforms.Normalize(
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                                downsample=8)
    index = random.randint(0, len(dataset))
    img, dmap = dataset[index]
    print(index, img.shape, dmap.shape)
