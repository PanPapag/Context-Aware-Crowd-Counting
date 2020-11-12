import argparse
import glob
import h5py
import os

import scipy
import scipy.io as io
import scipy.spatial
import numpy as np

from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def make_args_parser():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # fill parser with information about program arguments
    parser.add_argument('-r', '--root', nargs='+', type=str,
                        default='/Users/pantelis/Downloads/archive/ShanghaiTech/part_A',
                        help='define root path to dataset')
    # return an ArgumentParser object
    return parser.parse_args()

def gaussian_filter_density(img, points):
    img_shape = [img.shape[0],img.shape[1]]
    print("\tShape of current image: {}. Totally need generate {} gaussian kernels.".format(img_shape, len(points)))
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density

if __name__ == "__main__":
    # Define root path from command line arguments
    args = make_args_parser()

    part_A_train = os.path.join(args.root, 'train_data', 'images')
    part_A_test = os.path.join(args.root, 'test_data', 'images')

    data_paths = [part_A_train, part_A_test]
    # Generate all image paths
    img_paths = []
    for path in data_paths:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    # Create folders (if not exist) to store density maps of images
    part_A_train_density_map = '/'.join(part_A_train.split('/')[:-1]) + '/density_maps'
    part_A_test_density_map = '/'.join(part_A_test.split('/')[:-1]) + '/density_maps'
    if not os.path.exists(part_A_train_density_map):
        os.mkdir(part_A_train_density_map)
    if not os.path.exists(part_A_test_density_map):
        os.mkdir(part_A_test_density_map)

    for img_path in img_paths:
        print("Generating density map for image: {}".format(img_path.split('/')[-1]))
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
        img = plt.imread(img_path)
        density_map = np.zeros((img.shape[0],img.shape[1]))
        no_people = mat["image_info"][0,0][0,0][0]
        density_map = gaussian_filter_density(img, no_people)
        np.save(img_path.replace('.jpg','.npy').replace('images','density_maps'), density_map)
