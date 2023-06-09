import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import deeplake

from constants import IMG_SIZE

ds = deeplake.load('hub://activeloop/wiki-art')
class_names = np.unique(ds.labels.data()['text'])
n_class = len(class_names)

tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(IMG_SIZE),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalizes images to [-1, 1]. more compatible with the tanh in the generator
    # so that the real and fake images have a similar distribution

    # archived:
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # summary statistics calculated from ImageNet, so they are standard practice.
    # you can calculate the new mean and std on your training data, but otherwise using the Imagenet pretrained
    # model with its own mean and std is recommended.

    # ^ potential problem:
    # https://discuss.pytorch.org/t/gan-training-fails-for-different-image-normalization-constants/10574/3
    # besides, ImageNet stats would be better for real life natural images, not paintings probably
])

# used to revert normalization for visualization purposes
# invTrans = transforms.Normalize(
#                 mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#                 std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
invTrans = transforms.Normalize(
                mean=[-1, -1, -1],
                std=[2, 2, 2])
# identical to "image * 0.5 + 0.5"

class PaintingDataset(Dataset):
    '''
    Creates a PyTorch Dataset for the WikiArt painting data which is indexable and pre-applies specified image transformations.
    '''

    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds.images[idx].numpy()
        label = self.ds.labels[idx].numpy(fetch_chunks=True).astype(np.int32)
        # When loading data sequentially, or when randomly loading samples from a tensor that fits
        # into the cache (such as class_labels) it is recommended to set fetch_chunks = True.
        # This increases the data loading speed by avoiding separate requests for each individual sample.
        # This is not recommended when randomly loading large tensors, because the data is deleted from the
        # cache before adjacent samples from a chunk are used.

        if self.transform is not None:
            image = self.transform(image)

        sample = {"images": image, "labels": label}

        return sample

dataset = PaintingDataset(ds, transform = tform)