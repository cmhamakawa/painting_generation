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
    # standardize data using ImageNet-calculated mean and std for each of the 3 channels
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# used to revert normalization for visualization purposes
invTrans = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

class PaintingDataset(Dataset):
    '''
    TO-DO
    '''

    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds.images[idx].numpy()
        label = self.ds.labels[idx].numpy(fetch_chunks=True).astype(np.int32)

        if self.transform is not None:
            image = self.transform(image)

        sample = {"images": image, "labels": label}

        return sample

dataset = PaintingDataset(ds, transform = tform)