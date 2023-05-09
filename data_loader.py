import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

import deeplake

from constants import *

ds = deeplake.load('hub://activeloop/wiki-art')
class_names = np.unique(ds.labels.data()['text'])

tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # transforms.CenterCrop(IMG_SIZE*0.5),
    # transforms.RandomCrop(IMG_SIZE*0.75, padding=2),
    # transforms.RandomHorizontalFlip(),

    # ^ uncommenting these sometimes leads to visualization errors. and of course would make the images look
    # distorted and unnatural

    # standardize data using ImageNet-calculated mean and std for each of the 3 channels
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# can be used to revert normalization for visualization purposes
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
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, num_workers = 0, shuffle = True, pin_memory=True)
