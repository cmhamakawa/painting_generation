import torch
from torch.utils.data import DataLoader, Subset

from constants import *
from data.CAN_dataset import *
from networks.CAN_model import *
from CAN_train import train

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", EPOCHS)
    print("Batch size:", BATCH_SIZE)

    # Initialize train dataset
    dataset = PaintingDataset(ds, transform=tform)

    # Initialize, train and evaluate model
    gen = Generator()
    disc = Discriminator()

    gen.apply(weight_init)
    disc.apply(weight_init)

    # create subset for training (temp)
    sub_idx = list(range(0, len(dataset), 12))  # subset contains every 12th painting
    train_subset = Subset(dataset, sub_idx)
    train_subset_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, num_workers=8, shuffle=True, pin_memory=True)
    # num workers notes:

    # Every worker process is always responsible for loading a whole batch
    # num_workers = 0 means that it’s the main process that will do the data loading when needed
    # Having more workers will increase the memory usage and that’s the most serious overhead
    # Setting workers to number of cores is a good rule of thumb, but you could technically give more

    # Begin training
    train(
        train_loader=train_subset_loader,
        gen=gen,
        disc=disc,
        n_eval=N_EVAL,
        device=device,
    )


if __name__ == "__main__":
    main()
