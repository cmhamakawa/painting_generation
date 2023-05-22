import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from constants import *
from data.CAN_dataset import n_class
from networks.CAN_model import weight_init

import json

def train(train_loader, gen, disc, n_eval, device):
    """
    Trains and evaluates a GAN model.

    Args:
        train_loader:    PyTorch DataLoader containing training data.
        gen:             PyTorch generator model to be trained.
        disc:            PyTorch discriminator model to be trained.
        n_eval:          Interval at which we evaluate our model.
        device:          Device (CPU, GPU, etc.)
    """
    # Move model to GPU
    gen = gen.to(device)
    disc = gen.to(device)

    # Initialize weights
    gen.apply(weight_init);
    disc.apply(weight_init);

    # Loss functions
    loss = nn.BCELoss()
    style_loss = nn.CrossEntropyLoss()

    # Setup AdamW optimizers for both G and D
    optimizer_D = optim.AdamW(disc.parameters(), lr=0.0008, betas=(0.6, 0.999))
    optimizer_G = optim.AdamW(gen.parameters(), lr=0.0008, betas=(0.6, 0.999))

    # Real vs. fake labels training (as floats)
    real_label = 1.
    fake_label = 0.

    # For visualization of training process
    fixed_noise = torch.randn(BATCH_SIZE, NOISE_SIZE, 1, 1, device=device)

    img_list = []
    G_losses = []
    D_losses = []
    entropies = []
    iters = 0

    print("Beginning model training:")

    for epoch in range(EPOCHS):
        data_iter = iter(train_loader)
        for i in range(len(train_loader)):  # num of batches in epoch
            img, style_label = next(data_iter).values()
            # Part A: Train discriminator
            # i) Real painting data
            disc.zero_grad()

            style_label = style_label.to(device)
            img_cpu = img.to(device)
            b_size = img_cpu.shape[0]  # to deal with case of last batch when not equal to BATCH_SIZE
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through discriminator
            output, output_style = disc(img_cpu)
            # Calculate loss on all-real batch
            errD_real = loss(output.squeeze(), label)
            style_label = style_label.to(
                torch.long)  # equivalent to torch.int64 to appease categorical CE loss function input requirements
            errD_real = errD_real + style_loss(output_style.squeeze(), style_label.squeeze())
            # Calculate gradients in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # ii) Generated batch
            noise = torch.randn(b_size, 128, 1, 1, device=device)

            with torch.no_grad():
                fake = gen(noise)
            label.fill_(fake_label)
            # no fake.detach() needed now
            output, output_style = disc(fake)

            # This way generator operations will not build part of the graph so we get better performance
            # (it would in the below case but would be detached afterwards)

            # ---------------
            # another option:
            # fake = gen(noise)
            # label.fill_(fake_label)

            # # Forward pass
            # output, output_style = disc(fake.detach())
            # ---------------

            # Calculate loss on fake batch
            errD_fake = loss(output.squeeze(), label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Compute discriminator total loss
            errD = errD_real + errD_fake
            # Update discriminator
            optimizer_D.step()

            # Part B: Train generator
            gen.zero_grad()

            label.fill_(real_label)  # fake labels are real for generator cost
            # ^ ?
            # After updating discriminator, perform another forward pass of fake batch to compute new loss
            output, output_style = disc(fake.detach())

            # Uniform cross entropy
            logsoftmax = nn.LogSoftmax(dim=1)
            unif = torch.full((BATCH_SIZE, n_class), 1 / n_class)
            unif = unif.to(device)
            # Calculate G's loss based on this output
            errG = loss(output.squeeze(), label)
            errG = errG + torch.mean(-torch.sum(unif * logsoftmax(output_style), 1))
            # unconfirmed explanation of above: so we're summing across each row (an image), and there are 27 columns (styles).
            # This means each image (row) has predicted probabilities for each style (27), which are determined by 1/27 * logsoftmax(output_style)
            # we take the negative of the row sums and find the mean of that and thats our loss, negative because we don't want the discriminator to classify
            # style correctly

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizer_G.step()

            style_entropy = -1 * (
                        nn.functional.softmax(output_style, dim=1) * nn.functional.log_softmax(output_style, dim=1))
            # ^ ? I think adding this log_softmax heavily penalizes a correct prediction (hence the -1), just like explained earlier
            # (which is what we want to force the generator to deviate from style norms)
            # but what is the difference between this style_entropy and the second term we added to err_G?
            style_entropy = style_entropy.sum(dim=1).mean() / torch.log(torch.tensor(n_class).float())

            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\t Entropy: %.4f'
                      % (epoch + 1, EPOCHS, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, style_entropy))

            G_losses.append(errG.item())
            D_losses.append(errD.item())
            entropies.append(style_entropy)

            # To visualize generated images later
            if (iters % 500 == 0) or ((epoch == EPOCHS - 1) and (i == len(train_loader) - 1)):
                with torch.no_grad():
                    fake = gen(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, normalize=True))

            iters += 1

        # did not test yet
        torch.save(gen.state_dict(), f'/content/drive/MyDrive/Pic 16B/CAN/CAN_gen_epoch_{epoch + 1}.pt')
        torch.save(disc.state_dict(), f'/content/drive/MyDrive/Pic 16B/CAN/CAN_disc_epoch_{epoch + 1}.pt')

        with open(f'/content/drive/MyDrive/Pic 16B/CAN/img_list_epoch_{epoch}.json', 'w') as f1:
            json.dump(img_list, f1)
        with open(f'/content/drive/MyDrive/Pic 16B/CAN/G_losses_epoch_{epoch}.json', 'w') as f2:
            json.dump(G_losses, f2)
        with open(f'/content/drive/MyDrive/Pic 16B/CAN/D_losses_epoch_{epoch}.json', 'w') as f3:
            json.dump(D_losses, f3)
        with open(f'/content/drive/MyDrive/Pic 16B/CAN/entropies_epoch_{epoch}.json', 'w') as f4:
            json.dump(entropies, f4)
