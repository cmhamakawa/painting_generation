import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from constants import *
from data.CAN_dataset import n_class
from networks.CAN_model import weight_init

import json

def train(train_loader, gen, disc, device):
    """
    Trains and evaluates a GAN model.

    Args:
        train_loader:    PyTorch DataLoader containing training data.
        gen:             PyTorch generator model to be trained.
        disc:            PyTorch discriminator model to be trained.
        device:          Device (CPU, GPU, etc.)
    """
    # Move model to GPU
    gen = gen.to(device)
    disc = gen.to(device)

    # Initialize weights
    gen.apply(weight_init);
    disc.apply(weight_init);

    # Loss functions
    loss = nn.BCELoss() # BCELoss does not have label smoothing, will implement manually
    style_loss = nn.CrossEntropyLoss(label_smoothing=0.2)  # to help discriminator be less confident

    # Real vs. fake labels training (as floats)
    real_label = 1.
    fake_label = 0.

    # For visualization of training process. This assumes the model is trained in one session.
    # If training from checkpoints, you must re-load the same fixed noise tensor, of course
    fixed_noise = torch.randn(BATCH_SIZE, NOISE_SIZE, 1, 1, device=device)

    # Setup optimizers for both G and D
    optimizer_G = optim.AdamW(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.AdamW(disc.parameters(), lr=0.00002, betas=(0.5, 0.999))

    # Learning rate exponential decay
    scheduler_D = optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.794)
    # ensures we decay to 1% of LR after 20 epochs (before_lr * gamma^(num steps) = after_lr)

    # side remark: For Adam, every parameter in the network has its own specific learning rate
    # However, this is still useful. Each learning rate is computed using lambda (the initial learning rate) as an upper limit
    # This means that every single learning rate can vary from 0 (no update) to lambda (maximum update)

    # define LR warmup
    num_warmup_epochs = 7

    # archived: steep warmup function——1/1000, 1/100, etc.
    # def warmup(current_step: int):
    #     return 1 / (10 ** (float(num_warmup_epochs - current_step)))

    def warmup(current_step: int):
        if current_step < 3:
            return (current_step + 2) / 4  # current_step starts at 0
        else:
            return 1
            # ^ this setup yields 0.5*LR, 0.75*LR, LR...LR before decay begins

    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=warmup)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer_D, [warmup_scheduler, scheduler_D], [num_warmup_epochs - 1])

    img_list = []
    G_losses = []
    D_losses = []
    entropies = []
    iters = 0

    # for timing
    import time
    durations = []

    print("Beginning model training:")

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}, Discriminator learning rate: {scheduler.optimizer.param_groups[0]['lr']}")
        data_iter = iter(train_loader)
        for i in range(len(train_loader)):  # num of batches in epoch
            start = time.time()

            img, style_label = next(data_iter).values()
            # Part A: Train discriminator
            # i) Real painting data
            disc.zero_grad()

            style_label = style_label.to(device)
            img_cpu = img.to(device)
            b_size = img_cpu.shape[0]  # to deal with case of last batch when not equal to BATCH_SIZE
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # slight label smoothing from hard labels 1 to [0.7, 1.2)
            label = label - 0.3 + torch.rand(b_size, device=device) * 0.5

            # Forward pass real batch through discriminator
            output, output_style = disc(img_cpu)
            # Calculate loss on all-real batch
            errD_real = loss(output.squeeze(), label)
            style_label = style_label.to(torch.long)  # equivalent to torch.int64 to appease categorical CE loss function input requirements
            errD_real = errD_real + style_loss(output_style.squeeze(), style_label.squeeze())
            # Calculate gradients in backward pass
            errD_real.backward(retain_graph=True)
            D_x = output.mean().item()

            # ii) Generated batch
            noise = torch.randn(b_size, 128, 1, 1, device=device)

            fake = gen(noise)
            label.fill_(fake_label)

            # Forward pass
            output, output_style = disc(fake)

            # Calculate loss on fake batch
            errD_fake = loss(output.squeeze(), label)
            # Calculate the gradients for this batch
            errD_fake.backward(retain_graph=True)
            D_G_z1 = output.mean().item()

            # Compute discriminator total loss
            errD = errD_real + errD_fake

            # Clip the discriminator gradient norms (hyperparameter) between .backward and step().
            # current variant: aggressive clipping
            nn.utils.clip_grad_norm_(disc.parameters(), max_norm=10.0)

            # to-do: apparently this clips gradients after backprop, not during
            # https://stackoverflow.com/questions/54716377/how-to-do-gradient-clipping-in-pytorch
            # instead, you can register a backward hook:
            # for p in model.parameters():
            #   p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
            # This hook is called each time after a gradient has been computed, i.e. there's no need for manually clipping once the hook has been registered

            # Update discriminator
            optimizer_D.step()

            # Investigate discriminator gradient norms
            if i % 10 == 0:
                grads = [
                    param.grad.detach().flatten()
                    for param in disc.parameters()
                    if param.grad is not None
                ]
                norm = torch.cat(grads).norm()
                print(f"Discriminant gradient norm: {norm:.4f}")

            # Part B: Train generator
            gen.zero_grad()

            label.fill_(real_label)  # fake labels are real for generator cost

            # After updating discriminator, perform another forward pass of fake batch to compute new loss
            output, output_style = disc(fake)

            # Uniform cross entropy
            logsoftmax = nn.LogSoftmax(dim=1)
            unif = torch.full((b_size, n_class), 1 / n_class).to(device)
            # Calculate G's loss on new D output
            errG = loss(output.squeeze(), label)
            errG = errG + torch.mean(-torch.sum(unif * logsoftmax(output_style), 1))
            # style ambiguity loss——cross-entropy between the class posterior and a uniform distribution
            # breakdown: we sum across each row (an image), and there are 27 columns (styles).
            # This means each image (row) has predicted probabilities for each style (27), which are determined by 1/27 * logsoftmax(output_style)
            # in accordance with the formula, we find the mean of the row sums (+ negative sign)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()

            # Clip the generator gradient norms (hyperparameter) between .backward and step()
            # current variant: aggressive clipping
            nn.utils.clip_grad_norm_(gen.parameters(), max_norm=10.0)

            # Update G
            optimizer_G.step()

            # Investigate generator gradient norms
            if i % 10 == 0:
                grads = [
                    param.grad.detach().flatten()
                    for param in gen.parameters()
                    if param.grad is not None
                ]
                norm = torch.cat(grads).norm()
                print(f"Generator gradient norm: {norm:.4f}")

            # Not used to update the model, simply for training output
            # Written with an additional softmax to restrict the bounds to 0 to 1 for interpretability
            style_entropy = -1 * (nn.functional.softmax(output_style, dim=1) *
                                  nn.functional.log_softmax(output_style, dim=1))
            # ^ Log_softmax heavily penalizes mistakes in likelihood space (more so than vanilla softmax)
            style_entropy = style_entropy.sum(dim=1).mean() / torch.log(torch.tensor(n_class).float())

            stop = time.time()
            durations.append(stop - start)

            if i % 10 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\t Entropy: %.4f'
                      % (epoch + 1, EPOCHS, i, len(train_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, style_entropy))

            G_losses.append(errG.item())
            D_losses.append(errD.item())
            entropies.append(style_entropy.item())

            # Append generated images using model after each 75 batches or at end of training to visualize training process
            if (iters % 75 == 0) or ((epoch == EPOCHS - 1) and (i == len(train_loader) - 1)):
                with torch.no_grad():
                    fake = gen(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, normalize=True))

            iters += 1

        # learning rate scheduler for discriminator. step every epoch
        scheduler.step()

        print(
            f"Average execution time per batch of {BATCH_SIZE} in epoch {epoch + 1}: {np.array(durations).mean()} seconds")
        # remark: of course, the last batch of every epoch only has size b_size but given 100+ batches it should not matter too much

        # save models and progress every epoch
        # commented out because actual training and model checkpointing was done on Colab

        # torch.save({
        #     "model_state_dict": gen.state_dict(),
        #     "optimizer_state_dict": optimizer_G.state_dict()},
        #     f'/content/drive/MyDrive/Pic 16B/CAN/CAN_gen_epoch_{epoch + 1}.pt')
        #
        # torch.save({
        #     "model_state_dict": disc.state_dict(),
        #     "optimizer_state_dict": optimizer_D.state_dict(),
        #     "scheduler_state_dict": scheduler_D.state_dict()},  # scheduler
        #     f'/content/drive/MyDrive/Pic 16B/CAN/CAN_disc_epoch_{epoch + 1}.pt')
        #
        # torch.save(img_list, f'/content/drive/MyDrive/Pic 16B/CAN/img_list.pt')
        #
        # with open(f'/content/drive/MyDrive/Pic 16B/CAN/G_losses.json', 'w') as f2:
        #     json.dump(G_losses, f2)
        # with open(f'/content/drive/MyDrive/Pic 16B/CAN/D_losses.json', 'w') as f3:
        #     json.dump(D_losses, f3)
        # with open(f'/content/drive/MyDrive/Pic 16B/CAN/entropies.json', 'w') as f4:
        #     json.dump(entropies, f4)

        print(f"Epoch {epoch + 1} finished, model + data saved")
