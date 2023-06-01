#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 10:09:43 2023

@author: christinehamakawa
"""
import numpy as np
import matplotlib.pyplot as plt

import random
import torchvision.transforms as transforms
from constants import *

import streamlit as st
import time

from dcgan_model import *

# dcgan module
import torch
import torch.nn as nn


from PIL import Image

def load_dcgan_models():
    # Number of workers for dataloader
    
    # Batch size during training
    global batch_size
    batch_size = 64
    
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    global image_size
    image_size = 64
    
    # Number of channels in the training images. For color images this is 3
    global nc
    nc = 3
    
    # Size of z latent vector (i.e. size of generator input)
    global nz
    nz = 100
    
    # Size of feature maps in generator
    global ngf
    ngf = 64
    
    # Size of feature maps in discriminator
    global ndf
    ndf = 64
    global ngpu
    ngpu = 1

    

    subset_size = 5000
        
    # generator and discriminator (may not need discriminator)
    model_nameG = "dcgan_models/DCGAN_gen_epoch_5000_44.pt" # TODO: CHANGE LATER
    model_nameD = "dcgan_models/DCGAN_disc_epoch_5000_44.pt" # Idea: report error to show how closely it resembles a painting?
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    
    gen = Generator(ngpu).to(device)
    disc = Discriminator(ngpu).to(device)

    # NOTE currently the lower code adds CPU. this is because im loading rn w/o GPU access. otherwise use version directly below
    # gen.load_state_dict(torch.load('/content/drive/MyDrive/Pic 16B/CAN/CAN_gen_epoch_12.pt')["model_state_dict"]) # currently set to 12
    # disc.load_state_dict(torch.load('/content/drive/MyDrive/Pic 16B/CAN/CAN_disc_epoch_12.pt')["model_state_dict"])
    gen.load_state_dict(torch.load(model_nameG,
                                   map_location=torch.device('cpu'))["model_state_dict"]) # currently set to 12
    disc.load_state_dict(torch.load(model_nameD,
                                    map_location=torch.device('cpu'))["model_state_dict"])
    
#    modelD = torch.load(model_nameD)
#    modelG = torch.load(model_nameG)
    
    return gen, disc
def dcgan_fixed_noise():
    # Size of z latent vector (i.e. size of generator input)
    nz = 100
    ngpu = 1
    # generate random noise for input
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    return fixed_noise
    

def dcgan_generate_images(generator, discriminator):
    
    nz = 100
    ngpu = 1
    # generate random noise for input
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    # can make faster by adding count
    fake_img = generator(fixed_noise)
    output = discriminator(fake_img).view(-1)
    i = 0

    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.transpose(fake_img.detach()[i],(1,2,0)))
    return fig, fake_img[0]

def image_classifier(discriminator, file):
    img = Image.open(file)
    IMG_SIZE = 64
    tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    ])
    x  = tform(img)
    x = x[None, :] # change dimensions for model X
    output = discriminator(x).view(-1)
    probability = round(output[0].item() * 100,3)
    return str(probability)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
