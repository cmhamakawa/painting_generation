import torch.nn as nn
from constants import NOISE_SIZE
from data.CAN_dataset import n_class

from torch.nn.utils.parametrizations import spectral_norm

# Remarks:

# Conv2D is mainly used when you want to detect features, e.g., in the encoder part of an autoencoder model,
# and it may shrink your input shape. Conversely, Conv2DTranspose is used for creating features, for example,
# in the decoder part of an autoencoder model for constructing an image. It makes the input shape larger.

# In GANs, the recommendation is to not use pooling or fully-connected layers

class Generator(nn.Module):

    # input: 64 (batch size) x 128 (noise vector dim) x 1 x 1

    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(NOISE_SIZE, 1024, kernel_size=6, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1024)
        self.conv3 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(512)
        self.conv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, bias=False)

        # generally recommended to set bias = False because BatchNorm layer will re-centre the data anyway,
        # rendering the bias a useless trainable parameter.
        # However, since we have a non-linear activation like ReLU in, this statement may not apply

        self.relu = nn.ReLU()

        self.tanh = nn.Tanh()
        # ReLU should only be used within hidden layers. for generator output layer we use tanh

    # output: [64 (batch_size), 3 (# of channels), 128 (image_size), 128 (image_size)] (i.e. a properly sized image)

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.bn4(self.relu(self.conv4(x)))
        x = self.bn5(self.relu(self.conv5(x)))
        x = self.bn6(self.relu(self.conv6(x)))
        x = self.tanh(self.conv7(x))
        return x

        # We use relu before BN = design choice. BN needs to come last after relu to properly do its job of normalizing layer inputs


class Discriminator(nn.Module):

    # input: 64 (batch size) x 3 (channels) x 128 x 128 (image size)

    def __init__(self):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=2, bias=False))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = spectral_norm(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = spectral_norm(nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False))
        self.bn6 = nn.BatchNorm2d(512)
        # Spectral normalization is a GAN stability technique which prevents vanishing / exploding gradients by controlling the Lipschitz constant of the
        # discriminator (specifically. normalizing the spectral norm of the weight matrix)
        # This does not affect batch normalization AFAIK. SN standardizes the weights of the layer, while BN standardizes activations.

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        # LeakyRELU is popular in tasks where we may suffer from sparse gradients,
        # for example, training GANs

        self.discriminate = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(512*4*4, 256),
            # nn.Linear(256, 1),
            # nn.Sigmoid())
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid())

        self.classify = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(512*4*4, 1024),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(1024, 512),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(512, n_class))
            # # no softmax, will apply later
            nn.Conv2d(512, n_class, kernel_size=4, stride=1, padding=0, bias=False))

    def forward(self, x):
        # x = self.bn1(self.relu(self.conv1(gaussian_noise(x))))
        # x = self.bn2(self.relu(self.conv2(gaussian_noise(x))))
        # x = self.bn3(self.relu(self.conv3(gaussian_noise(x))))
        # x = self.bn4(self.relu(self.conv4(gaussian_noise(x))))
        # x = self.bn5(self.relu(self.conv5(gaussian_noise(x))))
        # x = self.bn6(self.relu(self.conv6(gaussian_noise(x))))
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.bn4(self.relu(self.conv4(x)))
        x = self.bn5(self.relu(self.conv5(x)))
        x = self.bn6(self.relu(self.conv6(x)))

        disc_p = self.discriminate(x)
        style_p = self.classify(x)
        return disc_p, style_p

        # note we use relu before BN, design choice. BN needs to come last after relu and pooling to properly do its job of normalizing layer inputs

def weight_init(model_layer):
    '''
    Custom weight initialization function to be applied to each model layer of a given model.
    '''
    layer_name = model_layer.__class__.__name__
    if layer_name.find('Conv') != -1:
        # originally used Xavier normal initialization to convolutional layers (varaince of activations same across every layer to prevent
        # gradient from exploding or vanishing)
        # nn.init.xavier_normal_(model_layer.weight.data)
        nn.init.normal_(model_layer.weight.data, 0, 0.02)
    elif layer_name.find('BatchNorm') != -1:
        # cannot use Xavier for BatchNorm layers because it's 1D not 2D, so cannot compute fan in/fan out values
        nn.init.normal_(model_layer.weight.data, 1, 0.02)
        nn.init.constant_(model_layer.bias.data, 0)