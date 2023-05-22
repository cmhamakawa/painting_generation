import torch.nn as nn
from constants import NOISE_SIZE
from data.CAN_dataset import n_class

# Remarks:

# Conv2D is mainly used when you want to detect features, e.g., in the encoder part of an autoencoder model,
# and it may shrink your input shape. Conversely, Conv2DTranspose is used for creating features, for example,
# in the decoder part of an autoencoder model for constructing an image. It makes the input shape larger.

# In GANs, the recommendation is to not use pooling or fully-connected layers

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(NOISE_SIZE, 512, kernel_size=5)
        # 128 just a hyperparameter for noise vector (did not make as a constant as of now)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=3, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=3, padding=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=2)

        self.relu = nn.ReLU()
        # ReLU is recommended for the generator, but not for the discriminator
        self.tanh = nn.Tanh()
        # note ReLU should only be used within hidden layers. for generator output layer we use tanh

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.bn4(self.relu(self.conv4(x)))
        x = self.tanh(self.conv5(x))
        return x

        # We use relu before BN = design choice. BN needs to come last after relu to properly do its job of normalizing layer inputs


class Discriminator(nn.Module):

    # input 64 (batch size) x 3 (channels) x 128 x 128 (image size)

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=3, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=3, padding=2)
        self.bn4 = nn.BatchNorm2d(256)

        self.relu = nn.LeakyReLU(0.1)
        # LeakyRELU is popular in tasks where we may suffer from sparse gradients,
        # for example, training GANs

        self.discriminate = nn.Sequential(
            # instead of linear layer 256 to 1?
            nn.Conv2d(256, 1, kernel_size=4),
            nn.Sigmoid())

        self.classify = nn.Sequential(
            nn.Conv2d(256, n_class, kernel_size=4))
        # in place of linear layers and softmax?
        # nn.Linear(256, 1024),
        # nn.LeakyReLU(0.1),
        # nn.Linear(1024, 512),
        # nn.LeakyReLU(0.1),
        # nn.Linear(512, n_class),
        # nn.Softmax(dim=1)) ???

    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.bn4(self.relu(self.conv4(x)))

        disc_p = self.discriminate(x)
        style_p = self.classify(x)
        return disc_p, style_p

        # note we use relu before BN, design choice. BN needs to come last after relu and pooling to properly
        # do its job of normalizing layer inputs


def weight_init(model_layer):
    '''
    TO-DO
    meant to be used with model.apply(weight_init) such that the function is called for each model layer
    '''
    layer_name = model_layer.__class__.__name__
    if layer_name.find('Conv') != -1:
        # Ads Xavier initialization to convolutional layers
        # design choice to use normal distribution rather than uniform.
        # Recall Xavier initialization initializes weights such that the variance of the
        # activations are the same across every layer; this prevents the gradient from exploding or vanishing
        nn.init.xavier_normal_(model_layer.weight.data)
    elif layer_name.find('BatchNorm') != -1:
        # batchnorm layer only has dim 1 so cannot compute fan in fan out values for Xavier initialization
        nn.init.normal_(model_layer.weight.data, 1, 0.02)
        nn.init.constant_(model_layer.bias.data, 0)