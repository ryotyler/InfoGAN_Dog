import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from module import *

"""
Architecture based on InfoGAN paper.
"""

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.tconv1 = nn.ConvTranspose2d(74, 1024, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)

        self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, 3, 4, 2, padding=1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))

        img = torch.sigmoid(self.tconv4(x))

        return img

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 128, 4, 2, 1)

        self.conv2 = nn.Conv2d(128, 128, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)

        return x

# class Generator(nn.Module):
#     def __init__(self, nz=128, ngf=256, bottom_width=4):
#         super().__init__()

#         self.l1 = nn.Linear(nz, (bottom_width ** 2) * ngf)
#         # self.unfatten = nn.Unflatten(1, (ngf, bottom_width, bottom_width))
#         self.block2 = GBlock(ngf, ngf, upsample=True)
#         self.block3 = GBlock(ngf, ngf, upsample=True)
#         self.block4 = GBlock(ngf, ngf, upsample=True)
#         self.b5 = nn.BatchNorm2d(ngf)
#         self.c5 = nn.Conv2d(ngf, 3, 3, 1, padding=1)
#         self.activation = nn.ReLU(True)

#         nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
#         nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

#     def forward(self, x):
#         h = self.l1(x)
#         # h = self.unfatten(h)
#         h = torch.reshape(h, (128, 256, 4, 4))
#         h = self.block2(h)
#         h = self.block3(h)
#         h = self.block4(h)
#         h = self.b5(h)
#         h = self.activation(h)
#         h = self.c5(h)
#         y = torch.tanh(h)
#         return y

# class Discriminator(nn.Module):
#     def __init__(self, ndf=128):
#         super().__init__()

#         self.block1 = DBlockOptimized(3, ndf)
#         self.block2 = DBlock(ndf, ndf, downsample=True)
#         self.block3 = DBlock(ndf, ndf, downsample=False)
#         self.block4 = DBlock(ndf, ndf, downsample=False)
#         self.l5 = SNLinear(ndf, 1)
#         self.activation = nn.ReLU(True)

#         nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

#     def forward(self, x):
#         h = x
#         h = self.block1(h)
#         h = self.block2(h)
#         h = self.block3(h)
#         h = self.block4(h)
#         h = self.activation(h)
#         h = torch.sum(h, dim=(2, 3))
#         y = self.l5(h)
#         return y

class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1024, 1, 2)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))
        return output

class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        disc_logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return disc_logits, mu, var