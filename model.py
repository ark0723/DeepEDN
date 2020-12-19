import os
import numpy as np

import torch
import torch.nn as nn

from layer import *


class DeepEDN(nn.Module):
    def __init__(self, in_channels, out_channels, nker=32, norm='bnorm', nblk=18):
        super(DeepEDN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nker = nker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CBR2d(self.in_channels, 1 * self.nker, kernel_size=7, stride=2, padding=2, norm=self.norm, relu=0.0)
        self.enc2 = CBR2d(1 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.enc3 = CBR2d(2 * self.nker, 4 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)

        if self.nblk:
            res = []
            for i in range(self.nblk):
                res += [ResBlock(4 * self.nker, 4 * self.nker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0)]

            self.res = nn.Sequential(*res)

        self.dec3 = DECBR2d(4 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.dec2 = DECBR2d(2 * self.nker, 1 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.dec1 = CBR2d(1 * self.nker, self.out_channels, kernel_size=7, stride=2, padding=2, norm=None, relu=None)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.res(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x

class SimpleROI(nn.Module): #input: encripted image -> 5 blocks: see table II on paper #based resnet50 block
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm', nblk=4): #in_channels= 3, out_channels = 2048
        super(SimpleROI, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nker = nker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        res = []
        res +=[ResBlock(self.in_channels, 1 * self.nker, kernel_size=7, stride=2, padding=2, norm=self.norm, relu=0.0)] #Block1
        res +=[ResBlock(1 * self.nker, 4 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)] #Blcok2
        res +=[ResBlock(4 * self.nker, 8 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)] #Block3
        res +=[ResBlock(8 * self.nker, 16 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)] #Block4
        res +=[ResBlock(16 * self.nker, self.out_channels, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)] #Block5

    def forward(self, x):

        x = self.res(x) #regression problem

        #x = torch.tanh(x)

        return x

class ROI(nn.Module): #input: encripted image -> 5 blocks: see table II on paper #based resnet50 block
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm'): #in_channels= 3, out_channels = 2048
        super(ROI, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nker = nker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        res = []
        for i in range(2):
            res +=[ResBlock(self.in_channels, 1 * self.nker, kernel_size=7, stride=2, padding=2, norm=self.norm, relu=0.0)] #Block1

        for i in range(3):
            res +=[ResBlock(1 * self.nker, 4 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)] #Block2

        for i in range(12):
            res +=[ResBlock(4 * self.nker, 8 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)] #Block3

        for i in range(18):
            res +=[ResBlock(8 * self.nker, 16 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)] #Block4

        res +=[ResBlock(16 * self.nker, self.out_channels, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)] #Block5

    def forward(self, x):

        x = self.res(x) #regression problem

        #x = torch.tanh(x)

        return x

class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm', nblk=6):
        super(CycleGAN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nker = nker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CBR2d(self.in_channels, 1 * self.nker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)
        self.enc2 = CBR2d(1 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.enc3 = CBR2d(2 * self.nker, 4 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)

        if self.nblk:
            res = []

            for i in range(self.nblk):
                res += [ResBlock(4 * self.nker, 4 * self.nker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0)]

            self.res = nn.Sequential(*res)

        self.dec3 = DECBR2d(4 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.dec2 = DECBR2d(2 * self.nker, 1 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.dec1 = CBR2d(1 * self.nker, self.out_channels, kernel_size=7, stride=1, padding=3, norm=None, relu=None)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.res(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, nker=32, norm="bnorm"):
        super(Discriminator, self).__init__()

        self.enc1 = CBR2d(1 * in_channels, 1 * nker, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=0.2, bias=False)

        self.enc2 = CBR2d(1 * nker, 2 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc3 = CBR2d(2 * nker, 4 * nker, kernel_size=4, stride=2,
                          padding=1, norm=norm, relu=0.2, bias=False)

        self.enc4 = CBR2d(4 * nker, out_channels, kernel_size=4, stride=2,
                          padding=1, norm=None, relu=None, bias=False)

    def forward(self, x):

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = torch.sigmoid(x)

        return x
