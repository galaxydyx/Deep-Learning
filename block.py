import torch
import torch.nn as nn


class UNetConv(nn.Module):
    def __init__(self, int_channel, out_channel):
        super(UNetConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(int_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.ReLU(inplace=True), nn.BatchNorm2d(out_channel),
                                  nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1,  # dilation=2,
                                            bias=False),
                                  nn.ReLU(inplace=True), nn.BatchNorm2d(out_channel))

    def forward(self, x):
        x = self.conv(x)
        return x


class Block(nn.Module):
    def __init__(self, int_channel, out_channel):
        super(Block, self).__init__()
        self.encode = nn.Sequential(nn.Conv2d(int_channel, out_channel, 3, 1, padding=1, bias=False),
                                    nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channel, out_channel, 3, 1, padding=1, bias=False),  # dilation=2,
                                    nn.BatchNorm2d(out_channel))
        self.relu = nn.ReLU(inplace=True)
        if int_channel == out_channel:
            self.down_sample = None
        else:
            self.down_sample = nn.Sequential(nn.Conv2d(int_channel, out_channel, 1, 1, bias=False),
                                             nn.BatchNorm2d(out_channel))

    def forward(self, x):
        identity = x
        if self.down_sample is not None:
            identity = self.down_sample(x)
        x = self.encode(x)
        x += identity
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, int_channel, bott_channel, out_channel):
        super(Bottleneck, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(int_channel, bott_channel, 1, 1, bias=False),
                                  nn.BatchNorm2d(bott_channel), nn.ReLU(inplace=True),
                                  nn.Conv2d(bott_channel, bott_channel, 3, 1, padding=1, bias=False),  # dilation=2，
                                  nn.BatchNorm2d(bott_channel), nn.ReLU(inplace=True),
                                  nn.Conv2d(bott_channel, out_channel, 1, 1, bias=False),
                                  nn.BatchNorm2d(out_channel))
        self.relu = nn.ReLU(inplace=True)
        if int_channel == out_channel:
            self.down_sample = None
        else:
            self.down_sample = nn.Sequential(nn.Conv2d(int_channel, out_channel, 1, 1, bias=False),
                                             nn.BatchNorm2d(out_channel))

    def forward(self, x):
        identity = x
        if self.down_sample is not None:
            identity = self.down_sample(x)
        x = self.conv(x)
        x += identity
        x = self.relu(x)
        return x


def many_blocks(block, int_channel, out_channel, blocks_num):
    layers = []
    for _ in range(blocks_num):
        layers.append(block(int_channel, out_channel))
        int_channel = out_channel
    return nn.Sequential(*layers)


def many_bott_blocks(block, int_channel, bott_channel, out_channel, blocks_num):
    layers = []
    for i in range(blocks_num):
        layers.append(block(int_channel, bott_channel, out_channel))
        int_channel = out_channel
    return nn.Sequential(*layers)


class DenseBlock(nn.Module):
    def __init__(self, channel, blocks_num):
        super(DenseBlock, self).__init__()
        block = []
        transition_layer = []
        for i in range(blocks_num):
            block.append(nn.Sequential(nn.Conv2d(channel, channel, 3, 1, padding=1, bias=False),  # dilation=2,
                                       nn.BatchNorm2d(channel), nn.ReLU(inplace=True)))
            if i < blocks_num-1:
                transition_layer.append(nn.Sequential(nn.Conv2d((i+2)*channel, channel, 1, 1, bias=False),
                                                      nn.BatchNorm2d(channel), nn.ReLU(inplace=True)))
        self.layers_num = blocks_num
        self.blocks = nn.Sequential(*block)
        self.transition_layers = nn.Sequential(*transition_layer)

    def forward(self, x):
        input = []
        x = self.blocks[0](x)
        for i in range(1, self.layers_num):
            input.append(x)
            x = self.blocks[i](x)
            for j in range(i):
                x = torch.cat([input[i-1-j], x], 1)
            x = self.transition_layers[i-1](x)
        return x


class LeakyDenseBlock(nn.Module):
    def __init__(self, channel, blocks_num):
        super(LeakyDenseBlock, self).__init__()
        block = []
        transition_layer = []
        for i in range(blocks_num):
            block.append(nn.Sequential(nn.Conv2d(channel, channel, 3, 1, padding=1, bias=False),  # dilation=2,
                                       nn.BatchNorm2d(channel), nn.LeakyReLU(0.2, inplace=True)))
            if i < blocks_num-1:
                transition_layer.append(nn.Sequential(nn.Conv2d((i+2)*channel, channel, 1, 1, bias=False),
                                                      nn.BatchNorm2d(channel), nn.LeakyReLU(0.2, inplace=True)))
        self.layers_num = blocks_num
        self.blocks = nn.Sequential(*block)
        self.transition_layers = nn.Sequential(*transition_layer)

    def forward(self, x):
        input = []
        x = self.blocks[0](x)
        for i in range(1, self.layers_num):
            input.append(x)
            x = self.blocks[i](x)
            for j in range(i):
                x = torch.cat([input[i-1-j], x], 1)
            x = self.transition_layers[i-1](x)
        return x