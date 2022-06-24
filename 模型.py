import torch
import torch.nn as nn
from block import UNetConv, Block, Bottleneck, many_blocks, many_bott_blocks, DenseBlock, LeakyDenseBlock


class BlockUNet(nn.Module):
    def __init__(self, block, blocks_num):
        super(BlockUNet, self).__init__()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.encode1 = many_blocks(block, 1, 64, blocks_num)
        self.encode2 = many_blocks(block, 64, 128, blocks_num)
        self.encode3 = many_blocks(block, 128, 256, blocks_num)
        self.encode4 = many_blocks(block, 256, 512, blocks_num)
        self.encode5 = many_blocks(block, 512, 1024, blocks_num)
        self.decode1 = many_blocks(block, 1024, 512, blocks_num)
        self.decode2 = many_blocks(block, 512, 256, blocks_num)
        self.decode3 = many_blocks(block, 256, 128, blocks_num)
        self.decode4 = many_blocks(block, 128, 64, blocks_num)
        self.tr_conv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.tr_conv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.tr_conv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.tr_conv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv = nn.Sequential(nn.Conv2d(64, 1, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.encode1(x)
        x = self.max_pool(x1)
        x2 = self.encode2(x)
        x = self.max_pool(x2)
        x3 = self.encode3(x)
        x = self.max_pool(x3)
        x4 = self.encode4(x)
        x = self.max_pool(x4)
        x = self.encode5(x)
        ###
        x = self.tr_conv1(x)
        x = self.decode1(torch.cat([x4, x], 1))
        x = self.tr_conv2(x)
        x = self.decode2(torch.cat([x3, x], 1))
        x = self.tr_conv3(x)
        x = self.decode3(torch.cat([x2, x], 1))
        x = self.tr_conv4(x)
        x = self.decode4(torch.cat([x1, x], 1))
        ###
        x = self.conv(x)
        return x


class BottleneckUNet(nn.Module):
    def __init__(self, block, blocks_num):
        super(BottleneckUNet, self).__init__()
        self.max_pool = nn.MaxPool2d(2, 2)
        self.encode1 = many_bott_blocks(block, 1, 64, 256, blocks_num)
        self.encode2 = many_bott_blocks(block, 256, 128, 512, blocks_num)
        self.encode3 = many_bott_blocks(block, 512, 256, 1024, blocks_num)
        self.encode4 = many_bott_blocks(block, 1024, 512, 2048, blocks_num)
        self.encode5 = many_bott_blocks(block, 2048, 1024, 4096, blocks_num)
        self.decode1 = many_bott_blocks(block, 4096, 1024, 2048, blocks_num)
        self.decode2 = many_bott_blocks(block, 2048, 512, 1024, blocks_num)
        self.decode3 = many_bott_blocks(block, 1024, 256, 512, blocks_num)
        self.decode4 = many_bott_blocks(block, 512, 64, 128, blocks_num)
        self.tr_conv1 = nn.ConvTranspose2d(4096, 2048, 2, 2)
        self.tr_conv2 = nn.ConvTranspose2d(2048, 1024, 2, 2)
        self.tr_conv3 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.tr_conv4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv = nn.Conv2d(128, 1, 1, 1)

    def forward(self, x):
        x1 = self.encode1(x)
        x = self.max_pool(x1)
        x2 = self.encode2(x)
        x = self.max_pool(x2)
        x3 = self.encode3(x)
        x = self.max_pool(x3)
        x4 = self.encode4(x)
        x = self.max_pool(x4)
        x = self.encode5(x)
        ###
        x = self.tr_conv1(x)
        x = self.decode1(torch.cat([x4, x], 1))
        x = self.tr_conv2(x)
        x = self.decode2(torch.cat([x3, x], 1))
        x = self.tr_conv3(x)
        x = self.decode3(torch.cat([x2, x], 1))
        x = self.tr_conv4(x)
        x = self.decode4(torch.cat([x1, x], 1))
        ###
        x = self.conv(x)
        return x


class DenseBlockUNet(nn.Module):
    def __init__(self, bite, block, blocks_num):
        super(DenseBlockUNet, self).__init__()
        self.avg_pool = nn.AvgPool2d(2, 2)
        en_conv = []
        de_conv = []
        encode = []
        decode = []
        tr_conv = []
        int_channel = 32
        self.conv1 = nn.Sequential(nn.Conv2d(1, int_channel, 1, 1, bias=False),
                                   nn.BatchNorm2d(int_channel), nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(int_channel, bite, 1, 1))
        for i in range(5):
            encode.append(block(int_channel, blocks_num))
            if i < 4:
                decode.append(block(int_channel, blocks_num))
                en_conv.append(nn.Sequential(nn.Conv2d(int_channel, int_channel * 2, 1, 1, bias=False),
                                             nn.BatchNorm2d(int_channel * 2), nn.LeakyReLU(0.2, inplace=True)))
                de_conv.append(nn.Sequential(nn.Conv2d(int_channel*2, int_channel, 1, 1, bias=False),
                                             nn.BatchNorm2d(int_channel), nn.LeakyReLU(0.2, inplace=True)))
                tr_conv.append(nn.ConvTranspose2d(int_channel*2, int_channel, 2, 2))
            int_channel *= 2
        self.en_conv = nn.Sequential(*en_conv)
        self.de_conv = nn.Sequential(*de_conv)
        self.encode = nn.Sequential(*encode)
        self.decode = nn.Sequential(*decode)
        self.tr_conv = nn.Sequential(*tr_conv)

    def forward(self, x):
        input = []
        x = self.conv1(x)
        for i in range(4):
            x = self.encode[i](x)
            input.append(x)
            x = self.avg_pool(x)
            x = self.en_conv[i](x)
        x = self.encode[4](x)
        for i in range(4):
            x = self.tr_conv[-(i+1)](x)
            x = self.de_conv[-(i + 1)](torch.cat([input[-(i+1)], x], 1))
            x = self.decode[-(i+1)](x)
        x = self.conv2(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, block, channel):
        super(DenseNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, channel, 3, 1, padding=1, bias=False),
                                   nn.BatchNorm2d(channel), nn.LeakyReLU(0.2, inplace=True))
        self.max_pool = nn.MaxPool2d(2, 2)
        self.average_pool = nn.AvgPool2d(2, 2)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局池化
        self.Dense_block1 = block(channel, 12)
        self.Dense_block2 = block(channel, 16)
        self.Dense_block3 = block(channel, 24)
        self.Dense_block4 = block(channel, 12)
        self.conv2 = nn.Sequential(nn.Conv2d(channel, 1, 1, 1),
                                   nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.Dense_block1(x)
        x = self.average_pool(x)
        x = self.Dense_block2(x)
        x = self.average_pool(x)
        x = self.Dense_block3(x)
        x = self.average_pool(x)
        x = self.Dense_block4(x)
        x = self.global_average_pool(x)
        x = self.conv2(x)
        return x


class GanDNet(nn.Module):
    def __init__(self):
        super(GanDNet, self).__init__()
        channel = 32
        self.conv1 = nn.Sequential(nn.Conv2d(1, channel, 3, 1, padding=1, bias=False),
                                   nn.BatchNorm2d(channel), nn.LeakyReLU(0.2, inplace=True))
        pool = []
        for i in range(4):
            pool.append(nn.Sequential(nn.Conv2d(channel, 2*channel, 2, 2, bias=False),
                                      nn.BatchNorm2d(2*channel), nn.LeakyReLU(0.2, inplace=True)))
            channel = 2*channel
        self.pool = nn.Sequential(*pool)
        self.conv2 = nn.Sequential(nn.Conv2d(channel, 1, 7, 1),
                                   nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        return x


def u_net():
    return BlockUNet(UNetConv, blocks_num=1)


def block_u_net():
    return BlockUNet(Block, blocks_num=1)


def bottleneck_u_net():
    return BottleneckUNet(Bottleneck, blocks_num=1)


def dense_block_u_net(bite):
    return DenseBlockUNet(bite, LeakyDenseBlock, blocks_num=4)


def dense_net():
    return DenseNet(LeakyDenseBlock, channel=32)

