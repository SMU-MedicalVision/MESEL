#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author: Jiewei Lai

import torch
import torch.nn as nn

k1, p1 = 3, 1
k2, p2 = 5, 2
k3, p3 = 9, 4
k4, p4 = 17, 8

class SELayer1D(nn.Module):

    def __init__(self, nChannels, reduction=16):
        super(SELayer1D, self).__init__()
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.se_block = nn.Sequential(
            nn.Linear(nChannels, nChannels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nChannels // reduction, nChannels, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        alpha = torch.squeeze(self.globalavgpool(x))
        alpha = self.se_block(alpha)
        alpha = torch.unsqueeze(alpha, -1)
        out = torch.mul(x, alpha)
        return out

class BranchConv1D(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(BranchConv1D, self).__init__()
        C = out_channels // 4
        self.b1 = nn.Conv1d(in_channels, C, k1, stride, p1, bias=False)
        self.b2 = nn.Conv1d(in_channels, C, k2, stride, p2, bias=False)
        self.b3 = nn.Conv1d(in_channels, C, k3, stride, p3, bias=False)
        self.b4 = nn.Conv1d(in_channels, C, k4, stride, p4, bias=False)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return out

class BasicBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, drop_out_rate, stride):
        super(BasicBlock1D, self).__init__()
        self.operation = nn.Sequential(
                BranchConv1D(in_channels, out_channels, stride),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(drop_out_rate),
                BranchConv1D(out_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels),
                SELayer1D(out_channels))

        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut.add_module('MaxPool', nn.MaxPool1d(stride, ceil_mode=True))
        if in_channels != out_channels:
            self.shortcut.add_module('ShutConv', nn.Conv1d(in_channels, out_channels, 1))
            self.shortcut.add_module('ShutBN', nn.BatchNorm1d(out_channels))

    def forward(self, x):
        operation = self.operation(x)
        shortcut = self.shortcut(x)
        out = torch.relu(operation + shortcut)
        return out

class Squeeze(nn.Module):

    def forward(self, x):
        out = torch.squeeze(x)
        return out

class HMCNFModule(nn.Module):

    def __init__(self, in_channels, num_classes1=12, num_classes2=24, num_classes3=48):
        super(HMCNFModule, self).__init__()
        self.fc_global1 = nn.Linear(in_channels, in_channels)
        self.fc_global2 = nn.Linear(in_channels, in_channels)
        self.fc_global3 = nn.Linear(in_channels, in_channels)
        self.fc_globals = nn.Linear(in_channels, num_classes1 + num_classes2 + num_classes3)

        self.fc_local1 = nn.Linear(in_channels, num_classes1)
        self.fc_local2 = nn.Linear(in_channels, num_classes2)
        self.fc_local3 = nn.Linear(in_channels, num_classes3)

    def forward(self, x):

        global1 = torch.relu(self.fc_global1(x))
        global2 = torch.relu(self.fc_global2(x + global1))
        global3 = torch.relu(self.fc_global3(x + global2))
        globals = self.fc_globals(x + global3)

        local1 = self.fc_local1(global1)
        local2 = self.fc_local2(global2)
        local3 = self.fc_local3(global3)
        locals = torch.cat([local1, local2, local3], dim=1)

        return globals, locals

class Net(nn.Module):

    def __init__(self, num_classes1=12, num_classes2=24, num_classes3=48, init_channels=12, growth_rate=16,
                 base_channels=64, stride=2, drop_out_rate=0.2):
        super(Net, self).__init__()
        self.num_channels = init_channels
        block_n = 8
        block_c = [base_channels + i * growth_rate for i in range(block_n)]

        self.blocks = nn.Sequential()
        for i, C in enumerate(block_c):
            module = BasicBlock1D(self.num_channels, C, drop_out_rate, stride)
            self.blocks.add_module("block{}".format(i), module)
            self.num_channels = C

        module = nn.AdaptiveAvgPool1d(1)
        self.blocks.add_module("GlobalAvgPool", module)

        self.head = HMCNFModule(self.num_channels, num_classes1, num_classes2, num_classes3)

    def forward(self, x):

        feature = self.blocks(x)
        feature = torch.squeeze(feature)
        globals, locals = self.head(feature)

        return globals, locals

class MENet(nn.Module):

    def __init__(self, num_classes1=12, num_classes2=24, num_classes3=48, init_channels=12, growth_rate=16,
                 base_channels=64, stride=2, drop_out_rate=0.2, experts=12):
        super(MENet, self).__init__()
        self.num_channels = init_channels
        block_n = 8
        block_c = [base_channels + i * growth_rate for i in range(block_n)]
        block_share = block_c[:6]
        block_private = block_c[-2:]

        self.blocks_share = nn.Sequential()
        for i, C in enumerate(block_share):
            module = BasicBlock1D(self.num_channels, C, drop_out_rate, stride)
            self.blocks_share.add_module("block{}".format(i), module)
            self.num_channels = C

        self.blocks_private = nn.ModuleList()
        for expert in range(experts):
            num_channels = self.num_channels
            projection_head = nn.Sequential()
            for i, C in enumerate(block_private):
                module = BasicBlock1D(num_channels, C, drop_out_rate, stride)
                projection_head.add_module("expert{}_block{}".format(expert, i), module)
                num_channels = C
            projection_head.add_module("GlobalAvgPool{}".format(expert), nn.AdaptiveAvgPool1d(1))
            projection_head.add_module("Squeeze{}".format(expert), Squeeze())
            projection_head.add_module("HMCNFModule{}".format(expert), HMCNFModule(num_channels, num_classes1, num_classes2, num_classes3))
            self.blocks_private.append(projection_head)

    def forward(self, x):

        feature = self.blocks_share(x)
        globals, locals = list(), list()
        for Head in self.blocks_private:
            out1 ,out2 = Head(feature)
            globals.append(out1)
            locals.append(out2)
        globals = torch.stack(globals)
        locals = torch.stack(locals)

        return globals, locals

if __name__ == "__main__":

    ecgs = torch.randn([64, 12, 7500])
    net = Net(num_classes1=21, num_classes2=46, num_classes3=187)
    y1, y2 = net(ecgs)
    print(y2.shape)
    paras = sum([p.data.nelement() for p in net.parameters()])
    print('Number of params: {:.2f} M.\n'.format(paras / (1024 ** 2)))