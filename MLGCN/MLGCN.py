#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author: Jiewei Lai

import torch
import torch.nn as nn
from torch.nn import Parameter
import math
import pickle
import numpy as np

k1, p1 = 3, 1
k2, p2 = 5, 2
k3, p3 = 9, 4
k4, p4 = 17, 8

def gen_A(num_classes, t, adj_file):

    _adj = adj_file

    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int32)

    return _adj

def gen_adj(A):

    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)

    return adj

class GraphConvolution(nn.Module):

    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

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

class MLGCN(nn.Module):

    def __init__(self, num_channels, num_classes, adj_matrix, t, inp):
        super(MLGCN, self).__init__()

        self.inp = inp
        _adj = gen_A(num_classes, t=t, adj_file=adj_matrix)

        self.A = Parameter(torch.from_numpy(_adj).float())
        self.gc1 = GraphConvolution(num_channels, 512)
        self.gc2 = GraphConvolution(512, num_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):

        adj = gen_adj(self.A).detach()

        fc = self.gc1(self.inp.to(device=x.device), adj)
        fc = self.relu(fc)
        fc = self.gc2(fc, adj)

        fc = fc.transpose(0, 1)
        out = torch.matmul(x, fc)

        return out

class Net(nn.Module):

    def __init__(self, num_classes=3, init_channels=12, growth_rate=16, base_channels=64, stride=2, drop_out_rate=0.2,
                 adj_matrix=np.random.rand(3, 3), t=0.4, inp=torch.randn([3, 176])):
        super(Net, self).__init__()
        self.num_channels = init_channels
        self.num_classes = num_classes

        block_n = 8
        block_c = [base_channels + i * growth_rate for i in range(block_n)]

        self.blocks = nn.Sequential()
        for i, C in enumerate(block_c):
            module = BasicBlock1D(self.num_channels, C, drop_out_rate, stride)
            self.blocks.add_module("block{}".format(i), module)
            self.num_channels = C

        module = nn.AdaptiveAvgPool1d(1)
        self.blocks.add_module("GlobalAvgPool", module)

        self.mlgcn = MLGCN(self.num_channels, num_classes, adj_matrix, t, inp)

    def forward(self, x):

        out = self.blocks(x)
        out = torch.squeeze(out)
        out = self.mlgcn(out)

        return out

class MENet(nn.Module):

    def __init__(self, num_classes=3, init_channels=12, growth_rate=16, base_channels=64, stride=2, drop_out_rate=0.2,
                 adj_matrix=np.random.rand(3, 3), t=0.4, inp=torch.randn([3, 176]), experts=12):
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
            projection_head.add_module("MLGCN{}".format(expert), MLGCN(num_channels, num_classes, adj_matrix, t, inp))
            self.blocks_private.append(projection_head)

    def forward(self, x):

        feature = self.blocks_share(x)
        out = list()
        for Head in self.blocks_private:
            out.append(Head(feature))
        out = torch.stack(out)

        return out

if __name__ == "__main__":

    num_classes = 254
    adj_matrix = np.random.rand(num_classes, num_classes)
    inp = torch.randn([num_classes, 176])

    ecgs = torch.randn([4, 12, 7500])

    net = Net(num_classes=num_classes, adj_matrix=adj_matrix, inp=inp)
    y = net(ecgs)

    print(y.shape)
    paras = sum([p.data.nelement() for p in net.parameters()])
    print('Number of params: {:.2f} M.\n'.format(paras / (1024 ** 2)))