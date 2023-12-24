#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author: Jiewei Lai

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eps = 1e-5

# BCELoss for single group output.
class BCELoss(nn.Module):

    # MS is mutualistic symbiosis, ME is mutual exclusion.
    # Optional activate functions: Sigmoid, Softmax, ConcurrentSoftmax, LocalSoftmax.
    # k1=0, k2=0 denotes the long-tailed loss.
    # k1=1, k2=0 denotes the balanced loss.
    # k1=1, k2=1 denotes the inversed loss.
    # symbiotic is the weight of the symbiotic ranking regularizer.
    # suppress is the hyperparameter τ.
    def __init__(self, cls_num_list, N, acti='Sigmoid', prior=None, k1=0, k2=0, symbiotic=0, suppress=9):
        super(BCELoss, self).__init__()

        self.C = len(cls_num_list)
        self.weight = N / cls_num_list
        self.acti = acti

        self.prior = prior
        self.prior_me = 1 - prior
        self.prior_me[self.prior_me != 1] = 0
        self.prior_ms = prior.clone()
        self.prior_ms[self.prior_ms != 1] = 0
        self.prior_ms -= torch.eye(self.C)

        self.v1_sigmoid = k1 * torch.log(N / cls_num_list - 1 + eps)
        self.v2_sigmoid = k2 * self.inverse(torch.log(N / cls_num_list - 1 + eps))
        self.v1_softmax = k1 * torch.log(cls_num_list / N + eps)
        self.v2_softmax = k2 * self.inverse(torch.log(cls_num_list / N + eps))

        self.symbiotic = symbiotic
        self.suppress = suppress

    def inverse(self, v):
        value, idx0 = torch.sort(v)
        _, idx1 = torch.sort(idx0)
        idx2 = v.shape[0] - 1 - idx1  # reverse the order
        result = value.index_select(0, idx2)
        return result

    def SymbioticRegularizer(self, logits):
        loss = 0
        for c in range(self.C):
            p = self.prior_ms[c]
            if p.sum():
                parent = logits[:, c]
                leafs = logits[:, p == 1]
                loss_exp = torch.exp(leafs - parent.unsqueeze(1))
                loss_log = torch.log(1 + loss_exp)
                loss += loss_log.mean()
        return loss / torch.count_nonzero(self.prior_ms.sum(1))

    def Sigmoid(self, logits):
        logits_acti = torch.sigmoid(logits - self.v1_sigmoid + self.v2_sigmoid)
        return logits_acti

    def Softmax(self, logits):
        logits_exp = torch.exp(logits + self.v1_softmax - self.v2_softmax)
        denominator = torch.sum(logits_exp, dim=1, keepdim=True)
        logits_acti = logits_exp / (denominator + eps)
        return logits_acti

    def ConcurrentSoftmax(self, logits):
        logits_exp = torch.exp(logits + self.v1_softmax - self.v2_softmax)
        denominator = torch.matmul(logits_exp, torch.pow(1 - self.prior, self.suppress)) + logits_exp
        logits_acti = logits_exp / (denominator + eps)
        return logits_acti

    def LocalSoftmax(self, logits):
        logits_exp = torch.exp(logits + self.v1_softmax - self.v2_softmax)
        denominator = torch.matmul(logits_exp, self.prior_me) + logits_exp
        logits_acti = logits_exp / (denominator + eps)
        return logits_acti

    def activate(self, logits):
        if self.acti is 'Sigmoid':
            logits_acti = self.Sigmoid(logits)
        if self.acti is 'Softmax':
            logits_acti = self.Softmax(logits)
        if self.acti is 'ConcurrentSoftmax':
            logits_acti = self.ConcurrentSoftmax(logits)
        if self.acti is 'LocalSoftmax':
            logits_acti = self.LocalSoftmax(logits)
        return logits_acti

    # logits: [N, *], target: [N, *]
    def forward(self, logits, target):

        logits_acti = torch.clamp(self.activate(logits), eps, 1 - eps)
        self.loss = - self.weight * (target * torch.log(logits_acti) + (1 - target) * torch.log(1 - logits_acti))
        self.regularization = self.symbiotic * self.SymbioticRegularizer(logits_acti) if self.symbiotic else 0

        return self.loss.mean() + self.regularization

    def cuda(self):
        self.weight = self.weight.cuda()
        self.prior = self.prior.cuda()
        self.prior_me = self.prior_me.cuda()
        self.prior_ms = self.prior_ms.cuda()
        self.v1_sigmoid = self.v1_sigmoid.cuda()
        self.v2_sigmoid = self.v2_sigmoid.cuda()
        self.v1_softmax = self.v1_softmax.cuda()
        self.v2_softmax = self.v2_softmax.cuda()

    def cpu(self):
        self.weight = self.weight.cpu()
        self.prior = self.prior.cpu()
        self.prior_me = self.prior_me.cpu()
        self.prior_ms = self.prior_ms.cpu()
        self.v1_sigmoid = self.v1_sigmoid.cpu()
        self.v2_sigmoid = self.v2_sigmoid.cpu()
        self.v1_softmax = self.v1_softmax.cpu()
        self.v2_softmax = self.v2_softmax.cpu()

# Multi-expert-BCELoss for 12 outputs.
class MultiExpertLoss(nn.Module):

    # The default is 12 experts.
    def __init__(self, cls_num_list, N, prior, symbiotic=4):
        super(MultiExpertLoss, self).__init__()

        self.C = len(cls_num_list)
        self.weight = N / cls_num_list

        self.prior_me = 1 - prior
        self.prior_me[self.prior_me != 1] = 0
        self.prior_ms = prior.clone()
        self.prior_ms[self.prior_ms != 1] = 0
        self.prior_ms -= torch.eye(self.C)

        self.v1_sigmoid = torch.log(N / cls_num_list - 1 + eps)
        self.v2_sigmoid = self.inverse(torch.log(N / cls_num_list - 1 + eps))
        self.v1_softmax = torch.log(cls_num_list / N + eps)
        self.v2_softmax = self.inverse(torch.log(cls_num_list / N + eps))

        self.symbiotic = symbiotic

    def inverse(self, v):
        value, idx0 = torch.sort(v)
        _, idx1 = torch.sort(idx0)
        idx2 = v.shape[0] - 1 - idx1  # reverse the order
        result = value.index_select(0, idx2)
        return result

    def SymbioticRegularizer(self, logits):
        loss = 0
        for c in range(self.C):
            p = self.prior_ms[c]
            if p.sum():
                parent = logits[:, c]
                leafs = logits[:, p == 1]
                loss_exp = torch.exp(leafs - parent.unsqueeze(1))
                loss_log = torch.log(1 + loss_exp)
                loss += loss_log.mean()
        return loss / torch.count_nonzero(self.prior_ms.sum(1))

    def SigmoidLongTailed(self, logits):
        logits_acti = torch.sigmoid(logits)
        return logits_acti

    def SigmoidBalanced(self, logits):
        logits_acti = torch.sigmoid(logits - self.v1_sigmoid)
        return logits_acti

    def SigmoidInversed(self, logits):
        logits_acti = torch.sigmoid(logits - self.v1_sigmoid + self.v2_sigmoid)
        return logits_acti

    def LocalSoftmaxLongTailed(self, logits):
        logits_exp = torch.exp(logits)
        denominator = torch.matmul(logits_exp, self.prior_me) + logits_exp
        logits_acti = logits_exp / (denominator + eps)
        return logits_acti

    def LocalSoftmaxBalanced(self, logits):
        logits_exp = torch.exp(logits + self.v1_softmax)
        denominator = torch.matmul(logits_exp, self.prior_me) + logits_exp
        logits_acti = logits_exp / (denominator + eps)
        return logits_acti

    def LocalSoftmaxInversed(self, logits):
        logits_exp = torch.exp(logits + self.v1_softmax - self.v2_softmax)
        denominator = torch.matmul(logits_exp, self.prior_me) + logits_exp
        logits_acti = logits_exp / (denominator + eps)
        return logits_acti

    def activate(self, logits):
        logits_acti = logits.clone()
        logits_acti[0] = self.SigmoidLongTailed(logits_acti[0])
        logits_acti[1] = self.SigmoidBalanced(logits_acti[1])
        logits_acti[2] = self.SigmoidInversed(logits_acti[2])
        logits_acti[3] = self.SigmoidLongTailed(logits_acti[3])
        logits_acti[4] = self.SigmoidBalanced(logits_acti[4])
        logits_acti[5] = self.SigmoidInversed(logits_acti[5])
        logits_acti[6] = self.LocalSoftmaxLongTailed(logits_acti[6])
        logits_acti[7] = self.LocalSoftmaxBalanced(logits_acti[7])
        logits_acti[8] = self.LocalSoftmaxInversed(logits_acti[8])
        logits_acti[9] = self.LocalSoftmaxLongTailed(logits_acti[9])
        logits_acti[10] = self.LocalSoftmaxBalanced(logits_acti[10])
        logits_acti[11] = self.LocalSoftmaxInversed(logits_acti[11])
        return logits_acti

    # logits: [12, N, *], target: [N, *]
    def forward(self, logits, target):

        logits_acti = torch.clamp(self.activate(logits), eps, 1 - eps)
        loss = 0
        for i in range(12):
            BCE = - self.weight * (target * torch.log(logits_acti[i]) + (1 - target) * torch.log(1 - logits_acti[i]))
            loss += BCE.mean()
        for j in [3, 4, 5, 9, 10, 11]:
            loss += self.symbiotic * self.SymbioticRegularizer(logits_acti[j])

        return loss

    def cuda(self):
        self.weight = self.weight.cuda()
        self.prior_me = self.prior_me.cuda()
        self.prior_ms = self.prior_ms.cuda()
        self.v1_sigmoid = self.v1_sigmoid.cuda()
        self.v2_sigmoid = self.v2_sigmoid.cuda()
        self.v1_softmax = self.v1_softmax.cuda()
        self.v2_softmax = self.v2_softmax.cuda()

    def cpu(self):
        self.weight = self.weight.cpu()
        self.prior_me = self.prior_me.cpu()
        self.prior_ms = self.prior_ms.cpu()
        self.v1_sigmoid = self.v1_sigmoid.cpu()
        self.v2_sigmoid = self.v2_sigmoid.cpu()
        self.v1_softmax = self.v1_softmax.cpu()
        self.v2_softmax = self.v2_softmax.cpu()

# Pair-wise Ranking Loss for CVPR2017 paper "Improving Pairwise Ranking for Multi-label Image Classification".
def LogSumExpPairwiseLoss(y_pred, y_true):

    assert len(y_pred) == len(y_true)
    batch_size = len(y_pred)
    loss = 0

    for i in range(batch_size):
        positive = y_pred[i, y_true[i] == 1.0]
        negative = y_pred[i, y_true[i] == 0.0]

        loss_exp = torch.exp(negative.unsqueeze(1) - positive.unsqueeze(0))
        loss_sum = torch.sum(loss_exp)
        loss_log = torch.log(1 + loss_sum)

        loss += loss_log

    return loss / batch_size

# Class for Pair-wise Ranking Loss.
class LSPE(nn.Module):

    # logits: [N, *], target: [N, *]
    def forward(self, logits, target):
        loss = LogSumExpPairwiseLoss(logits, target)
        return loss

# Multi-expert Class for Pair-wise Ranking Loss.
class MultiExpertLSPE(nn.Module):

    # logits: [12, N, *], target: [N, *]
    def forward(self, logits, target):

        loss = 0
        for i in range(12):
            loss += LogSumExpPairwiseLoss(logits[i], target)

        return loss

# Multi-Label Softmax for IEEE TMI 2023 paper "Multi-Label Softmax Networks for Pulmonary Nodule Classification Using Unbalanced and Dependent Categories".
class MLSoftmax(nn.Module):
    def __init__(self, gamma_pos=1., gamma_neg=1.):
        super(MLSoftmax, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, outputs, targets):
        targets = targets.float()
        outputs = (1 - 2 * targets) * outputs
        y_pred_neg = outputs - targets * 1e15
        y_pred_pos = outputs - (1 - targets) * 1e15
        zeros = torch.zeros_like(outputs[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

        neg_loss = (1 / self.gamma_neg) * torch.log(torch.sum(torch.exp(self.gamma_neg * y_pred_neg), dim=-1))
        pos_loss = (1 / self.gamma_pos) * torch.log(torch.sum(torch.exp(self.gamma_pos * y_pred_pos), dim=-1))

        loss = torch.mean(neg_loss + pos_loss)
        return loss

# Multi-expert Class for Multi-Label Softmax.
class MultiExpertMLSoftmax(nn.Module):
    def __init__(self, gamma_pos=1., gamma_neg=1.):
        super(MultiExpertMLSoftmax, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, logits, targets):
        targets = targets.float()

        loss = 0
        for i in range(12):
            outputs = logits[i]
            outputs = (1 - 2 * targets) * outputs
            y_pred_neg = outputs - targets * 1e15
            y_pred_pos = outputs - (1 - targets) * 1e15
            zeros = torch.zeros_like(outputs[..., :1])
            y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
            y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)

            neg_loss = (1 / self.gamma_neg) * torch.log(torch.sum(torch.exp(self.gamma_neg * y_pred_neg), dim=-1))
            pos_loss = (1 / self.gamma_pos) * torch.log(torch.sum(torch.exp(self.gamma_pos * y_pred_pos), dim=-1))

            loss += torch.mean(neg_loss + pos_loss)

        return loss

if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    y_true = torch.randint(0, 2, [100, 254]).float().cuda()
    y_pred = torch.randn(y_true.shape).cuda()
    prior = pd.read_excel('Prior.xlsx')
    keys = prior.columns[1:]
    prior = prior[keys].values
    prior = torch.from_numpy(prior).float()

    cls_num_list = y_true.sum(0)
    N = y_true.__len__()
    weights = N / cls_num_list

    criterion1 = torch.nn.BCEWithLogitsLoss(weight=weights)
    criterion1.cuda()
    criterion2 = torch.nn.CrossEntropyLoss(weight=weights)
    criterion2.cuda()
    criterion3 = BCELoss(cls_num_list, N, 'LocalSoftmax', prior, k1=1, k2=1, symbiotic=5, suppress=9)
    criterion3.cuda()

    loss1 = criterion1(y_pred, y_true)
    loss2 = criterion2(y_pred, y_true)
    loss3 = criterion3(y_pred, y_true)

    y_pred = torch.randn([12, 100, 254]).cuda()
    criterion = MultiExpertLoss(cls_num_list, N, prior)
    criterion.cuda()
    loss = criterion(y_pred, y_true)
    print(loss)