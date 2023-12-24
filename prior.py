#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#@author: Jiewei Lai

import os
import sys
import h5py
import scipy
import torch
import shutil
import random
import warnings
import argparse
import setproctitle
warnings.filterwarnings('error')

import numpy as np
import pandas as pd
import neurokit2 as nk
import torch.optim as optim
import sklearn.metrics as M
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from scipy import signal
from torch.autograd import Variable
from torch.utils.data import DataLoader

import MSDNN

seed = 10000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#=======================================================================================================================
# Read the excel of hierarchical labels.
def read_hierarchic_label_excel(HierarchicLabelExcel):

    hierarchic_label = pd.read_excel(HierarchicLabelExcel)
    codes_dict1, codes_dict2, codes_dict3 = dict(), dict(), dict()

    for i, row in hierarchic_label.iterrows():

        if type(row['一级标签']) is str:
            term1 = row['一级标签']
            codes_dict1[term1] = list()
            codes_dict1[term1].append(str(row['标签ID']))
        else:
            codes_dict1[term1].append(str(row['标签ID']))

        if type(row['二级标签']) is str:
            term2 = row['二级标签']
            codes_dict2[term2] = list()
            codes_dict2[term2].append(str(row['标签ID']))
        else:
            codes_dict2[term2].append(str(row['标签ID']))

        codes_dict3[row['三级标签']] = str(row['标签ID'])

    return codes_dict1, codes_dict2, codes_dict3
#=======================================================================================================================

if __name__=='__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    rootpath = os.path.dirname(os.path.abspath('..'))
    ecgpath = os.path.join(rootpath, 'ECG/训练集数据')
    datapath = os.path.join(rootpath, 'data')
    dsetpath = os.path.join(datapath, 'datasets')
    setproctitle.setproctitle("laijiewei")

    # Get the subordination of hierarchical labels.
    HierarchicLabelExcel = os.path.join(ecgpath, "标签分类.xlsx")
    codes_dict1, codes_dict2, codes_dict3 = read_hierarchic_label_excel(HierarchicLabelExcel)

    # Load data and labels.
    data = h5py.File(os.path.join(dsetpath, 'ECGDataSetV2.h5'), 'r')
    ecgs = data['ecgs']
    ids = np.copy(data['ids'])
    y1 = pd.DataFrame(data['y1'], columns=codes_dict1.keys())
    y2 = pd.DataFrame(data['y2'], columns=codes_dict2.keys())
    y3 = pd.DataFrame(data['y3'], columns=codes_dict3.keys())

    # Selection of labels with a number of positive samples greater than 50.
    nums1 = y1.sum().sort_values(ascending=False)
    keys1 = [key for key, num in zip(nums1.index, nums1.values) if num >= 50]
    nums2 = y2.sum().sort_values(ascending=False)
    keys2 = [key for key, num in zip(nums2.index, nums2.values) if num >= 50]
    nums3 = y3.sum().sort_values(ascending=False)
    keys3 = [key for key, num in zip(nums3.index, nums3.values) if num >= 50]

    # Remove duplicate labels.
    set1, set2, set3 = set(keys1), set(keys2), set(keys3)
    for key in list(set1 & set2):
        keys2.remove(key)
    for key in list(set2 & set3):
        keys3.remove(key)

    keys = keys1 + keys2 + keys3
    print("There are {} keys.".format(len(keys)))
    y = pd.concat([y1[keys1], y2[keys2], y3[keys3]], axis=1).values
    y = torch.from_numpy(y).cuda()
    prior = pd.DataFrame(index=keys, columns=keys)

    # Compute the concurrence conditional probability of hierarchical multiple labels.
    for j, key in enumerate(keys):

        sys.stdout.write('\r')
        sys.stdout.write('Processing: {}-{}|{}.'.format(key, j, len(keys)))
        sys.stdout.flush()

        P_j = torch.unsqueeze(y[:, j], dim=0)
        P_ij = torch.matmul(P_j, y)
        p_c = P_ij / P_j.sum()
        prior[key] = torch.squeeze(p_c).data.cpu().numpy()

    prior.to_excel('Prior.xlsx')