#!/usr/bin/env python3

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

sys.path.append('../')
try:
    import Loss
except:
    from loss import Loss
import DRClassifier

seed = 10000
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#=======================================================================================================================
def train(net, trainLoader, criterion, optimizer, epoch, args, logs):

    net.train()
    nProcessed = 0
    train_loss = list()
    nTrain = len(trainLoader.dataset)
    criterion.cuda()

    for batch_idx, (batch_data, batch_label) in enumerate(trainLoader):

        batch_data = Variable(batch_data).cuda()
        batch_label = Variable(batch_label).cuda()

        optimizer.zero_grad()
        batch_logits = net(batch_data)
        batch_loss = criterion(batch_logits, batch_label)
        batch_loss.backward()
        optimizer.step()

        nProcessed += len(batch_data)
        partialEpoch = epoch + batch_idx / len(trainLoader)
        train_loss.append(batch_loss.item())

        sys.stdout.write('\r')
        sys.stdout.write('Train Epoch: {:.2f}|{} [{}/{}] \t Loss: {:.4f}'.
            format(partialEpoch, args.num_epochs, nProcessed, nTrain, batch_loss.item()))
        sys.stdout.flush()

    train_loss = np.mean(train_loss)

    sys.stdout.write('\r')
    sys.stdout.write('Train Epoch: {:.2f}|{} [{}/{}] \t Loss: {:.4f} '.
                     format(epoch + 1, args.num_epochs, nProcessed, nTrain, train_loss))
    sys.stdout.flush()

    logs.result.loc[epoch, 'train_loss'] = train_loss

def test(net, testLoader, criterion, epoch, args, logs):

    net.eval()
    samples = len(testLoader.dataset)
    criterion.cpu()

    labels = torch.empty([samples, len(logs.keys)])
    logits = torch.empty([samples, len(logs.keys)])

    with torch.no_grad():

        for batch_idx, (batch_data, batch_label) in enumerate(testLoader):

            bs = len(batch_data)
            batch_data = Variable(batch_data).cuda()
            batch_logits = net(batch_data)

            logits[batch_idx * args.batch_size:batch_idx * args.batch_size + bs] = batch_logits.data.cpu()
            labels[batch_idx * args.batch_size:batch_idx * args.batch_size + bs] = batch_label

    test_loss = criterion(logits, labels).item()
    logs.result.loc[epoch, 'test_loss'] = test_loss

    logits = criterion.activate(logits)
    for i, v in enumerate(logs.columns):
        AUPRC = M.average_precision_score(labels[:, i].data.numpy(), logits[:, i].data.numpy())
        logs.result.loc[epoch, v] = AUPRC
    mean_AUPRC = logs.result.loc[epoch, logs.columns].mean()
    logs.result.loc[epoch, 'mean_AUPRC'] = mean_AUPRC

    print('\nTest set: {} \t Average loss: {:.4f}, AUPRC: {:.4f}'.format(samples, test_loss, mean_AUPRC))

# Save training and testing logs
class Logs(object):

    def __init__(self, keys, args):
        self.keys = keys
        self.args = args
        self.inherent = ['train_loss', 'test_loss', 'mean_AUPRC', 'epoch', 'best_AUPRC']
        self.columns = ['AUPRC_' + key for key in self.keys]
        self.result = pd.DataFrame(columns=self.inherent + self.columns)
        self.result.loc[0, 'best_AUPRC'] = None

    # Save the model with the best test mean AUPRC
    def save_logs(self, epoch, net):

        mean_AUPRC = self.result.loc[epoch, 'mean_AUPRC']
        best_AUPRC = self.result.loc[0, 'best_AUPRC']

        if best_AUPRC == None:
            print("The test AUPRC of the model is {:.4f}.\n".format(mean_AUPRC))
            self.result.loc[0, 'epoch'] = epoch
            self.result.loc[0, 'best_AUPRC'] = mean_AUPRC
            torch.save(net.state_dict(), os.path.join(self.args.save, 'model.pth'))
        elif best_AUPRC <= mean_AUPRC:
            print("The test AUPRC of the model is improved from {:.4f} to {:.4f}.\n".format(best_AUPRC, mean_AUPRC))
            self.result.loc[0, 'epoch'] = epoch
            self.result.loc[0, 'best_AUPRC'] = mean_AUPRC
            torch.save(net.state_dict(), os.path.join(self.args.save, 'model.pth'))
        else:
            print("The test AUPRC of the model didn't improved from {:.4f}.\n".format(best_AUPRC))

        self.result.to_excel(os.path.join(self.args.save, 'result.xlsx'))

    def plot_result(self, name='result'):

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        epoch = self.result.loc[0, 'epoch']

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.result['train_loss'], c='darkred', label='train_loss')
        ax1.plot(self.result['test_loss'], c='darkblue', label='test_loss')
        ax1.semilogy()
        ax1.legend(loc=1)
        ax1.set_ylabel('Loss')
        ax2 = ax1.twinx()
        ax2.plot(self.result['mean_AUPRC'], c='purple', label='mean_AUPRC')
        ax2.scatter(epoch, self.result.loc[0, 'best_AUPRC'], c='purple', s=50, marker='*')
        ax2.set_ylabel('AUPRC')
        plt.title('Best test AUPRC: {:.4f}'.format(self.result.loc[0, 'best_AUPRC']))
        plt.savefig(os.path.join(self.args.save, name + '-loss'), dpi=500)
        plt.clf()
        plt.close()

        plt.figure()
        plt.grid()
        for key in self.columns:
            plt.plot(self.result[key], label="{}: {:.4f} ".format(key, self.result.loc[epoch, key]))
        plt.legend(bbox_to_anchor=(1.3, 1.5), ncol=6)
        plt.title('Best AUPRC: {:.4f}'.format(self.result.loc[0, 'best_AUPRC']))
        plt.savefig(os.path.join(self.args.save, name + '-AUPRCs'), dpi=500, bbox_inches='tight')
        plt.clf()
        plt.close()

def learning_rate_seq1(num_epochs=100, learning_rate=0.01):

    half_len = int(num_epochs * 0.45)
    x1 = np.linspace(0.1 * learning_rate, learning_rate, half_len)
    x2 = np.linspace(x1[-1], 0.1 * learning_rate, half_len + 1)[1:]
    x3 = np.linspace(x2[-1], 0.001 * learning_rate, num_epochs - 2 * half_len + 1)[1:]

    x = np.concatenate([x1, x2, x3])

    return x

def learning_rate_seq2(num_epochs=130, learning_rate=0.01):

    one_cycle = num_epochs - 30
    half_len = int(one_cycle * 0.45)
    x1 = np.linspace(0.1 * learning_rate, learning_rate, half_len)
    x2 = np.linspace(x1[-1], 0.1 * learning_rate, half_len + 1)[1:]
    x3 = np.linspace(x2[-1], 0.001 * learning_rate, one_cycle - 2 * half_len + 1)[1:]

    x4 = np.linspace(0.5 * learning_rate, 0.001 * learning_rate, 10)

    x = np.concatenate([x1, x2, x3, x4, x4, x4])

    return x

class ECGDataSet(torch.utils.data.Dataset):
    
    def __init__(self, ecgs, targets, data_idx, transform=None):
        self.ecgs = ecgs
        self.targets = targets
        self.data_idx = data_idx
        self.transform = transform
        
    def __getitem__(self, index):
        index = self.data_idx[index]
        ecg = self.ecgs[index]
        y = self.targets.loc[index].values
        if self.transform is not None:
            ecg = self.transform(ecg)

        return ecg.astype(np.float32), y.astype(np.float32)
        
    def __len__(self):
        return len(self.data_idx)

# Drop out in frequency domain
class ECGFrequencyDropOut(object):
    def __init__(self, rate=0.3, default_len=7500):
        self.rate = rate
        self.default_len = default_len
        self.num_zeros = int(self.rate * self.default_len)

    def __call__(self, data):
        num_zeros = random.randint(0, self.num_zeros)
        zero_idxs = sorted(np.random.choice(np.arange(self.default_len), num_zeros, replace=False))
        data_dct = scipy.fft.dct(data.copy())
        data_dct[:, zero_idxs] = 0
        data_idct = scipy.fft.idct(data_dct)

        return data_idct

# Randomly select signals longer than n seconds and adjust them to 15s
class ECGCropResize(object):
    def __init__(self, n=2, default_len=7500, fs=500):
        self.min_len = n * fs
        self.default_len = default_len

    def __call__(self, data):
        crop_len = random.randint(self.min_len, self.default_len)
        crop_start = random.randint(0, self.default_len - crop_len)
        data_crop = data[:, crop_start:crop_start + crop_len]
        data_resize = np.empty_like(data)
        x = np.linspace(0, crop_len-1, crop_len)
        xnew = np.linspace(0, crop_len-1, self.default_len)
        for i in range(data.shape[0]):
            f = scipy.interpolate.interp1d(x, data_crop[i], kind='cubic')
            data_resize[i] = f(xnew)

        return data_resize

# Select a certain length signal in each heartbeat cycle and set it to zero
class ECGCycleMask(object):
    def __init__(self, rate=0.5, fs=500):
        self.rate = rate
        self.fs = fs

    def __call__(self, data):
        try:
            # Extract R-peaks locations
            _, rpeaks = nk.ecg_peaks(np.float32(data[1]), sampling_rate=self.fs)
            r_peaks = rpeaks['ECG_R_Peaks']
            if len(r_peaks) > 1:
                cycle_len = int(np.mean(np.diff(r_peaks)))
                cut_len = int(self.rate * cycle_len)
                cut_start = random.randint(0, cycle_len - cut_len)
                data_ = data.copy()
                for r_idx in r_peaks:
                    data_[:, r_idx + cut_start:r_idx + cut_start + cut_len] = 0
                return data_
            else:
                return data
        except:
            return data

# Randomly select less than the number of masks and set these channels to zero
class ECGChannelMask(object):
    def __init__(self, masks=6, default_channels=12):
        self.masks = masks
        self.channels = np.arange(default_channels)

    def __call__(self, data):
        masks = random.randint(1, self.masks)
        channels_mask = np.random.choice(self.channels, masks, replace=False)
        data_ = data.copy()
        for channel_mask in channels_mask:
            data_[channel_mask] = 0
        return data_

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=130)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    ecgpath = os.path.join(rootpath, 'ECG/训练集数据')
    datapath = os.path.join(rootpath, 'data')
    dsetpath = os.path.join(datapath, 'datasets')
    args.save = os.path.join(datapath, 'Others/Single/DRClassifier')
    setproctitle.setproctitle("laijiewei")

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save)

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
   y = pd.concat([y1[keys1], y2[keys2], y3[keys3]], axis=1)
   cls_num_list, N = torch.from_numpy(y.sum(0).values).to(torch.float32), y.__len__()

   # Divide the training set and test set.
    info = pd.read_excel(os.path.join(dsetpath, "InfomationV2.xlsx"))
    train_idxs = info.loc[info['is_test'] == 0].index.tolist()
    test_idxs = info.loc[info['is_test'] != 0].index.tolist()
    print('train {}, test {}'.format(len(train_idxs), len(test_idxs)))

    augmentation = [transforms.RandomApply([ECGFrequencyDropOut(rate=0.1)]),
                    transforms.RandomApply([ECGCycleMask(rate=0.4)]),
                    transforms.RandomApply([ECGCropResize(n=2)]),
                    transforms.RandomApply([ECGChannelMask(masks=8)])]

    train_data = ECGDataSet(ecgs, y, train_idxs, transform=transforms.Compose(augmentation))
    test_data = ECGDataSet(ecgs, y, test_idxs)
    trainLoader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=8)
    testLoader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2)

    net = DRClassifier.Net(len(keys1), len(keys2), len(keys3)).cuda()
    paras = sum([p.data.nelement() for p in net.parameters()])
    print('Number of params: {:.2f} M.\n'.format(paras / (1024 ** 2)))
    torch.backends.cudnn.benchmark = True

    # state_dict = torch.load(os.path.join(args.save, 'model.pth'), map_location='cpu')
    # net.load_state_dict(state_dict)

    prior = pd.read_excel(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Prior.xlsx'))
    prior = prior[keys].values
    prior = torch.from_numpy(prior).to(torch.float32)

    criterion = Loss.BCELoss(cls_num_list, N, 'Sigmoid', prior, 0, 0, 0, 0)
    lrs = learning_rate_seq2(args.num_epochs, args.lr)
    logs = Logs(keys, args)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    for epoch in range(args.num_epochs):

        optimizer.param_groups[0]['lr'] = lrs[epoch]
        train(net, trainLoader, criterion, optimizer, epoch, args, logs)
        test(net, testLoader, criterion, epoch, args, logs)
        logs.save_logs(epoch, net)

    logs.plot_result()