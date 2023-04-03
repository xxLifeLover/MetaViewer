import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
import os
import json


def normalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)


def process_data(file_path):
    data = sio.loadmat(file_path)
    views = []
    for key in data.keys():
        if 'x' in key:
            views.append(data[key].astype('float16'))
    label = np.squeeze(data['gt'].astype('int64'))
    if np.min(label) == 0:
        label += 1
    return views, label


class get_data(Dataset):
    def __init__(self, file_path, file_name, split, view=-1):
        self.file_path = file_path
        self.file_name = file_name
        self.data, self.targets = process_data(os.path.join(file_path, file_name+'.mat'))
        # self.data = [normalize(x, 0) for x in self.data]
        self.view = [i for i in range(len(self.data))] if view == -1 else view
        with open(os.path.join(file_path, file_name + '_split.json'), "r") as f:
            row_data = json.load(f)
        for v in self.view:
            self.data[v] = self.data[v][row_data[split]]
        self.targets = self.targets[row_data[split]]
        print('DATA SHAPE: ', [self.data[v].shape for v in self.view], self.targets.shape)
        self.samples_num = len(self.data[self.view[0]])

    def get_num_class(self):
        num_class = len(np.unique(self.targets))
        return num_class

    def get_full_data(self):
        data = [self.data[v][:] for v in self.view]
        target = self.targets
        return data, target

    def __len__(self):
        return self.samples_num

    def __getitem__(self, index):
        data = [self.data[v][index] for v in self.view]
        target = self.targets[index]
        return data, target
