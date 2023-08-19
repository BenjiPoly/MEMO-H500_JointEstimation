import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd

import utilities
import random

class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 57)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x, _ = torch.max(x, dim=2)
        x = self.relu(self.bn6(self.fc1(x)))
        x = self.relu(self.bn7(self.fc2(x)))
        x = self.fc3(x)
        return x


class ToTensor(object):
    def __call__(self, point_cloud, articulations):
        return torch.from_numpy(point_cloud).float(), torch.from_numpy(articulations).float()


class PointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, path, cams, pcd_file, skeletons, transform=None):
        self.transform = transform
        self.cams = cams
        self.ids = pd.read_csv(path + '/csv/train_' + str(len(cams)) + 'cam_' + utilities.cam_list_tostring(cams) + '.csv').to_numpy()
        self.path = path
        self.pcd_file = pcd_file
        self.skeletons = skeletons

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        l = []
        l_safe = []
        for i in range(1, len(self.cams)+1):
            try:
                noise = random.choices([-3, -2, -1, 0, 1, 2, 3], weights=(2, 8, 15, 50, 15, 8, 2))[0]
                if str(self.cams[i-1])+'/' + str(self.ids[index+noise][0]) in self.pcd_file:
                    l += self.pcd_file[str(self.cams[i-1])][str(self.ids[index+noise][0])][:].tolist()
                elif str(self.cams[i - 1]) + '/' + str(self.ids[index][0]) in self.pcd_file:
                    l += self.pcd_file[str(self.cams[i - 1])][str(self.ids[index][0])][:].tolist()
            except:
                if str(self.cams[i - 1]) + '/' + str(self.ids[index][0]) in self.pcd_file:
                    l += self.pcd_file[str(self.cams[i - 1])][str(self.ids[index][0])][:].tolist()
        if len(l) > 0:
            cloud = np.transpose(utilities.get_sample_pcd(l, 5000), (1, 0))
        else:
            print('We have a problem')
        skeleton = self.skeletons[str(self.ids[index][0])][:]
        sample = (cloud, skeleton)

        if self.transform:
            sample = self.transform(sample)

        return sample
