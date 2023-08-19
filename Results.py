import json
import math
import random

import torch
import torchsummary
import utilities
import numpy as np
import open3d as o3d
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')

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


def get_results(image_id):
    model = PointNet().to('cpu')

    model.load_state_dict(torch.load("longtestavecsaveloss.pt", map_location=torch.device('cpu')))
    model.eval()

    pcd_name = "pcd_list/" + str(image_id)
    with open(pcd_name, "r") as fp:
        l = json.load(fp)
    cloud_list = random.sample(l, 5000)
    cloud = np.transpose(cloud_list, (1, 0))
    cloud_stack = np.expand_dims(cloud, 0)

    cloud_stack = torch.from_numpy(cloud_stack).float()

    output = model(cloud_stack)
    output = output.detach().numpy()
    output = np.reshape(output, (3, 19))
    output = np.transpose(output, (1, 0))

    articulation = o3d.geometry.PointCloud()
    articulation.points = o3d.utility.Vector3dVector(output)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_list)

    truth_skeleton = utilities.get_skeleton_position(image_id)

    return pcd, articulation, truth_skeleton


def get_accuracy(image_id):
    real_positions = utilities.get_skeleton_points(image_id)
    pcd, articulation, truth_skeleton = get_results(image_id)
    guessed_positions = np.asarray(articulation.points)
    distances = []
    joint_name = ["Neck", "Nose", "Body Center", "Left Shoulder", "Left Elbow", "Left Wrist", "Left Hip", "Left Knee", "Left Ankle", "Right Shoulder", "Right Elbow", "Right Wrist", "Right Hip", "Right Knee", "Right Ankle", "left Eye", "Left Ear", "Right Eye", "Right Ear"]
    for i in range(19):
        distances.append(math.dist(real_positions[i], guessed_positions[i]))

    print('Mean error : ' + str(sum(distances)/(len(distances)+1)))
    out = {}
    for key, value in zip(joint_name, distances):
        out[key] = value
    return out


def display_results(image_id):
    pcd, skeleton, truth_skeleton = get_results(image_id)
    utilities.plot_point_clouds(pcd, skeleton)
    plot_skeleton(np.asarray(skeleton.points)*-1)


def plot_skeleton(skeleton):
    bone_list = np.array(
        [[0, 3], [0, 9], [3, 4], [4, 5], [3, 6], [6, 7], [7, 8], [9, 10], [10, 11], [9, 12], [12, 13], [13, 14],
         [6, 12], [16, 15], [15, 1], [1, 17], [17, 18], [0, 2]])
    x = skeleton[:, 0]+100
    y = (skeleton[:, 1]*-1)+200
    z = skeleton[:, 2]+100

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.axes.set_xlim3d(left=200, right=0)
    ax.axes.set_ylim3d(bottom=0, top=200)
    ax.axes.set_zlim3d(bottom=0, top=200)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    for i in range(19):
        if i !=15:
            if i !=17:
                ax.text(x[i], z[i], y[i], str(i))

    plt.title('Skeleton')

    for i in range(len(bone_list)):
        bone = bone_list[i]
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        plt.plot([x[bone[0]], x[bone[1]]], [z[bone[0]], z[bone[1]]], [y[bone[0]], y[bone[1]]], color[i % 7])
    plt.show()


#print(get_accuracy(4350))
#display_results(4350)
skeleton = utilities.get_skeleton_points(1300)
plot_skeleton(skeleton*-1)
