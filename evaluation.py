###########################
# Function used to evaluate the models and generate the loss graphs
###########################
import h5py
import matplotlib.pyplot as plt
import torch
import json
import model2 as model
import utilities
import pandas as pd
import numpy as np
import math
import tabulate
import random


def show_loss(train_loss_path, val_loss_path, name):
    with open(train_loss_path, 'r') as fp:
        loss = json.load(fp)
    train_loss = []
    for i in range(50):
        train_loss.extend([np.mean(loss[int(i*(len(loss)/50)):int((i+1)*(len(loss)/50))])])

    with open(val_loss_path, 'r') as fp:
        val_loss = json.load(fp)

    fig = plt.figure()
    plt.title("Train and validation loss" + name)

    plt.plot(train_loss[10:])
    plt.plot(val_loss[10:], 'r')
    plt.savefig("fig_loss_cut/"+name)


def test_model_multicam(cams, vs_cam, model_name, test_res=5000):
    pointmodel = model.PointNet().to('cuda')

    print('Test resolution ' + str(test_res) + ' vs ' + utilities.cam_list_tostring(vs_cam))

    #pointmodel.load_state_dict(torch.load('model/model_' + str(len(cams)) + 'cam_'+ utilities.cam_list_tostring(cams)+'/epoch50'))
    pointmodel.load_state_dict((torch.load(model_name)))
    pointmodel.eval()

    csv_file_path = 'Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose2/csv/train_' +str(len(vs_cam)) + 'cam_' + utilities.cam_list_tostring(vs_cam) + '.csv'
    csv_file = pd.read_csv(csv_file_path).to_numpy()
    distance = np.empty((len(csv_file), 20))

    skeleton = h5py.File('D:/MEMBEN/skeleton_test.hdf5', 'r')
    pcd_file = h5py.File('D:/MEMBEN/all_pcd_test.hdf5', 'r')

    for k in range(int(len(csv_file)/40)):
        ground_truth = skeleton[str(csv_file[k*40][0])][:]
        ground_truth = np.transpose(np.reshape(ground_truth, (3, 19)), (1, 0))
        model_result = get_model_result_multicam(pointmodel, str(csv_file[k*40][0]), vs_cam, pcd_file, test_res)
        distance_frame = np.zeros(20)
        for l in range(19):
            distance_frame[l] += math.dist(ground_truth[l], model_result[l])
        distance_frame[19] = np.sum(distance_frame) / 19
        distance[k] = distance_frame
    np.save('resolution/5k_vs_' + str(test_res) + utilities.cam_list_tostring(cams)+'_VS_'+ utilities.cam_list_tostring(vs_cam), distance)


def get_model_result_multicam(pointmodel, frame, cams, pcd_file, test_res=5000):
    l = []
    for i in range(1, len(cams) + 1):
        noise = 0#random.choices([-3, -2, -1, 0, 1, 2, 3], weights=(2, 8, 15, 50, 15, 8, 2))[0]
        try:
            if str(cams[i - 1]) + '/' + str(int(frame)+noise) in pcd_file:
                l += pcd_file[str(cams[i - 1])][str(int(frame)+noise)][:].tolist()
            elif str(cams[i - 1]) + '/' + str(frame) in pcd_file:
                l += pcd_file[str(cams[i - 1])][frame][:].tolist()
        except:
            continue

    cloud_list = utilities.get_sample_pcd(l, test_res)
    cloud = np.transpose(cloud_list, (1, 0))
    cloud_stack = np.expand_dims(cloud, 0)

    cloud_stack = torch.from_numpy(cloud_stack).float()

    output = pointmodel(cloud_stack.to('cuda'))
    output = output.cpu().detach().numpy()
    output = np.reshape(output, (3, 19))
    return np.transpose(output, (1, 0))
