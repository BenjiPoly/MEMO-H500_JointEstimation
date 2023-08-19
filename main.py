import pandas as pd

import pre_processing
import pre_processing as pp
import train
import train2
import train3
import utilities
import evaluation
import time
import numpy as np
import more_itertools
import os

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    utilities.set_fixed_seed(29051999)

    train_folder = 'Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose1'
    test_folder = 'Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose2'
    #cameras_combinations = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [2, 5], [2, 7], [4, 7], [5, 10], [1, 2], [2, 5, 9], [1, 2, 7], [1, 2, 3], [2, 5, 6, 9], [1, 2, 7, 8]]
    cameras_combinations = [[2, 5, 6, 9]]
    cameras = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #cameras_combinations = [[2, 5, 6, 9], [1, 2, 3], [1, 2, 7], [2, 5], [1, 2, 7, 8]]
    for res in range(11000, 12001, 1000):
        for i in range(1, 5):
            vs_cameras_combinations = more_itertools.distinct_combinations(cameras, i)
            for vs_cams in vs_cameras_combinations:
                if os.path.exists('resolution/2_5k_vs_' + str(res) + utilities.cam_list_tostring(cameras_combinations[0]) + '_VS_' + utilities.cam_list_tostring(vs_cams) + '.npy') == False:
                    model_name = 'model/model_4cam_2569/epoch50'
                    evaluation.test_model_multicam(cameras_combinations[0], vs_cams, model_name, test_res=res)
    #for cams in cameras_combinations:
    #    train_loss_path = "train_loss/train_loss_model_" + str(len(cams)) + "cam_" + utilities.cam_list_tostring(cams) + "/epoch50"
    #    val_loss_path = "val_loss/val_loss_model_" + str(len(cams)) + "cam_" + utilities.cam_list_tostring(cams)
    #    name = "model_" + str(len(cams)) + "cam_" + utilities.cam_list_tostring(cams)
    #    evaluation.show_loss(train_loss_path, val_loss_path, name)
    # cameras = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # for cams in cameras_combinations:
    #     for i in range(1, 5):
    #         vs_cameras_combinations = more_itertools.distinct_combinations(cameras, i)
    #         for vs_cams in vs_cameras_combinations:
    #             if os.path.exists('noise/train_noise_' + utilities.cam_list_tostring(cams) + '_VS_' + utilities.cam_list_tostring(vs_cams) + '.npy') == False:
    #                 model_name = 'model/noise_model_' + str(len(cams)) + 'cam_' + utilities.cam_list_tostring(cams)+'/epoch50'
    #                 evaluation.test_model_multicam(cams, vs_cams, model_name)
    # for i in range(len(cameras_combinations)):
    #     model_file = '7_5k_model_' + str(len(cameras_combinations[i])) + 'cam_' + utilities.cam_list_tostring(cameras_combinations[i])
    #     train3.train_model(train_folder, test_folder, model_file, cams=cameras_combinations[i], epochs=50, learning_rate=0.0001)
    #     #evaluation.test_model_multicam(cameras_combinations[i])

# First step, finding the valid frames
#    for i in range(1, 11):
#        pp.create_csv_valid_frame("train_1cam_"+str(i)+".csv", 27678, "171204_pose1/pointcloud_1cam_"+str(i))

# Second step, create list of points instead of .ply
#    for i in range(1, 11):
#        pp.ply_to_list("train_1cam_"+str(i)+".csv", "171204_pose2/pointcloud_1cam_"+str(i), str(i), 'Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose2/')

# The train the networks on a fixed seed, here for all the single cameras
# utilities.set_fixed_seed(29051999)
#     for i in range(1, 11):
#         train_folder = 'Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose1'
#         test_folder = 'Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose2'
#         model_file = 'model_1cam_' + str(i)
#         cam = [i]
#         train.train_model(train_folder, test_folder, model_file, cams=cam, epochs=30)

# Compute the tests and save values
# evaluation.test_model()

# Create table with data
#    mean, var = evaluation.compute_mean_articulation(19)
#    evaluation.print_table(mean)

# Multicam tests
# cameras_combinations = [[2, 5], [2, 7], [4, 7], [5, 10], [1, 2], [2, 5, 9], [1, 2, 7], [1, 2, 3], [2, 5, 6, 9],
#                         [1, 2, 7, 8]]
# train_folder = 'Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose1'
# test_folder = 'Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose2'
# model_file = 'model_2cam_25'
# cam = [2, 5]
# train.train_model(train_folder, test_folder, model_file, cams=cam, epochs=30)


# train_folder = 'Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose1'
# test_folder = 'Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose2'
# model_file = 'model_4cam_2569'
# cam = [2, 5, 6, 9]
# train2.train_model(train_folder, test_folder, model_file, cams=cam, epochs=30, learning_rate=0.0001)
# evaluation.test_model_multicam(cam)
# part_dist = np.transpose(np.load('distances_2cam_2569.npy'))
# for i in range(20):
#     mean = np.mean(part_dist[i])
#     var = np.var(part_dist[i])
#     print('Articulation ID ' + str(i) + ' : Mean = ' + str(mean) + ', Var = ' + str(var))
