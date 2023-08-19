#################
# Set of utility functions. Some are deprecated after data management optimisation
#################

import os

import open3d as o3d
import random
import numpy as np
import json

import torch


def get_point_cloud(path, image_id):
    image_id = '{:0>8}'.format(image_id)
    pcd = o3d.io.read_point_cloud(path + "/ptcloud_hd" + image_id + ".ply")
    pcd = pcd.voxel_down_sample(1.5)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.5)
    return pcd.select_by_index(ind)


def print_progress(frame, total_frames):
    if frame % 100 == 0:
        print(str(int(frame*100/total_frames)) + "%", flush=True)


def get_sample_pcd(pcd_list, sample_size):
    while len(pcd_list) < sample_size:
        pcd_list = pcd_list + pcd_list
    cloud_list = random.sample(pcd_list, sample_size)
    return cloud_list


def get_skeleton_points(index, relative_path):
    image_id = '{:0>8}'.format(index)
    skeleton_file = json.load(open(relative_path + "/skeleton/body3DScene_" + image_id + ".json"))
    skeleton = np.reshape(np.asarray(skeleton_file['bodies'][0]['joints19']), (19, 4))
    skeleton = np.hsplit(skeleton, np.array([3, 6]))
    return skeleton[0]


def set_fixed_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"The seed is set to {seed}")


def cam_list_tostring(cams):
    cams_string = ""
    for i in range(len(cams)):
        cams_string += str(cams[i])

    return cams_string
