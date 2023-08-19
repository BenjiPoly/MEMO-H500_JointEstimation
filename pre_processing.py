import csv
import json
import numpy as np
import pandas as pd
import open3d as o3d
import h5py

import utilities


def create_csv_valid_frame(csv_name, frame_number, cloud_path, relative_path):
    print("Creating csv " + csv_name)
    with open(csv_name, 'w', newline='') as out:
        writer = csv.writer(out, delimiter=';')

        for i in range(1, frame_number):
            formatnum = f'{i:0>8}'
            try:
                skeleton_file = json.load(open(relative_path +'/skeleton/body3DScene_' + formatnum + '.json'))
            except:
                print("No skeleton for frame " + formatnum)
                continue
            pcd = utilities.get_point_cloud('Y:/MFE/bhainaut/panoptic-toolbox-master/'+cloud_path, i)

            try:
                skeleton = np.reshape(np.asarray(skeleton_file['bodies'][0]['joints19']), (19, 4))
                if len(pcd.points) > 2000:
                    writer.writerows([[i]])
                    if i%100 == 0:
                        print(str(int(i*100/frame_number))+"%")
                else:
                    print("Less than 3000 points in frame " + str(i))
            except IndexError:
                print("Incomplete skeleton for frame " + formatnum)


def ply_to_list(csv_file, cloud_path, out_path, relative_path):
    print("Generating list of point cloud : " + csv_file + '\n')
    ids = pd.read_csv(relative_path + 'csv/'+ csv_file).to_numpy()

    for index in range(len(ids)):
        utilities.print_progress(index, len(ids))
        cloud = utilities.get_point_cloud('Y:/MFE/bhainaut/panoptic-toolbox-master/'+cloud_path, ids[index][0])
        cloud = np.asarray(cloud.points)
        path = relative_path + "pcd_list_1cam_" + out_path + "/" + str(ids[index][0]) + ".json"
        with open(path, "w") as fp:
            json.dump(cloud.tolist(), fp)


def fuse_csv(cams):
    frame_ids = []
    for i in cams:
        frame_1cam = pd.read_csv("Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose2/csv/train_1cam_" + str(i) + ".csv").to_numpy()
        frame_ids += frame_1cam.tolist()
    frame_ids = np.asarray(frame_ids)
    frame_ids = np.unique(frame_ids)
    frame_ids = frame_ids.tolist()

    csv_name = 'train_' + str(len(cams)) + 'cam_' + utilities.cam_list_tostring(cams) + '.csv'
    with open(csv_name, 'w', newline='') as out:
        writer = csv.writer(out, delimiter=';')
        for i in range(len(frame_ids)):
            writer.writerows([[frame_ids[i]]])
            if i % 1000 == 0:
                print('frame ' + str(i) + '/' + str(len(frame_ids)))


def list_to_hdf5():
    with h5py.File('all_pcd_test.hdf5', 'w') as f:
        for i in range(10):
            ids = pd.read_csv("Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose2/csv/train_1cam_" + str(i+1) + ".csv").to_numpy()
            grp = f.create_group(str(i+1))

            for j in range(len(ids)):
                if j % 100 == 0:
                    print('Cam ' + str(i+1) + ' Frame ' + str(j))
                with open('Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose2/pcd_list_1cam_' + str(i+1) + '/' + str(ids[j][0]) + '.json', 'r') as fp:
                    frame = json.load(fp)
                grp.create_dataset(str(ids[j][0]), data=frame)


def skeleton_to_hdf5():
    with h5py.File('skeleton_test.hdf5', 'w') as f:
        for i in range(4500):
            formatnum = f'{i:0>8}'
            try:
                skeleton_file = json.load(open('Y:/MFE/bhainaut/panoptic-toolbox-master/171204_pose2/skeleton/body3DScene_' + formatnum + '.json'))
                skeleton = np.reshape(np.asarray(skeleton_file['bodies'][0]['joints19']), (19, 4))
                skeleton = np.hsplit(skeleton, np.array([3, 6]))[0]
                skeleton = np.transpose(skeleton, (1, 0))
                skeleton = np.reshape(skeleton, 57)
                if i % 100 == 0:
                    print(i)
                f.create_dataset(str(i), data=skeleton)
            except:
                continue
