################
# Function used to score the models. Due to time constraints, the parametrisation is limited.
# Parameters changed for every model and every parameters
################

import numpy as np
import more_itertools
import utilities
import os
import pandas as pd
import openpyxl

cameras_combinations = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [2, 5], [2, 7], [4, 7], [5, 10], [1, 2], [2, 5, 9], [1, 2, 7], [1, 2, 3], [2, 5, 6, 9], [1, 2, 7, 8]]
cameras = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def score(file_name, j):
    result_global = np.load(file_name)
    result_mean = np.transpose(result_global)[j]
    result_mean = np.sort(result_mean)
    test = result_mean[result_mean[:]<10]
    ecart = np.sqrt(np.var(result_mean))
    score_cam = np.mean(result_mean)
    #score_cam = ecart
    #score_cam = len(test)/len(result_mean)
    #score_cam = np.median(result_mean)
    return score_cam

cameras_combinations = [[2, 5, 6, 9], [1, 2, 7]]
#vs_cameras_combinations = [[2,7], [3,8], [4,9], [4, 10], [1,6], [1,2,7], [4,5,10], [3,8,9], [1,2,7,8],[4,5,9,10]]
vs_cameras_combinations = [[1,9], [2,10], [5,8], [3,6], [4,7], [1,5,8], [2,5,9], [4,6,8], [2,5,6,9], [3,4,6,7]]
score_models = []
working_array = np.zeros((2, 20))


for id, cams in enumerate(cameras_combinations):
    score_model = np.zeros(20)
    print(id)
    #for i in range(1, 5):
        #vs_cameras_combinations = more_itertools.distinct_combinations(cameras, i)
    for vs_cams in vs_cameras_combinations:
        file_name = 'distance/distances_' + utilities.cam_list_tostring(cams) + '_VS_' + utilities.cam_list_tostring(
                    vs_cams) + '.npy'
        if os.path.exists(file_name):
            for j in range(20):
                score_model[j] += score(file_name, j)
    score_models.append(('Model ' + utilities.cam_list_tostring(cams), score_model))
    working_array[id][:] = score_model

# for res in range(1000,10001,1000):
#     score_res = np.zeros(20)
#     print(res)
#     for i in range(1,5):
#         vs_cameras_combinations = more_itertools.distinct_combinations(cameras, i)
#         for vs_cams in vs_cameras_combinations:
#             file_name = 'resolution/5k_vs_'+str(res)+'2569_VS_' + utilities.cam_list_tostring(vs_cams) + '.npy'
#             if os.path.exists(file_name):
#                 for j in range(20):
#                     score_res[j] += score(file_name, j)
#     working_array[int(res/1000-1)][:] = score_res

# cameras_combinations = [[1, 2, 7], [2, 5, 9]]
#
# working_array = np.zeros((2, 20))
#
# for id, cams in enumerate(cameras_combinations):
#     score_model = np.zeros(20)
#     print(id)
#     for i in range(1, 4):
#         vs_cameras_combinations = more_itertools.distinct_combinations(cameras, i)
#         for vs_cams in vs_cameras_combinations:
#             file_name = 'distance/distances_' + utilities.cam_list_tostring(cams) + '_VS_' + utilities.cam_list_tostring(
#                     vs_cams) + '.npy'
#             if os.path.exists(file_name):
#                 for j in range(20):
#                     score_model[j] += score(file_name, j)
#     score_models.append(('Model ' + utilities.cam_list_tostring(cams), score_model))
#     working_array[id][:] = score_model

number_view = 0
for j in range(1, 5):
    number_view += len(list(more_itertools.distinct_combinations(cameras, j)))

to_save_array = np.delete(working_array, -1, axis=1)/10

df = pd.DataFrame(to_save_array)
df.to_excel(excel_writer="pos_far_mean.xlsx")
print('end')
#Attention beaucoup de 0 dans les resolutions

