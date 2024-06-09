import collections
import json
import multiprocessing
import os
import shutil
import sys
import time
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool
from random import shuffle
from subprocess import Popen
import h5py
import numpy as np
from tqdm import tqdm
import MinkowskiEngine.utils as ME_utils

# 从点云生成正负样本，用于训练深度学习模型来检测点云中的线条。
# 使用点云中的边缘信息（ground truth edge）来生成正样本（实际存在的边缘）和负样本（点云中不存在的边缘）
def gen_line(point_path, edge_gt, index=0, rotate_angle=None, random_rotate=True, patch_size=20, clean_noise='noise', sigma=0.01, clip=0.02, train_val_test='test'):
    train_val_test = train_val_test
    if random_rotate:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../simulateRoof_data/patches_{}_noise_sigma{}clip{}_rotate/{}'.format(patch_size, sigma, clip, train_val_test))
    else:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../simulateRoof_data/patches_{}_noise_sigma{}clip{}/{}'.format(patch_size, sigma, clip, train_val_test))

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    
    if index is None:
        save_root_path = os.path.join(base_dir, point_path.split('/')[-1])
    else:
        save_root_path = os.path.join(base_dir, point_path.split('/')[-1]).replace('.xyz', '_{}.xyz'.format(index))

    point_down_path = save_root_path.replace('.xyz', '.down')

    '''begin: down_sample'''
    pointcloud_down = np.loadtxt(point_down_path)
    '''end: down_sample'''


    '''begin: line samples'''
    edge_points = edge_gt
    # 旋转边缘点处理，如果提供了旋转角度 rotate_angle，则创建旋转矩阵并将其应用于边缘点的坐标。
    if rotate_angle is not None:
        angles = rotate_angle
        Rx = np.array([[1, 0, 0],
                        [0, np.cos(angles[0]), -np.sin(angles[0])],
                        [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                        [0, 1, 0],
                        [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                        [np.sin(angles[2]), np.cos(angles[2]), 0],
                        [0, 0, 1]])
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

        edge_points[:,:3] = np.dot(edge_points[:,:3], rotation_matrix)
        edge_points[:,3:] = np.dot(edge_points[:,3:], rotation_matrix)

    # point_num_in_line 指定了每条边中间要插值生成的点的数量（30个）。
    point_num_in_line = 30

    # positive samples: all edges
    # 通过在边缘点之间插值生成中间点来创建正样本，确保所有中间点都在阈值距离（0.03）以内。
    # positive_edge_index 用于存储正样本点的索引。
    positive_edge_index = []
    # positive_edge_end_point_index 用于存储每条边的起点和终点索引。
    positive_edge_end_point_index = []
    # 遍历 edge_points 中的每一条边（并为每条边生成正样本）：
    for edge in edge_points:
        e1 = np.argmin(np.linalg.norm(pointcloud_down-edge[:3], axis=1))
        e2 = np.argmin(np.linalg.norm(pointcloud_down-edge[3:], axis=1))
        # mid_point_index = np.argmin(np.linalg.norm(pointcloud_down - (pointcloud_down[e1] + pointcloud_down[e2]) / 2.0, axis=1))
        inter_point_list_positive = []
        valid_line = True
        for inter_point in range(1, point_num_in_line+1):
            # inter_point_dist 是中间点到点云中所有点的距离。
            inter_point_dist = np.linalg.norm(pointcloud_down - ((float(inter_point)/(point_num_in_line+1))*pointcloud_down[e1] + (1-float(inter_point)/(point_num_in_line+1))*pointcloud_down[e2]), axis=1)
            # 如果最小距离大于 0.03，标记为无效线条，并退出循环。
            if np.min(inter_point_dist) > 0.030:
                valid_line = False
                break
            # 找到距离最小的中间点索引，并添加到 inter_point_list_positive。
            inter_point_index = np.argmin(inter_point_dist)
            inter_point_list_positive.append(inter_point_index)
        if not valid_line:
            continue
        # 如果边是有效的，将边的起点、中间点、终点和标记（1表示正样本）添加到 positive_edge_index。
        positive_edge_index.append(e1)
        positive_edge_index.extend(inter_point_list_positive)
        positive_edge_index.append(e2)
        positive_edge_index.append(1)
        # 将起点和终点索引添加到 positive_edge_end_point_index。
        positive_edge_end_point_index.append([e1, e2])
        # positive_edge_index.append([e1, e2, mid_point_index, 1])
    # 将 positive_edge_index 转换为 NumPy 数组，并重新调整形状，使每一行包含一条边的信息（包括起点、中间点、终点和标记）。       
    positive_edge_index = np.array(positive_edge_index)
    positive_edge_index = np.reshape(positive_edge_index, (-1, point_num_in_line+3))

    # negative samples: vertices in the same face but no edge
    # negative_edge_index 用于存储负样本点的索引。
    # negative_edge_end_point_index 用于存储每条负样本边的起点和终点索引。
    negative_edge_index = []
    negative_edge_end_point_index = []
    # 对于每条正样本边，提取其起点 e1 和终点 e2。
    for edge in positive_edge_index:
        e1, e2 = edge[0], edge[-2]
        # e2_edges_0 = positive_edge_index[positive_edge_index[:,0] == e2][:,1]
        # 通过 positive_edge_index[:,0] == e2 筛选出所有以 e2 作为起点的边。
        # 从这些边中提取它们的终点（即倒数第二列 [-2]），这些终点代表了与 e2 相连的另一端点。
        e2_edges_0 = positive_edge_index[positive_edge_index[:,0] == e2][:,-2] # diagonal
        # 通过 positive_edge_index[:,0] == e2 筛选出所有以 e2 作为起点的边。
        # 从这些边中提取它们的第五列元素 [5]，这些元素可能代表了与 e2 相连的另一种方式的点（具体取决于数据格式和上下文）。
        e2_edges_1 = positive_edge_index[positive_edge_index[:,0] == e2][:,5] # offset
        # 将上面两步得到的结果 e2_edges_0 和 e2_edges_1 合并，得到 e2_edges。e2_edges 包含了所有与当前边 edge 的终点 e2 相连的其他边的终点。
        e2_edges = np.concatenate((e2_edges_0, e2_edges_1))
        for e2_v in e2_edges:
            if ([e1, e2_v] in negative_edge_end_point_index) or ([e2_v, e1] in negative_edge_end_point_index):
                continue
            if ([e1, e2_v] in positive_edge_end_point_index) or ([e2_v, e1] in positive_edge_end_point_index) or (e1 == e2_v):
                continue
            if np.linalg.norm(e2 - e2_v) <= 0.03:
                continue
            mid_point_dist = np.min(np.linalg.norm(pointcloud_down - (pointcloud_down[e1] + pointcloud_down[e2_v]) / 2.0, axis=1))
            if mid_point_dist >= 0.030:
                continue
            inter_point_list_negative = []
            valid_line = True
            for inter_point in range(1, point_num_in_line+1):
                inter_point_dist = np.min(np.linalg.norm(pointcloud_down - (pointcloud_down[e1] + pointcloud_down[e2_v]) / 2.0, axis=1))
                if inter_point_dist >= 0.030:
                    valid_line = False
                    break
                inter_point_index = np.argmin(np.linalg.norm(pointcloud_down - (float(inter_point)/(point_num_in_line+1)*pointcloud_down[e1] + (1-float(inter_point)/(point_num_in_line+1))*pointcloud_down[e2_v]), axis=1))
                inter_point_list_negative.append(inter_point_index)
            if not valid_line:
                continue
            negative_edge_index.append(e1)
            negative_edge_index.extend(inter_point_list_negative)
            negative_edge_index.append(e2_v)
            negative_edge_index.append(0)
            negative_edge_end_point_index.append([e1, e2_v])
    negative_edge_index = np.array(negative_edge_index)
    negative_edge_index = np.reshape(negative_edge_index, (-1, point_num_in_line+3))


    np.savetxt(save_root_path.replace('.xyz', '.mini_line'), np.concatenate((positive_edge_index, negative_edge_index)))

    '''end: line samples'''

