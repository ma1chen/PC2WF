import os
curr_dir = os.path.dirname(os.path.realpath(__file__))
import shutil
import numpy as np
from glob import glob
# from cal_prec_line import *

def filter_prob_vertex(vertex_pred, vertex_probs, line_pred, line_probs, prob_th=0.5):
    '''filter vertex with probability'''
    print("开始：通过概率值过滤顶点")
    print("检测到"+str(len(vertex_probs))+"个顶点")
    dropped_vertex_index = []
    for vertex_i in range(len(vertex_probs)):
        if vertex_probs[vertex_i] < prob_th:
            dropped_vertex_index.append(vertex_i)
    dropped_vertex_index = [i+1 for i in dropped_vertex_index]
    print("丢弃"+str(len(dropped_vertex_index))+"个顶点")
    keep_line_index = []
    for line_i in range(len(line_pred)):
        if (line_pred[line_i][0] not in dropped_vertex_index) and (line_pred[line_i][1] not in dropped_vertex_index):
            keep_line_index.append(line_i)
    line_pred = line_pred[keep_line_index]
    line_probs = line_probs[keep_line_index]
    return line_pred, line_probs

def filter_nms_vertex(vertex_pred, vertex_probs, line_pred, line_probs, nms_th=0.02):
    '''filter vertex with NMS'''
    print("开始：通过非极大值抑制过滤顶点")
    print("检测到"+str(len(vertex_probs))+"个顶点")
    print(f"vertex_pred shape: {vertex_pred.shape}")
    print(f"vertex_probs shape: {vertex_probs.shape}")

    # 确保 vertex_pred 是二维数组
    if vertex_pred.ndim == 1:
        vertex_pred = vertex_pred.reshape(-1, 1)
    elif vertex_pred.ndim != 2:
        raise ValueError("vertex_pred should be a 2D array with shape (n, 3) or (n, 2)")
    dropped_vertex_index = []
    for vertex_i in range(len(vertex_probs)):
        if vertex_i in dropped_vertex_index:
            continue
        dist_all = np.linalg.norm(vertex_pred - vertex_pred[vertex_i], axis=1)
        same_region_indexes = (dist_all < nms_th).nonzero()
        for same_region_i in same_region_indexes[0]:
            if same_region_i == vertex_i:
                continue
            if vertex_probs[same_region_i] <= vertex_probs[vertex_i]:
                dropped_vertex_index.append(same_region_i)
            else:
                dropped_vertex_index.append(vertex_i)

    dropped_vertex_index = list(set(dropped_vertex_index))  # 移除重复的索引
    print("丢弃"+str(len(dropped_vertex_index))+"个顶点")
    keep_line_index = []
    for line_i in range(len(line_pred)):
        if (line_pred[line_i][0] not in dropped_vertex_index) and (line_pred[line_i][1] not in dropped_vertex_index):
            keep_line_index.append(line_i)

    line_pred = line_pred[keep_line_index]
    line_probs = line_probs[keep_line_index]
    return line_pred, line_probs

def merge_vertex(vertex_pred, vertex_probs, merge_th=0.02):
    '''merge vertex that close to each other'''
    print("开始：合并距离小于"+str(merge_th)+"的顶点")
    print("检测到"+str(len(vertex_probs))+"个顶点")
    to_merge_index = [] # vertex that to be merged
    merge_to_index = [] # which vertex merge to
    for vertex_i in range(len(vertex_probs)):
        dist_all = np.linalg.norm(vertex_pred-vertex_pred[vertex_i], axis=1)
        same_region_indexes = (dist_all < merge_th).nonzero()
        for same_region_i in same_region_indexes[0]:
            if same_region_i == vertex_i:
                continue
            if vertex_probs[same_region_i] <= vertex_probs[vertex_i]:
                to_merge_index.append(same_region_i)
                merge_to_index.append(vertex_i)
            else:
                to_merge_index.append(vertex_i)
                merge_to_index.append(same_region_i)
    
    for merge_i in range(len(to_merge_index)):
        vertex_pred[to_merge_index[merge_i]] = vertex_pred[merge_to_index[merge_i]]
        vertex_probs[to_merge_index[merge_i]] = vertex_probs[merge_to_index[merge_i]]
    print("合并后剩余"+str(len(vertex_probs))+"个顶点")
    return vertex_pred, vertex_probs


def filter_prob_line(line_pred, line_probs, prob_th=0.5):
    '''filter line with probability'''
    print("开始：根据概率值过滤线段")
    print(f"line_pred shape: {line_pred.shape}")
    print(f"line_probs shape: {line_probs.shape}")
    # print(line_probs)
    # 确保 line_probs 是一维数组
    if line_probs.ndim != 1:
        raise ValueError("line_probs should be a 1D array")
    filter_line = []
    filter_probs = []
    for line_i in range(len(line_probs)):
        if line_probs[line_i] >= prob_th:
            filter_line.append(line_pred[line_i])
            filter_probs.append(line_probs[line_i])
    return np.array(filter_line), np.array(filter_probs)


def filter_short_line(vertex_pred, line_pred, line_probs, len_th=0.01):
    '''filter short lines'''
    print("开始：过滤长度小于"+str(len_th)+"的线段")
    filter_line = []
    filter_probs = []
    for line_i in range(len(line_probs)):
        l0, l1 = vertex_pred[line_pred[line_i][0]-1], vertex_pred[line_pred[line_i][1]-1]
        if np.linalg.norm(l0-l1) > len_th:
            filter_line.append(line_pred[line_i])
            filter_probs.append(line_probs[line_i])
    return np.array(filter_line), np.array(filter_probs)


def filter_nms_line(vertex_pred, line_pred, line_probs, nms_th=0.05):
    '''filter lines with nms, sum of two endpoints <= nms_th'''
    print("开始：通过非极大值抑制过滤线段")
    dropped_line_index = []
    line_pred = line_pred.tolist()
    for line_i in range(len(line_probs)):
        if line_i in dropped_line_index:
            continue
        dist_l0 = np.linalg.norm(vertex_pred-vertex_pred[line_pred[line_i][0]-1], axis=1)
        dist_l1 = np.linalg.norm(vertex_pred-vertex_pred[line_pred[line_i][1]-1], axis=1)
        same_region_indexes_0 = (dist_l0 < nms_th).nonzero()[0]
        same_region_indexes_1 = (dist_l1 < nms_th).nonzero()[0]
        for region_i_0 in same_region_indexes_0:
            for region_i_1 in same_region_indexes_1:
                if ([region_i_0+1, region_i_1+1] == line_pred[line_i]) or ([region_i_1+1, region_i_0+1] == line_pred[line_i]):
                    continue
                if (dist_l0[region_i_0]+dist_l1[region_i_1])>nms_th:
                    continue
                close_line_index = -1
                if ([region_i_0+1, region_i_1+1] in line_pred):
                    close_line_index = line_pred.index([region_i_0+1, region_i_1+1])
                elif ([region_i_1+1, region_i_0+1] in line_pred):
                    close_line_index = line_pred.index([region_i_1+1, region_i_0+1])
                if close_line_index != -1:
                    if line_probs[close_line_index] <= line_probs[line_i]:
                        dropped_line_index.append(close_line_index)
                    else:
                        dropped_line_index.append(line_i)

    keep_line_index = [i for i in range(len(line_pred)) if i not in dropped_line_index]
    filter_line = np.array(line_pred)[keep_line_index]
    filter_probs = line_probs[keep_line_index]
    return np.array(filter_line), np.array(filter_probs)


def remove_extra_vertex(vertex_pred, line_pred):
    vertex_pred = vertex_pred.tolist()
    new_vertex_pred = []
    new_line_pred = []
    for v_i, v in enumerate(vertex_pred):
        if v not in new_vertex_pred and ((v_i+1) in line_pred):
            new_vertex_pred.append(v)
    
    for line in line_pred:
        line0, line1 = new_vertex_pred.index(vertex_pred[line[0]-1])+1, new_vertex_pred.index(vertex_pred[line[1]-1])+1
        if ([line0, line1] not in new_line_pred) and ([line1, line0] not in new_line_pred):
            new_line_pred.append([line0, line1])
    return np.array(new_vertex_pred), np.array(new_line_pred)

# def merge_vertex(vertex_pred, vertex_probs, merge_th=0.02):
#     '''merge vertex that close to each other'''
#     to_merge_index = [] # vertex that to be merged
#     merge_to_index = [] # which vertex merge to
#     for vertex_i in range(len(vertex_probs)):
#         dist_all = np.linalg.norm(vertex_pred-vertex_pred[vertex_i], axis=1)
#         same_region_indexes = (dist_all < merge_th).nonzero()
#         for same_region_i in same_region_indexes[0]:
#             if same_region_i == vertex_i:
#                 continue
#             if vertex_probs[same_region_i] <= vertex_probs[vertex_i]:
#                 to_merge_index.append(same_region_i)
#                 merge_to_index.append(vertex_i)
#             else:
#                 to_merge_index.append(vertex_i)
#                 merge_to_index.append(same_region_i)
    
#     for merge_i in range(len(to_merge_index)):
#         vertex_pred[to_merge_index[merge_i]] = vertex_pred[merge_to_index[merge_i]]
#         vertex_probs[to_merge_index[merge_i]] = vertex_probs[merge_to_index[merge_i]]
#     return vertex_pred, vertex_probs

def line_to_obj(vertex_pred, line_pred, save_to_path):
    with open(save_to_path, 'w') as f:
        for v in vertex_pred:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for l in line_pred:
            f.write(f'l {l[0]} {l[1]}\n')

def vertex_to_xyz(vertex_pred,save_to_path):
    with open(save_to_path, 'w') as f:
        for v in vertex_pred:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
    

if __name__ == "__main__":

    sigma = 0.01
    clip = 0.01

    patch_size = 50
    total_precision = 0.0
    total_recall = 0.0
    total_line_pred = 0
    total_line_gt = 0
    total = 0

    save_to_dir = os.path.join(curr_dir, f'visualize_line/patch{patch_size}sigma{sigma}clip{clip}/')
    if not os.path.exists(save_to_dir):
        os.makedirs(save_to_dir)

    # vertex_pred_list = glob(os.path.join(curr_dir, f'run_test_result/patch{patch_size}sigma{sigma}clip{clip}/*_vertex.txt'))
    vertex_pred_list = glob(os.path.join(curr_dir, f'run_test_result/patch{patch_size}sigma{sigma}clip{clip}/*_vertex.txt'))
    for vertex_pred_f in vertex_pred_list:
        # load data
        vertex_pred = np.loadtxt(vertex_pred_f)
        vertex_probs = np.loadtxt(vertex_pred_f.replace('_vertex.txt', '_vprobs.txt'))
        line_pred = np.loadtxt(vertex_pred_f.replace('_vertex.txt', '_line.txt'), dtype=np.int32)
        line_probs = np.loadtxt(vertex_pred_f.replace('_vertex.txt', '_lprobs.txt'))
        if len(line_pred.shape) == 1:
            line_pred = np.expand_dims(line_pred, 0)
            line_probs = np.expand_dims(line_probs, 0)
        
        # post-processing
        # line_pred, line_probs = filter_prob_vertex(vertex_pred, vertex_probs, line_pred, line_probs, prob_th=0.85)
        print(line_pred)
        # line_pred, line_probs = filter_nms_vertex(vertex_pred, vertex_probs, line_pred, line_probs, nms_th=0.01)
        # vertex_pred, vertex_probs = merge_vertex(vertex_pred, vertex_probs, merge_th=0.04)
        line_pred, line_probs = filter_prob_line(line_pred, line_probs, prob_th=0.5)
        print("通过概率值过滤线段后，剩余"+str(len(line_pred))+"线段")
        line_pred, line_probs = filter_short_line(vertex_pred, line_pred, line_probs, len_th=0.03)
        line_pred, line_probs = filter_nms_line(vertex_pred, line_pred, line_probs, nms_th=0.03)

        vertex_pred, vertex_probs = merge_vertex(vertex_pred, vertex_probs, merge_th=0.03)

        vertex_pred, line_pred = remove_extra_vertex(vertex_pred, line_pred)
        print("合并顶点后，剩余"+str(len(vertex_pred))+"顶点")

        save_to_path = os.path.join(save_to_dir, vertex_pred_f.split('/')[-1].replace('_vertex.txt', '_pred.obj'))
        vertex_save_to_path = os.path.join(save_to_dir, vertex_pred_f.split('/')[-1].replace('_vertex.txt', '_postvertex_pred.xyz'))
        vertex_to_xyz(vertex_pred,vertex_save_to_path)
        line_to_obj(vertex_pred, line_pred, save_to_path)
