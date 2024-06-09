import os
curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('..')
# sys.path.append('/home/mc/proj/PC2WF')
from train_end2end import patchNet, vertexNet, lineNet, PointcloudDataset
from tqdm import tqdm
import random
import shutil
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import MinkowskiEngine.utils as ME_utils
from model.resunet import ResUNetBN2C
import numpy as np
from itertools import combinations
from glob import glob


def predict(test_file_path, backbone_pth, patch_pth, vertex_pth, line_pth):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load backbone_net
    backbone_net = ResUNetBN2C(1, 32, normalize_feature=True, conv1_kernel_size=7, D=3)
    backbone_net.load_state_dict(torch.load(backbone_pth))
    backbone_net = backbone_net.to(device)
    backbone_net.eval()
    # load patch_net
    patch_net = patchNet()
    patch_net.load_state_dict(torch.load(patch_pth))
    patch_net = patch_net.to(device)
    patch_net.eval()
    # load vertex_net
    vertex_net = vertexNet()
    vertex_net.load_state_dict(torch.load(vertex_pth))
    vertex_net = vertex_net.to(device)
    vertex_net.eval()
    # load line_net
    line_net = lineNet()
    line_net.load_state_dict(torch.load(line_pth))
    line_net = line_net.to(device)
    line_net.eval()

    '''first, load test_data and feed into backbone_net'''
    pc_down = np.loadtxt(test_file_path, dtype=np.float32) # Ndx3
    feats = np.expand_dims(np.loadtxt(test_file_path.replace('.down', '.feats'), dtype=np.float32), 1) # Ndx1
    coords = np.loadtxt(test_file_path.replace('.down', '.coords'), dtype=np.float32) # Ndx3
    patch_index = np.loadtxt(test_file_path.replace('.down', '.patch_index'), dtype=np.int32) # Ndx3

    # stensor = ME.SparseTensor(torch.from_numpy(feats).float(), coords=torch.from_numpy(coords)).to(device)
    # 将 feats 和 coords 移动到device
    feats = torch.Tensor(feats).to(device)
    coords = torch.Tensor(coords).to(device)
    # 创建 CoordinateManager
    dimension_of_coords = coords.shape[1]
    coordinate_manager = ME.CoordinateManager(D=3)
    stensor = ME.SparseTensor(features=feats, coordinates=coords,coordinate_manager=coordinate_manager)
    stensor.C.to(device)
    stensor.F.to(device)
    # stensor = ME.SparseTensor(torch.from_numpy(feats).float(), coordinates=torch.from_numpy(coords),device=device)

    features = backbone_net(stensor).F

    '''second, feed into patch_net to find patches with vertex'''
    patch_features = []
    patch_coords = []
    pc_down = torch.from_numpy(pc_down)
    patch_index = torch.from_numpy(patch_index).long()
    if len(patch_index.shape) == 1:
        patch_index = patch_index.unsqueeze(-1)
    # 遍历 patch_index 中的每个索引。
    for index in patch_index:
        # 从 features 中提取对应的特征，并使用 unsqueeze(0) 增加一个维度后添加到 patch_features 列表。
        patch_features.append(features[index].unsqueeze(0))
        # 从 pc_down 中提取对应的坐标，并添加到 patch_coords 列表。
        curr_coord = pc_down[index.long()]
        patch_coords.append(curr_coord)
    
    # 将 patch_features 列表中的所有张量沿第一个维度连接起来，形成一个整体张量，并将其移动到 CUDA 设备上。
    batch_features = torch.cat(patch_features, 0).cuda()
    # 将 patch_coords 列表中的所有张量堆叠成一个张量，并将其移动到 CUDA 设备上。
    batch_coords = torch.stack(patch_coords, 0).cuda()
    # 将 batch_coords 和 batch_features 在第三维度上连接起来，形成 batch_input_patch。然后使用 transpose(1, 2) 将维度交换，使其符合 patch_net 的输入要求。
    batch_input_patch = torch.cat([batch_coords, batch_features], 2).transpose(1, 2)
    # 通过 patch_net 前向传播
    batch_output_patch = patch_net(batch_input_patch)
    
    # select pacthes with positive vertex
    # 对 batch_output_patch 进行 sigmoid 激活，得到补丁的预测概率 predicted_patch_index。
    predicted_patch_index = torch.sigmoid(batch_output_patch.squeeze())
    out_predicted_patch=[]
    # 准备输入vertex_net的数据
    # 初始化四个空列表，用于存储顶点相关的数据。
    batch_input_vertex = []
    batch_input_vertex_prob = []
    batch_coords_center_vertex = []
    batch_coords_lwh_vertex = []
    # 遍历 predicted_patch_index 中的每个预测概率
    for i, predicted_index in enumerate(predicted_patch_index):
      # 如果概率大于 0.85，则将对应的 batch_input_patch 和 predicted_index 添加到 batch_input_vertex 和 batch_input_vertex_prob 列表中。
      if predicted_index > 0.85:
        out_predicted_patch.append(patch_index[i])
        batch_input_vertex.append(batch_input_patch[i])
        batch_input_vertex_prob.append(predicted_index)
    

    # print("batch_input_vertex")
    # print(batch_input_vertex)
    out_batch_input_vertex=batch_input_vertex
    # 将 batch_input_vertex 列表堆叠成一个张量，以便输入到 vertex_net 中。
    batch_input_vertex = torch.stack(batch_input_vertex, 0)

    # 保存输入给 vertexNet 的数据
    # np.save(test_file_path.replace('.down', '_vertex_input.npy'), batch_input_vertex.detach().cpu().numpy())
    # np.save(test_file_path.replace('.down', '_vertex_input_prob.npy'), np.array(batch_input_vertex_prob))


    '''third, feed into vertex_net to produce new vertex'''
    # 将 batch_input_vertex 输入到 vertex_net 网络中，得到输出 batch_output_vertex，这是新的顶点坐标。
    batch_output_vertex = vertex_net(batch_input_vertex)
    # 将 batch_output_vertex 保存到 batch_output_vertex_coord，并转换为 NumPy 数组 predicted_vertex_list
    batch_output_vertex_coord = batch_output_vertex
    predicted_vertex_list = batch_output_vertex_coord.detach().cpu().numpy()

    # NMS to select vertex
    nms_threshhold = 0.01
    # 初始化空列表 dropped_vertex_index 用于存储被抑制的顶点索引
    dropped_vertex_index = []
    # 遍历 predicted_vertex_list 中的每个顶点：
    for i in range(len(predicted_vertex_list)):
        # 如果顶点索引在 dropped_vertex_index 中，跳过当前顶点。
        if i in dropped_vertex_index:
            continue
        # 计算当前顶点与其他顶点的距离 dist_all。
        dist_all = np.linalg.norm(predicted_vertex_list-predicted_vertex_list[i], axis=1)
        # 找到与当前顶点距离小于 nms_threshhold 的顶点索引 same_region_indexes。
        same_region_indexes = (dist_all < nms_threshhold).nonzero()
        # 对于每个与当前顶点距离小于阈值的顶点（保留距离相近的顶点中置信度最大的顶点）：
        for same_region_i in same_region_indexes[0]:
            if same_region_i == i:
                continue
            # 如果该顶点的概率小于等于当前顶点，将其索引添加到 dropped_vertex_index。
            if batch_input_vertex_prob[same_region_i] <= batch_input_vertex_prob[i]:
                dropped_vertex_index.append(same_region_i)
            # 否则，将当前顶点的索引添加到 dropped_vertex_index。
            else:
                dropped_vertex_index.append(i)
    # 选择未被抑制的顶点索引 selected_vertex_index。
    selected_vertex_index = [i for i in range(len(predicted_vertex_list)) if i not in dropped_vertex_index]
    # 使用 selected_vertex_index 更新 batch_output_vertex_coord，保留未被抑制的顶点。
    batch_output_vertex_coord = batch_output_vertex_coord[selected_vertex_index]
    
    # batch_input_vertex_prob = np.array(batch_input_vertex_prob)[selected_vertex_index]

    #以下为修改代码
    # 将 batch_input_vertex_prob 中的每个张量转换为 NumPy 数组，并存储在一个新的列表中
    converted_probs = [tensor.detach().cpu().numpy() for tensor in batch_input_vertex_prob]
    # 使用 selected_vertex_index 来选择相应的元素
    selected_probs = [converted_probs[i] for i in selected_vertex_index]    
    # 将 selected_probs 转换为 NumPy 数组
    batch_input_vertex_prob = np.array(selected_probs)


    predicted_vertex_list = batch_output_vertex_coord.detach().cpu().numpy()
    predicted_vertex_probs = np.array(batch_input_vertex_prob)

    # 初始化空列表 predicted_vertex_features 用于存储预测顶点的特征。
    predicted_vertex_features = []
    # 遍历 batch_output_vertex_coord 中的每个顶点坐标：
    for coord in batch_output_vertex_coord.detach().cpu():
        # 计算该顶点坐标与 pc_down 中所有点的距离，并找到距离最小的点索引 pred_vertex_index。
        pred_vertex_index = torch.argmin(torch.norm(pc_down - coord, dim=1))
        # 使用 pred_vertex_index 从 features 中提取对应的特征，并添加到 predicted_vertex_features 列表中。
        predicted_vertex_features.append(features[pred_vertex_index])


    '''forth, feed into line_net to predict lines'''
    # 设置每条线段中的点数 point_num_in_line 为 30。
    point_num_in_line = 30
    input_line_features = predicted_vertex_features
    pc_down = pc_down.to(device)
    # 初始化列表 batch_input_line, batch_index_line, 和 batch_index_dist 分别用于存储线段特征、线段索引和线段距离。
    batch_input_line = []
    batch_index_line = []
    batch_index_dist = []
    # 遍历每对顶点 e1 和 e2 以生成候选线段：
    for i1, e1 in enumerate(batch_output_vertex_coord):
        for i2, e2 in enumerate(batch_output_vertex_coord):
            # 果 i1 大于或等于 i2，跳过当前顶点对。
            if i1 >= i2:
                continue
            # 计算中点与 pc_down 中所有点的距离 mid_point_dist，如果最小距离大于 0.03，跳过当前顶点对。
            mid_point_dist = torch.min(torch.norm(pc_down - (e1 + e2) / 2.0, dim=1))
            if mid_point_dist >= 0.03:
                continue
            # 初始化线段特征 tmp_input_line，存储第一个顶点的特征。
            tmp_input_line = [input_line_features[i1]]
            tmp_input_dist = 0
            valid_line = True
            # 遍历线段中的中间点：
            for inter_point in range(1, point_num_in_line+1):
                # 计算中间点的坐标 inter_point_coord。
                inter_point_coord = (float(inter_point)/(point_num_in_line+1)*e1 + (1-float(inter_point)/(point_num_in_line+1))*e2)
                # 计算中间点与 pc_down 中所有点的距离 inter_point_dist，如果最小距离大于 0.03，跳过当前线段。
                inter_point_dist = torch.norm(pc_down - inter_point_coord, dim=1)
                if torch.min(inter_point_dist) >= 0.03:
                    valid_line = False
                    break
                # 累加最小距离到 tmp_input_dist。
                tmp_input_dist += torch.min(inter_point_dist).cpu().item()
                inter_point_index = torch.argmin(inter_point_dist)
                # 将最近的特征添加到 tmp_input_line。
                tmp_input_line.append(features[inter_point_index])
            if not valid_line:
                continue
            # 将第二个顶点的特征添加到 tmp_input_line。
            tmp_input_line.append(input_line_features[i2])
            # 将有效的线段特征、索引和距离分别添加到 batch_input_line, batch_index_line, 和 batch_index_dist。
            batch_input_line.append(torch.stack(tmp_input_line))
            batch_index_line.append([i1+1, i2+1])
            batch_index_dist.append(tmp_input_dist/point_num_in_line)
    
    # 将 batch_input_line 转换为张量并进行维度变换。
    batch_input_line = torch.stack(batch_input_line).transpose(1, 2)
    # 将转换后的 batch_input_line 输入到 line_net 网络中，得到输出 batch_output_line，这是线段的预测值。
    batch_output_line = line_net(batch_input_line)
    # 将 batch_output_line 通过 Sigmoid 函数转换为概率 predicted_line_index。
    predicted_line_index = torch.sigmoid(batch_output_line.squeeze())
    # 初始化列表 predicted_line_list 和 predicted_line_probs 分别用于存储预测的线段索引和概率。
    predicted_line_list = []
    predicted_line_probs = []
    if len(predicted_line_index.shape) == 0:
        predicted_line_index = predicted_line_index.unsqueeze(0)
    # 遍历 predicted_line_index 中的每个预测值：
    for i, predicted_index in enumerate(predicted_line_index):
        # 如果预测值大于 0.5，将对应的线段索引和概率添加到 predicted_line_list 和 predicted_line_probs。
        if predicted_index > 0.5:
            predicted_line_list.append(batch_index_line[i])
            predicted_line_probs.append(predicted_index)
    #添加代码
    predicted_line_probs = [tensor.detach().cpu().numpy() for tensor in predicted_line_probs]
    out_predicted_patch = [tensor.detach().cpu().numpy() for tensor in out_predicted_patch]
    return np.array(predicted_vertex_list), np.array(predicted_vertex_probs), np.array(predicted_line_list), np.array(predicted_line_probs),  np.array(out_predicted_patch)
    


if __name__ == "__main__":
    sigma = 0.01
    clip = 0.01
    patch_size = 50

    save_to_folder = os.path.join(curr_dir, 'run_test_result', f'patch{patch_size}sigma{sigma}clip{clip}')
    if not os.path.exists(save_to_folder):
        os.makedirs(save_to_folder)
    print(os.path.join(curr_dir, f'../roof3d_data/patches_{patch_size}_noise_sigma{sigma}clip{clip}/test/*.down'))
    # test_file_list = glob(os.path.join(curr_dir, f'../abc_data/patches_{patch_size}_noise_sigma{sigma}clip{clip}/test/*.down'))
    test_file_list = glob(os.path.join(curr_dir, f'../roof3d_data/patches_{patch_size}_noise_sigma{sigma}clip{clip}/test/*.down'))
    # test_file_list = glob(os.path.join(curr_dir, f'../abc_data/patches_{patch_size}_noise_sigma{sigma}clip{clip}/test/points.down'))
    test_file_list.sort()
    test_file_list.sort()

    for test_file in tqdm(test_file_list):
        if os.path.exists(os.path.join(save_to_folder, test_file.split('/')[-1].replace('.down', '_line.txt'))):
            continue
        
        predicted_vertex_list, predicted_vertex_probs, predicted_line_list, predicted_line_probs, batch_input_vertex = predict(
            test_file,
            os.path.join(curr_dir, f'../checkpoint_sigma{sigma}clip{clip}/backbone_patchSize{patch_size}_miniBatch512_nmsTh0.03_linePosTh0.01_lineNegTh0.01_lossweightP1.0V50.0L1.0_Val.pth'),
            os.path.join(curr_dir, f'../checkpoint_sigma{sigma}clip{clip}/patchnet_patchSize{patch_size}_miniBatch512_nmsTh0.03_linePosTh0.01_lineNegTh0.01_lossweightP1.0V50.0L1.0_Val.pth'),
            os.path.join(curr_dir, f'../checkpoint_sigma{sigma}clip{clip}/vertexnet_patchSize{patch_size}_miniBatch512_nmsTh0.03_linePosTh0.01_lineNegTh0.01_lossweightP1.0V50.0L1.0_Val.pth'),
            os.path.join(curr_dir, f'../checkpoint_sigma{sigma}clip{clip}/linenet_patchSize{patch_size}_miniBatch512_nmsTh0.03_linePosTh0.01_lineNegTh0.01_lossweightP1.0V50.0L1.0_Val.pth'),
        )

        # print("predicted_vertex_list")
        # print(predicted_vertex_list)
        np.savetxt(os.path.join(save_to_folder, test_file.split('/')[-1].replace('.down', '_vertex.txt')), predicted_vertex_list)
        np.savetxt(os.path.join(save_to_folder, test_file.split('/')[-1].replace('.down', '_vprobs.txt')), predicted_vertex_probs)
        np.savetxt(os.path.join(save_to_folder, test_file.split('/')[-1].replace('.down', '_line.txt')), predicted_line_list)
        np.savetxt(os.path.join(save_to_folder, test_file.split('/')[-1].replace('.down', '_lprobs.txt')), predicted_line_probs)
        np.savetxt(os.path.join(save_to_folder, test_file.split('/')[-1].replace('.down', '_batch_input_vertex.txt')), batch_input_vertex)