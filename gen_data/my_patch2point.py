import numpy as np

def load_point_cloud(file_path):
    """
    读取点云文件并返回点云数据
    :param file_path: 点云文件路径
    :return: 点云数据（N x 3的numpy数组）
    """
    return np.loadtxt(file_path)

def load_patch_indices(patch_file):
    """
    读取patch文件并返回索引列表
    :param patch_file: patch文件路径
    :return: patch索引列表
    """
    patches = []
    with open(patch_file, 'r') as f:
        for line in f:
            # 确保索引是整数
            indices = [int(float(x)) for x in line.strip().split()]
            patches.append(indices)
    return patches

def convert_indices_to_coordinates(point_cloud, patch_indices):
    """
    使用索引从点云数据中提取相应的坐标
    :param point_cloud: 点云数据（N x 3的numpy数组）
    :param patch_indices: patch索引列表
    :return: 提取的坐标列表
    """
    patch_coordinates = []
    for indices in patch_indices:
        coords = point_cloud[indices, :]
        patch_coordinates.append(coords)
    return patch_coordinates

def save_patch_coordinates(patch_coordinates, output_file):
    """
    将提取的坐标存储到新的点云文件中
    :param patch_coordinates: 提取的坐标列表
    :param output_file: 输出文件路径
    """
    with open(output_file, 'w') as f:
        for coords in patch_coordinates:
            for point in coords:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

if __name__ == "__main__":
    # 示例文件路径
    # 示例文件路径
    point_cloud_file = '/home/mc/proj/PC2WF/roof3d_data/patches_50_noise_sigma0.01clip0.01/train/000000.down'
    patch_file = '/home/mc/proj/PC2WF/visualize/run_test_result/patch50sigma0.01clip0.01/000000_batch_input_vertex.txt'
    output_file = '/home/mc/proj/PC2WF/visualize/run_test_result/patch50sigma0.01clip0.01/000000_predict_vertex_patch.txt'

    # 读取点云数据
    point_cloud = load_point_cloud(point_cloud_file)

    # 读取patch索引
    patch_indices = load_patch_indices(patch_file)

    # 将索引转换为坐标
    patch_coordinates = convert_indices_to_coordinates(point_cloud, patch_indices)

    # 保存新的点云文件
    save_patch_coordinates(patch_coordinates, output_file)


