import os
import numpy as np
from glob import glob

def process_point_cloud(file_path):
    # 读取点云数据，只保留前三维
    data = np.loadtxt(file_path, usecols=(0, 1, 2))
    
    # 保存处理后的数据覆盖源文件
    np.savetxt(file_path, data, fmt='%.6f')

def process_all_files(directories, extension='*.xyz'):
    for directory in directories:
        # 获取所有点云文件
        file_list = glob(os.path.join(directory, extension))
        
        # 处理每个文件
        for file_path in file_list:
            process_point_cloud(file_path)
            print(f"Processed file: {file_path}")

if __name__ == '__main__':
    # 设置点云数据目录列表
    point_cloud_directories = [
        '/home/mc/proj/PC2WF/building3d_mini/clean/xyz/test',
        '/home/mc/proj/PC2WF/building3d_mini/clean/xyz/train'
    ]
    
    # 处理所有点云文件
    process_all_files(point_cloud_directories)
