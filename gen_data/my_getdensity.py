import numpy as np
from scipy.spatial import cKDTree
from glob import glob
import os

def compute_point_density(pointcloud, radius=0.1):
    tree = cKDTree(pointcloud)
    densities = []
    for point in pointcloud:
        indices = tree.query_ball_point(point, radius)
        density = len(indices) / (4/3 * np.pi * radius**3)
        densities.append(density)
    return np.array(densities)

def adjust_threshold_based_on_density(pointcloud, base_threshold=0.03, radius=0.1):
    densities = compute_point_density(pointcloud, radius)
    mean_density = np.mean(densities)
    adjusted_threshold = base_threshold / mean_density
    return adjusted_threshold

def process_point_cloud(file_path, base_threshold=0.03, radius=0.1):
    data = np.loadtxt(file_path, usecols=(0, 1, 2))
    adjusted_threshold = adjust_threshold_based_on_density(data, base_threshold, radius)
    print(f"File: {file_path}, Adjusted Threshold: {adjusted_threshold}")
    return adjusted_threshold

def process_all_files(directories, extension='*.xyz', base_threshold=0.03, radius=0.1):
    for directory in directories:
        file_list = glob(os.path.join(directory, extension))
        for file_path in file_list:
            process_point_cloud(file_path, base_threshold, radius)

if __name__ == '__main__':
    point_cloud_directories = [
        '/home/mc/proj/PC2WF/building3d_data/clean/xyz/test',
        # '/home/mc/proj/PC2WF/building3d_data/clean/xyz/train'
    ]
    process_all_files(point_cloud_directories)
