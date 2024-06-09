import numpy as np

def vector_angle(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return np.pi  # 如果任一向量是零向量，返回180度
    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def point_line_distance(point, line_start, line_end):
    if np.all(line_start == line_end):
        return np.linalg.norm(point - line_start)
    return np.linalg.norm(np.cross(line_end - line_start, line_start - point)) / np.linalg.norm(line_end - line_start)

def remove_redundant_edges(vertices, edges, angle_threshold=np.pi / 180, distance_threshold=0.01):
    edge_vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    
    keep_edges = set()
    edge_set = set(map(tuple, map(tuple, edges)))
    
    for edge in edge_set:
        v1, v2 = vertices[edge[0]], vertices[edge[1]]
        redundant = False
        for other_edge in edge_set:
            if edge != other_edge:
                u1, u2 = vertices[other_edge[0]], vertices[other_edge[1]]
                if vector_angle(v1 - v2, u1 - u2) < angle_threshold:
                    if point_line_distance(u1, v1, v2) < distance_threshold and point_line_distance(u2, v1, v2) < distance_threshold:
                        redundant = True
                        if np.linalg.norm(v1 - v2) > np.linalg.norm(u1 - u2):
                            keep_edges.add(edge)
                        else:
                            keep_edges.add(other_edge)
        if not redundant:
            keep_edges.add(edge)
    
    return vertices, np.array(list(keep_edges))

def read_obj(file_path):
    vertices = []
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('l '):
                edges.append(list(map(int, line.strip().split()[1:])))
    vertices = np.array(vertices)
    edges = np.array(edges) - 1  # Convert to zero-based index
    return vertices, edges

def write_obj(file_path, vertices, edges):
    with open(file_path, 'w') as f:
        for vertex in vertices:
            f.write(f"v {' '.join(map(str, vertex))}\n")
        for edge in edges:
            f.write(f"l {' '.join(map(lambda x: str(x + 1), edge))}\n")  # Convert to one-based index

def main():
    input_file = '/home/mc/proj/PC2WF/visualize/visualize_line/patch50sigma0.01clip0.01/gen_points_pred.obj'
    output_file = '/home/mc/proj/PC2WF/visualize/visualize_line/patch50sigma0.01clip0.01/remove_edges_gen_points_pred.obj'
    
    vertices, edges = read_obj(input_file)
    vertices, edges = remove_redundant_edges(vertices, edges)
    write_obj(output_file, vertices, edges)

if __name__ == "__main__":
    main()
