from glob import glob
import os
import numpy as np
import open3d as o3d
import meshplot as mp
import torch
from PIL.features import codecs
from tqdm import tqdm


def normalize_to_box(input):
    """
    normalize point cloud to unit bounding box
    center = (max - min)/2
    scale = max(abs(x))
    input: pc [N, P, dim] or [P, dim]
    output: pc, centroid, furthest_distance

    From https://github.com/yifita/pytorch_points
    """
    if len(input.shape) == 2:
        axis = 0
        P = input.shape[0]
        D = input.shape[1]
    elif len(input.shape) == 3:
        axis = 1
        P = input.shape[1]
        D = input.shape[2]
    else:
        raise ValueError()

    if isinstance(input, np.ndarray):
        maxP = np.amax(input, axis=axis, keepdims=True)
        minP = np.amin(input, axis=axis, keepdims=True)
        centroid = (maxP + minP) / 2
        input = input - centroid
        furthest_distance = np.amax(np.abs(input), axis=(axis, -1), keepdims=True)
        input = input / furthest_distance
    elif isinstance(input, torch.Tensor):
        maxP = torch.max(input, dim=axis, keepdim=True)[0]
        minP = torch.min(input, dim=axis, keepdim=True)[0]
        centroid = (maxP + minP) / 2
        input = input - centroid
        in_shape = list(input.shape[:axis]) + [P * D]
        furthest_distance = torch.max(torch.abs(input).reshape(in_shape), dim=axis, keepdim=True)[0]
        furthest_distance = furthest_distance.unsqueeze(-1)
        input = input / furthest_distance
    else:
        raise ValueError()

    return input, centroid, furthest_distance


# Save obj file
def get_edge_list(vertice_array):
    edge_list = []
    p_count = 0
    for point_list in vertice_array:
        p_count += 1
        for i in range(len(point_list) - 1):
            edge_list.append([p_count, p_count + 1])
            p_count += 1
    return edge_list


def save_obj_file(file_path, v, l, is_utf8=False):
    if is_utf8:
        f = codecs.open(file_path, 'w', 'utf-8')
    else:
        f = open(file_path, 'w')
    for item in v:
        f.write('v %.6f %.6f %.6f \n' % (item[0], item[1], item[2]))
    for item in l:
        f.write('l %s %s \n' % (item[0], item[1]))
    if len(l) >= 1:
        f.write('l %s %s \n' % (l[-1][0], l[-1][1]))
    f.close()
    # print("save path:", file_path)


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# name file
noise_list = ['1_sketch']

# 加载原始草图
# 从 raw_dir 加载原始草图，请将其替换为原始草图 obj 文件的保存目录。
def process_data(output_path):
    raw_dir = os.path.join(output_path, 'step1')

    obj_files = []
    txt_files = []
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.obj'):
                obj_files.append(os.path.join(root, file))
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    # print(f"Number of obj: {len(obj_files)}")
    # print(f"Number of txt: {len(txt_files)}")

    for i in tqdm(range(len(obj_files)), "Processing data"):
        obj_file = obj_files[i]
        # print("load obj name: ", obj_file)
        timestamp = txt_files[i]
        # print("load timestamp name: ", timestamp)

        time_array = np.loadtxt(timestamp, delimiter=' ')

        xyz = time_array[:, :3]

        norm, centroid, furthest_distance = normalize_to_box(xyz)

        # Filtering
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        nb_points = int(norm.shape[0] * 0.1)
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points,
                                            radius=0.4)  # ind is the index list of all remained points
        t = time_array[:, 3]
        begin = np.where(t == 0)[0]  # the start index of each stroke

        filtered_strokes = []
        for i, start in enumerate(begin):
            end = begin[i + 1] if i < len(begin) - 1 else t.shape[0] - 1
            middle = int((start + end) / 2)
            if middle in ind:
                filtered_strokes.append(time_array[start:end])

        save_path = obj_file[:obj_file.rfind('/')].replace('step1', 'step2')
        ensure_directory(save_path)

        vertice_list = [item for stroke in filtered_strokes for item in stroke[:, :3]]  # filtered_strokes[:, :3]
        edg = []
        for stroke in filtered_strokes:
            edg.append(stroke[:, :3])
        edge_list = get_edge_list(edg)

        new_obj = obj_file.replace('step1', 'step2')
        save_obj_file(new_obj, vertice_list, edge_list)

        # Save timestamp file
        new_txt = timestamp.replace('step1', 'step2')
        timestamp_list = [item for stroke in filtered_strokes for item in stroke]
        np.savetxt(new_txt, timestamp_list, fmt='%0.8f')

if __name__ == "__main__":
    root = r"C:\dataset\VR_Sketch\CVPR11.26"  # 设置根目录
    process_data(root)  # 调用 process_data 函数来处理数据