import point_cloud_utils as pcu
import numpy as np
import os
from multiprocessing import Pool

point_num = 15000


def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def interpolate(v0, v1, step):
    dv = v0 - v1
    length = np.linalg.norm(dv)
    length_list = np.arange(0, length, step)
    point_list = [v0]
    for l_i in length_list:
        point_list.append(v1 + dv / length * l_i)
    return point_list


def read_obj(model_path):
    objFile = open(model_path, 'r')
    vertexList = []
    lineList = []
    for line in objFile:
        split = line.split()
        # if blank line, skip
        if not len(split):
            continue
        if split[0] == "v":
            vertexList.append([float(split[1]), float(split[2]), float(split[3])])
        elif split[0] == "l":
            lineList.append(split[1:])
    objFile.close()
    return vertexList, lineList


def sample_pointcloud_edge(model_path):
    vertexList, lineList = read_obj(model_path)
    if len(vertexList) < 1 or len(lineList) < 2:
        return None
    sum_length = 0
    for edge in lineList:
        v0 = np.array(vertexList[int(edge[0]) - 1])
        v1 = np.array(vertexList[int(edge[1]) - 1])
        sum_length += np.linalg.norm(v0 - v1)
    step = sum_length / point_num

    point_list = []
    for edge in lineList:
        v0 = np.array(vertexList[int(edge[0]) - 1])
        v1 = np.array(vertexList[int(edge[1]) - 1])
        point_list.extend(interpolate(v0, v1, step))
    sample_index = np.random.choice(len(point_list), point_num, replace=False)
    new_point_list = np.array(point_list)[sample_index]
    return new_point_list


def sample_pointcloud_mesh(obj_path):
    off_v, off_f, off_n = pcu.read_obj(obj_path)
    if off_n.shape[0] != off_v.shape[0]:
        off_n = np.array([])
    v_dense, n_dense = pcu.sample_mesh_random(off_v, off_f, off_n, num_samples=point_num)
    return v_dense


def run_shapenet(work_info):
    model_file, data_type = work_info
    if os.path.exists(model_file):
        if data_type == 'sketch':
            point_list = sample_pointcloud_edge(model_file)
        else:
            point_list = sample_pointcloud_mesh(model_file)
        if point_list is None:
            print("Something wrong:", model_file)
        else:
            point_list = np.array(point_list, dtype='float32')
            save_path = model_file[:model_file.rfind('/')].replace('step2', 'step3')
            ensure_directory(save_path)
            np_file = model_file.replace('step2', 'step3').replace('.obj', '.npy')
            np.save(np_file, point_list)


def process_data(root):
    obj_dir = os.path.join(root, 'step2')

    model_files = []
    for root, dirs, files in os.walk(obj_dir):
        for file in files:
            if file.endswith('.obj'):
                model_files.append(os.path.join(root, file))

    data_type = 'sketch'

    work_info = [(path, data_type) for path in model_files]

    with Pool(16) as p:
        p.map(run_shapenet, work_info)


if __name__ == '__main__':
    root = r"C:\dataset\VR_Sketch\CVPR11.26"
    process_data(root)