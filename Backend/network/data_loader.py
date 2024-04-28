import torch
from pathlib import Path
import numpy as np
import os
from utils.shapenet_utils import snc_category_to_synth_id_all, snc_synth_id_to_category_all, TSDF_VALUE, snc_category_to_synth_id_5, snc_category_to_synth_id_13
import skimage.measure
from utils.sketch_utils import Projection_List, Projection_List_zero
from random import random, randint
from utils.utils import SKETCH_PER_VIEW, SKETCH_NUMBER, GLOBAL_INDEX
from utils.sketch_utils import create_random_pose, get_P_from_transform_matrix
from network import cam_read
import glob


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data

def read_obj_vertices(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertex_data = line.strip().split()[1:]  # 剔除行首的 'v ' 并拆分数据
                vertex = [float(coord) for coord in vertex_data]  # 将坐标转换为浮点数
                vertices.append(vertex)
    return vertices

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def normalize_pc(pc):
    # normalize pc to [-1, 1]
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc

class occupancy_field_Dataset(torch.utils.data.Dataset):
    def __init__(self, sdf_folder: str, sketch_folder: str,
                 data_class: str = "all", data_augmentation=False,
                 size: int = 64,
                 feature_drop_out: float = 0.1,
                 split_dataset: bool = False,
                 vit_global: bool = False,
                 vit_local: bool = True,
                 elevation_zero: bool = True,
                 detail_view: bool = False,
                 use_text_condition: bool = False,
                 use_sketch_condition: bool = True
                 ):
        super().__init__()

        self.multiplier = 128 // size
        self.sketch_paths = []
        self.sdf_paths = []
        filelist = os.path.join(sketch_folder, 'list', 'train_val.txt')
        sketch_folder = os.path.join(sketch_folder, 'point_cloud', 'aligned_sketch')
        
        with open(filelist) as fid:
            lines = fid.readlines()
        
        for i in range(len(lines)):
            search_pattern = f'{sketch_folder}/*{lines[i].rstrip()}*'
            matching_files = glob.glob(search_pattern, recursive=True)
            self.sketch_paths += matching_files
            
        for i in range(len(lines)):
            self.sdf_paths.append(os.path.join(sdf_folder, '03001627', lines[i].rstrip()+'.npy'))
            
    def __len__(self):
        return len(self.sdf_paths)

    def __getitem__(self, index):
        sketch_path = self.sketch_paths[index]
        sketch_pcd = normalize_pc(np.load(sketch_path))
        sketch_pcd = random_scale_point_cloud(sketch_pcd[None, ...])
        sketch_pcd = shift_point_cloud(sketch_pcd)
        sketch_pcd = rotate_perturbation_point_cloud(sketch_pcd)
        sketch_pcd = rotate_point_cloud(sketch_pcd)
        sketch_pcd = sketch_pcd.squeeze()
        
        rgb = np.ones_like(sketch_pcd) * 0.4
        sketch_pcd = np.concatenate((sketch_pcd, rgb),axis=-1)
        
        sdf_path = self.sdf_paths[index]
        sdf = np.load(sdf_path)
        
        res = {}
        occupancy_high = np.where(abs(sdf) < TSDF_VALUE, np.ones_like(
            sdf, dtype=np.float32), np.zeros_like(sdf, dtype=np.float32))
        occupancy_low = skimage.measure.block_reduce(
            occupancy_high, (self.multiplier, self.multiplier, self.multiplier), np.max)
        res["occupancy"] = np.expand_dims(2 * occupancy_low - 1, axis=0)
        res["sketch_pcd"] = sketch_pcd
        # with open('ds_train_val.txt', 'a') as file:
        #     content_to_append = f"{sketch_path.split('/')[-1]}\n"
        #     file.write(content_to_append)
        return res
        
        
class occupancy_field_Dataset_car(torch.utils.data.Dataset):
    def __init__(self, sdf_folder: str, sketch_folder: str,
                 data_class: str = "all", data_augmentation=False,
                 size: int = 64,
                 feature_drop_out: float = 0.1,
                 split_dataset: bool = False,
                 vit_global: bool = False,
                 vit_local: bool = True,
                 elevation_zero: bool = True,
                 detail_view: bool = False,
                 use_text_condition: bool = False,
                 use_sketch_condition: bool = True
                 ):
        super().__init__()
        print("-------------traning car-------------")
        self.multiplier = 128 // size
        self.sketch_paths = []
        self.sdf_paths = []
        filelist = os.path.join(sketch_folder, 'list', 'car_train.txt')
        sketch_folder = os.path.join(sketch_folder, 'car_pcd', 'NPY')
        
        with open(filelist) as fid:
            lines = fid.readlines()
        
        for i in range(len(lines)):
            search_pattern = f'{sketch_folder}/*{lines[i].rstrip()}*'
            matching_files = glob.glob(search_pattern, recursive=True)
            self.sketch_paths += matching_files
            
        for i in range(len(lines)):
            self.sdf_paths.append(os.path.join(sdf_folder, '02958343', lines[i].rstrip()+'.npy'))

    def __len__(self):
        return len(self.sdf_paths)

    def __getitem__(self, index):
        sketch_path = self.sketch_paths[index]
        sketch_pcd = normalize_pc(np.load(sketch_path))
        sketch_pcd = random_scale_point_cloud(sketch_pcd[None, ...])
        sketch_pcd = shift_point_cloud(sketch_pcd)
        sketch_pcd = rotate_perturbation_point_cloud(sketch_pcd)
        sketch_pcd = rotate_point_cloud(sketch_pcd)
        sketch_pcd = sketch_pcd.squeeze()
        
        rgb = np.ones_like(sketch_pcd) * 0.4
        sketch_pcd = np.concatenate((sketch_pcd, rgb),axis=-1)
        
        sdf_path = self.sdf_paths[index]
        sdf = np.load(sdf_path)
        
        res = {}
        occupancy_high = np.where(abs(sdf) < TSDF_VALUE, np.ones_like(
            sdf, dtype=np.float32), np.zeros_like(sdf, dtype=np.float32))
        occupancy_low = skimage.measure.block_reduce(
            occupancy_high, (self.multiplier, self.multiplier, self.multiplier), np.max)
        res["occupancy"] = np.expand_dims(2 * occupancy_low - 1, axis=0)
        res["sketch_pcd"] = sketch_pcd
        # with open('ds_train_val.txt', 'a') as file:
        #     content_to_append = f"{sketch_path.split('/')[-1]}\n"
        #     file.write(content_to_append)
        return res


class occupancy_field_Dataset_cl0(torch.utils.data.Dataset):
    def __init__(self, sdf_folder: str, sketch_folder: str,
                 data_class: str = "all", data_augmentation=False,
                 size: int = 64,
                 feature_drop_out: float = 0.1,
                 split_dataset: bool = False,
                 vit_global: bool = False,
                 vit_local: bool = True,
                 elevation_zero: bool = True,
                 detail_view: bool = False,
                 use_text_condition: bool = False,
                 use_sketch_condition: bool = True
                 ):
        super().__init__()

        self.multiplier = 128 // size
        self.sketch_paths1 = []
        self.sdf_paths1 = []
        self.sketch_paths2 = []
        self.sdf_paths2 = []
        self.sketch_paths3 = []
        self.sdf_paths3 = []
        self.sketch_paths4 = []
        self.sdf_paths4 = []
        self.iter_index = 0
        pooling_list1 = os.path.join(sketch_folder, 'list', 'cl_list', 'cl_lst0.txt')
        pooling_list2 = os.path.join(sketch_folder, 'list', 'cl_list', 'cl_lst1.txt')
        pooling_list3 = os.path.join(sketch_folder, 'list', 'cl_list', 'cl_lst2.txt')
        pooling_list4 = os.path.join(sketch_folder, 'list', 'cl_list', 'cl_lst3.txt')
        sketch_folder = os.path.join(sketch_folder, 'point_cloud', 'aligned_sketch')
        
        with open(pooling_list1) as fid:
            lines = fid.readlines()
            for i in range(len(lines)):
                search_pattern = f'{sketch_folder}/*{lines[i].rstrip()}*'
                matching_files = glob.glob(search_pattern, recursive=True)
                self.sketch_paths1 += matching_files
                self.sdf_paths1.append(os.path.join(sdf_folder, '03001627', lines[i].rstrip()))
        with open(pooling_list2) as fid:
            lines = fid.readlines()
            for i in range(len(lines)):
                search_pattern = f'{sketch_folder}/*{lines[i].rstrip()}*'
                matching_files = glob.glob(search_pattern, recursive=True)
                self.sketch_paths2 += matching_files
                self.sdf_paths2.append(os.path.join(sdf_folder, '03001627', lines[i].rstrip()))
        with open(pooling_list3) as fid:
            lines = fid.readlines()
            for i in range(len(lines)):
                search_pattern = f'{sketch_folder}/*{lines[i].rstrip()}*'
                matching_files = glob.glob(search_pattern, recursive=True)
                self.sketch_paths3 += matching_files
                self.sdf_paths3.append(os.path.join(sdf_folder, '03001627', lines[i].rstrip()))
        with open(pooling_list4) as fid:
            lines = fid.readlines()
            for i in range(len(lines)):
                search_pattern = f'{sketch_folder}/*{lines[i].rstrip()}*'
                matching_files = glob.glob(search_pattern, recursive=True)
                self.sketch_paths4 += matching_files
                self.sdf_paths4.append(os.path.join(sdf_folder, '03001627', lines[i].rstrip()))

    def __len__(self):
        return len(self.sdf_paths4)

    def __getitem__(self, index):
        # print(index)
        if index < 64:
            random_index = randint(0, 159)
            sketch_path = self.sketch_paths1[random_index]
            sdf_path = self.sdf_paths1[random_index]
        elif index < 128:
            random_index = randint(0, 303)
            sketch_path = self.sketch_paths2[random_index]
            sdf_path = self.sdf_paths2[random_index]
        elif index < 192:
            random_index = randint(0, 578)
            sketch_path = self.sketch_paths3[random_index]
            sdf_path = self.sdf_paths3[random_index]
        else:
            random_index = randint(0, 802)
            sketch_path = self.sketch_paths4[random_index]
            sdf_path = self.sdf_paths4[random_index]
            
        sketch_pcd = normalize_pc(np.load(sketch_path))
        sketch_pcd = random_scale_point_cloud(sketch_pcd[None, ...])
        sketch_pcd = shift_point_cloud(sketch_pcd)
        sketch_pcd = rotate_perturbation_point_cloud(sketch_pcd)
        sketch_pcd = rotate_point_cloud(sketch_pcd)
        sketch_pcd = sketch_pcd.squeeze()
        
        rgb = np.ones_like(sketch_pcd) * 0.4
        sketch_pcd = np.concatenate((sketch_pcd, rgb),axis=-1)
        
        sdf = np.load(sdf_path)
        
        res = {}
        occupancy_high = np.where(abs(sdf) < TSDF_VALUE, np.ones_like(
            sdf, dtype=np.float32), np.zeros_like(sdf, dtype=np.float32))
        occupancy_low = skimage.measure.block_reduce(
            occupancy_high, (self.multiplier, self.multiplier, self.multiplier), np.max)
        res["occupancy"] = np.expand_dims(2 * occupancy_low - 1, axis=0)
        res["sketch_pcd"] = sketch_pcd

        return res    

class occupancy_field_Dataset_for_prior(torch.utils.data.Dataset):
    def __init__(self, sdf_folder: str, sketch_folder: str,
                 data_class: str = "all", data_augmentation=False,
                 size: int = 64,
                 feature_drop_out: float = 0.1,
                 split_dataset: bool = False,
                 vit_global: bool = False,
                 vit_local: bool = True,
                 elevation_zero: bool = True,
                 detail_view: bool = False,
                 use_text_condition: bool = False,
                 use_sketch_condition: bool = True
                 ):
        super().__init__()
        self.feature_drop_out = feature_drop_out
        self.multiplier = 128 // size
        self.sketch_paths = []
        self.sdf_paths = []
        filelist = os.path.join(sketch_folder, 'train_.lst')
        self.feature_folder = os.path.join(sketch_folder, "feature_clip_normalize")
        self.white_image_feature = np.load(os.path.join(
                self.feature_folder, "white_image_feature.npy"))
        
        with open(filelist) as fid:
            lines = fid.readlines()
        for i in range(len(lines)):
            self.sdf_paths.append(os.path.join(sdf_folder, lines[i].rstrip()+'.npy'))
        

    def __len__(self):
        return len(self.sdf_paths)

    def __getitem__(self, index):      
        sdf_path = self.sdf_paths[index]
        sdf = np.load(sdf_path)

        model_name = str(sdf_path).split(".")[0]
        data_id = model_name.split("/")[-2]
        model_name = os.path.join(model_name.split(
            "/")[-2], model_name.split("/")[-1])  
        
        sketch_view_index = np.random.randint(0, 49)
        try:
            image_feature = np.load(os.path.join(
                self.feature_folder, model_name, 'easy', f"{sketch_view_index:02d}.npy"))[None]
            # print(os.path.join(
            #     self.feature_folder, model_name, 'easy', f"{sketch_view_index:02d}.npy"))
        except Exception as e:
            print(str(e))
            image_feature = self.white_image_feature[None]
        if random() < self.feature_drop_out:
            image_feature = self.white_image_feature[None]
        res = {}
        image_feature = image_feature
        res["sketch_pcd"] = image_feature
        occupancy_high = np.where(abs(sdf) < TSDF_VALUE, np.ones_like(
            sdf, dtype=np.float32), np.zeros_like(sdf, dtype=np.float32))
        occupancy_low = skimage.measure.block_reduce(
            occupancy_high, (self.multiplier, self.multiplier, self.multiplier), np.max)
        res["occupancy"] = np.expand_dims(2 * occupancy_low - 1, axis=0)

        return res
    
        
class occupancy_field_Dataset_4cls_50num(torch.utils.data.Dataset):
    def __init__(self, sdf_folder: str, sketch_folder: str,
                 data_class: str = "all", data_augmentation=False,
                 size: int = 64,
                 feature_drop_out: float = 0.1,
                 split_dataset: bool = False,
                 vit_global: bool = False,
                 vit_local: bool = True,
                 elevation_zero: bool = True,
                 detail_view: bool = False,
                 use_text_condition: bool = False,
                 use_sketch_condition: bool = True
                 ):
        super().__init__()
        print("-------------4cls_50num-------------")
        self.multiplier = 128 // size
        self.sketch_paths = []
        self.sdf_paths = []
            
        filelist_car = "/public2/home/huyuanqi/data/VRSketch/3DV_VRSketches/list/4cls_50num/car_50.txt"    
        filelist_chair = "/public2/home/huyuanqi/data/VRSketch/3DV_VRSketches/list/4cls_50num/chair_50.txt"    
        filelist_sofa = "/public2/home/huyuanqi/data/VRSketch/3DV_VRSketches/list/4cls_50num/sofa_50.txt"    
        filelist_airplane = "/public2/home/huyuanqi/data/VRSketch/3DV_VRSketches/list/4cls_50num/airplane_50.txt" 
           
        car_sketch_folder = "/public2/home/huyuanqi/data/VRSketch/3DV_VRSketches/car_pcd/NPY"
        chair_sketch_folder = "/public2/home/huyuanqi/data/VRSketch/3DV_VRSketches/point_cloud/aligned_sketch"
        sofa_sketch_folder = "/public2/home/huyuanqi/data/VRSketch/3DV_VRSketches/sofa/NPY"
        airplane_sketch_folder = "/public2/home/huyuanqi/data/VRSketch/3DV_VRSketches/airplane/NPY"
        
        with open(filelist_chair) as fid:
            lines = fid.readlines()
            for i in range(len(lines)):
                search_pattern = f'{chair_sketch_folder}/*{lines[i].rstrip()}*'
                matching_files = glob.glob(search_pattern, recursive=True)
                if len(matching_files) == 0:
                    print(lines[i])
                    continue
                self.sketch_paths += matching_files
                self.sdf_paths.append(os.path.join(sdf_folder, '03001627', lines[i].rstrip()+'.npy'))
        
        with open(filelist_car) as fid:
            lines = fid.readlines()
            for i in range(len(lines)):
                search_pattern = f'{car_sketch_folder}/*{lines[i].rstrip()}*'
                matching_files = glob.glob(search_pattern, recursive=True)
                if len(matching_files) == 0:
                    print(lines[i])
                    continue
                self.sketch_paths += matching_files
                self.sdf_paths.append(os.path.join(sdf_folder, '02958343', lines[i].rstrip()+'.npy'))
        
        with open(filelist_sofa) as fid:
            lines = fid.readlines()
            for i in range(len(lines)):
                search_pattern = f'{sofa_sketch_folder}/*{lines[i].rstrip()}*'
                matching_files = glob.glob(search_pattern, recursive=True)
                if len(matching_files) == 0:
                    print(lines[i])
                    continue
                self.sketch_paths += matching_files
                self.sdf_paths.append(os.path.join(sdf_folder, '04256520', lines[i].rstrip()+'.npy'))
                
        with open(filelist_airplane) as fid:
            lines = fid.readlines()
            for i in range(len(lines)):
                search_pattern = f'{airplane_sketch_folder}/*{lines[i].rstrip()}*'
                matching_files = glob.glob(search_pattern, recursive=True)
                if len(matching_files) == 0:
                    print(lines[i])
                    continue
                self.sketch_paths += matching_files
                self.sdf_paths.append(os.path.join(sdf_folder, '02691156', lines[i].rstrip()+'.npy'))
                

    def __len__(self):
        return len(self.sdf_paths)

    def __getitem__(self, index):
        sketch_path = self.sketch_paths[index]
        sketch_pcd = normalize_pc(np.load(sketch_path))
        sketch_pcd = random_scale_point_cloud(sketch_pcd[None, ...])
        sketch_pcd = shift_point_cloud(sketch_pcd)
        sketch_pcd = rotate_perturbation_point_cloud(sketch_pcd)
        sketch_pcd = rotate_point_cloud(sketch_pcd)
        sketch_pcd = sketch_pcd.squeeze()
        
        rgb = np.ones_like(sketch_pcd) * 0.4
        sketch_pcd = np.concatenate((sketch_pcd, rgb),axis=-1)
        
        sdf_path = self.sdf_paths[index]
        sdf = np.load(sdf_path)
        
        res = {}
        occupancy_high = np.where(abs(sdf) < TSDF_VALUE, np.ones_like(
            sdf, dtype=np.float32), np.zeros_like(sdf, dtype=np.float32))
        occupancy_low = skimage.measure.block_reduce(
            occupancy_high, (self.multiplier, self.multiplier, self.multiplier), np.max)
        res["occupancy"] = np.expand_dims(2 * occupancy_low - 1, axis=0)
        res["sketch_pcd"] = sketch_pcd
        # with open('ds_train_val.txt', 'a') as file:
        #     content_to_append = f"{sketch_path.split('/')[-1]}\n"
        #     file.write(content_to_append)
        return res
    
       
class occupancy_field_Dataset_7cls_200num(torch.utils.data.Dataset):
    def __init__(self, sdf_folder: str, sketch_folder: str,
                 data_class: str = "all", data_augmentation=False,
                 size: int = 64,
                 feature_drop_out: float = 0.1,
                 split_dataset: bool = False,
                 vit_global: bool = False,
                 vit_local: bool = True,
                 elevation_zero: bool = True,
                 detail_view: bool = False,
                 use_text_condition: bool = False,
                 use_sketch_condition: bool = True
                 ):
        super().__init__()
        print("-------------7cls_200num-------------")
        self.multiplier = 128 // size
        self.sketch_paths = []
        self.sdf_paths = []
            
        filelist = "/public2/home/huyuanqi/data/VRSketch/VR_draw/train.txt"    
           
        sketch_root = "/public2/home/huyuanqi/data/VRSketch/VR_draw/train/CVPR"
        cls_id_map = {
            "car": "02958343",
            "airplane": "02691156",
            "bench": "02828884",
            "display": "03211117",
            "sofa": "04256520",
            "table": "04379243",
            "watercraft": "04530566",
        }
        
        with open(filelist) as fid:
            lines = fid.readlines()
            for i in range(len(lines)):
                cls_name = lines[i].split("/")[0]
                ins_name = lines[i].split("/")[1]
                if cls_name == "car":
                    car_root = "/public2/home/huyuanqi/data/VRSketch/3DV_VRSketches/car_pcd/NPY"
                    search_pattern = f'{car_root}/*{ins_name.rstrip()}*'
                else:
                    search_pattern = f'{os.path.join(sketch_root, cls_name, "NPY")}/*{ins_name.rstrip()}*'
                matching_files = glob.glob(search_pattern, recursive=True)
                if len(matching_files) == 0:
                    print(lines[i])
                    continue
                self.sketch_paths += matching_files
                self.sdf_paths.append(os.path.join(sdf_folder, cls_id_map[cls_name], ins_name.rstrip()+'.npy'))

    def __len__(self):
        return len(self.sdf_paths)

    def __getitem__(self, index):
        sketch_path = self.sketch_paths[index]
        sketch_pcd = normalize_pc(np.load(sketch_path))
        sketch_pcd = random_scale_point_cloud(sketch_pcd[None, ...])
        sketch_pcd = shift_point_cloud(sketch_pcd)
        sketch_pcd = rotate_perturbation_point_cloud(sketch_pcd)
        sketch_pcd = rotate_point_cloud(sketch_pcd)
        sketch_pcd = sketch_pcd.squeeze()
        
        rgb = np.ones_like(sketch_pcd) * 0.4
        sketch_pcd = np.concatenate((sketch_pcd, rgb),axis=-1)
        
        sdf_path = self.sdf_paths[index]
        sdf = np.load(sdf_path)
        
        res = {}
        occupancy_high = np.where(abs(sdf) < TSDF_VALUE, np.ones_like(
            sdf, dtype=np.float32), np.zeros_like(sdf, dtype=np.float32))
        occupancy_low = skimage.measure.block_reduce(
            occupancy_high, (self.multiplier, self.multiplier, self.multiplier), np.max)
        res["occupancy"] = np.expand_dims(2 * occupancy_low - 1, axis=0)
        res["sketch_pcd"] = sketch_pcd
        # with open('ds_train_val.txt', 'a') as file:
        #     content_to_append = f"{sketch_path.split('/')[-1]}\n"
        #     file.write(content_to_append)
        return res
    
