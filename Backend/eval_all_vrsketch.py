import torch
from torch.utils.data import Dataset
import numpy as np
from network.model_trainer import DiffusionModel
from utils.mesh_utils import voxel2mesh
from utils.utils import str2bool, ensure_directory
from utils.utils import num_to_groups

import os
from tqdm import tqdm

from utils.sketch_utils import _transform, create_random_pose, get_P_from_transform_matrix


def eval_model(
    occupancy_model,
    data_root: str,
    num_generate: int = 4,
    steps: int = 50,
    ema: bool = True,
    w: float = 1.0,
    view_information: int = 0,
    detail_view: bool = False,
    rotation: float() = 0.0,
    elevation: float() = 0.0,
    kernel_size: float = 2,
    truncated_time: float = 0.0,
):
    # model load
    #print(occupancy_model_path)
    #discrete_diffusion = DiffusionModel.load_from_checkpoint(occupancy_model_path).cuda()
    discrete_diffusion = occupancy_model
    # view load
    from utils.sketch_utils import Projection_List, Projection_List_zero
    if detail_view:
        projection_matrix = get_P_from_transform_matrix(
            create_random_pose(rotation=rotation, elevation=elevation))
    elif view_information == -1:
        projection_matrix = None
    else:
        if discrete_diffusion.elevation_zero:

            projection_matrix = Projection_List_zero[view_information]
        else:
            projection_matrix = Projection_List[view_information]

    # generate number
    batches = num_to_groups(num_generate, 32)
    generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model

    # proccess data and return generate data for dic (file_name: generate_np)
    voxel_dic = {}
    print(data_root)
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('npy'):
                vrsketch = os.path.join(root, file)
                print(vrsketch)
                if os.path.exists(vrsketch):
                    print(vrsketch)
                    voxel_lst = []
                    sketch_pcd = np.load(vrsketch)
                else:
                    continue
            
                sketch_pcd = normalize_pc(sketch_pcd)
                rgb = np.ones_like(sketch_pcd) * 0.4
                sketch_pcd = np.concatenate((sketch_pcd, rgb), axis=-1)

                for batch in batches:
                    with torch.no_grad():
                        import time
                        s = time.time()
                        res_tensor = generator.sample_with_sketch(sketch_c=sketch_pcd, batch_size=batch,
                                                                projection_matrix=projection_matrix, kernel_size=kernel_size,
                                                                steps=steps, truncated_index=truncated_time, sketch_w=w)
                        e = time.time()
                        print("time cost:", e-s)
                    for i in range(batch):
                        voxel = res_tensor[i].squeeze().cpu().numpy()
                        voxel_lst.append(voxel)
                voxel_dic[file.split('.')[0]] = voxel_lst
    return voxel_dic


def eval_model_from_path(
    occupancy_model_path: str,
    output_path: str,
    img_feat_list_path: str,
    data_root: str,
    num_generate: int = 4,
    steps: int = 50,
    ema: bool = True,
    w: float = 1.0,
    view_information: int = 0,
    detail_view: bool = False,
    rotation: float() = 0.0,
    elevation: float() = 0.0,
    kernel_size: float = 2,
    truncated_time: float = 0.0,
):

# 1. load occupancy-model and super-resolution-model and other option
    model_name, model_id = occupancy_model_path.split('/')[-2], occupancy_model_path.split('/')[-1]
    discrete_diffusion = DiffusionModel.load_from_checkpoint(occupancy_model_path).cuda()
    dataset_name = "chair_fanhua1"
    postfix = f"{model_name}_{model_id}_{dataset_name}_{ema}_{w}"
    root_dir = os.path.join(output_path, postfix)
    ensure_directory(root_dir)
    preprocess = _transform(224)
    device = "cuda"

    from utils.sketch_utils import Projection_List, Projection_List_zero
    if detail_view:
        projection_matrix  = get_P_from_transform_matrix(
            create_random_pose(rotation=rotation, elevation=elevation))
    elif view_information == -1:
        projection_matrix = None
    else:
        if discrete_diffusion.elevation_zero:

            projection_matrix = Projection_List_zero[view_information]
        else:
            projection_matrix = Projection_List[view_information]
    batches = num_to_groups(num_generate, 32)
    generator = discrete_diffusion.ema_model if ema else discrete_diffusion.model
    # generator = generator.eval()
# 2. dataset and dataloader: load image feature from shapenet list
    
    res_index = 0
    voxel_lst = []
    for ins_name in tqdm(os.listdir(data_root)):
        vrsketch_path = os.path.join(data_root, ins_name)
        try:
            sketch_pcd = np.load(vrsketch_path)
        except:
            print(vrsketch_path, ' not found !')
            continue
        sketch_pcd = normalize_pc(sketch_pcd)
        rgb = np.ones_like(sketch_pcd) * 0.4
        sketch_pcd = np.concatenate((sketch_pcd, rgb),axis=-1)
        save_name = os.path.join(root_dir, f'{ins_name[:-4]}')
        ensure_directory(save_name)
        index = 0
        for batch in batches:
            with torch.no_grad():
                import time
                s = time.time()
                res_tensor = generator.sample_with_sketch(sketch_c=sketch_pcd, batch_size=batch,
                                                        projection_matrix=projection_matrix, kernel_size=kernel_size,
                                                        steps=steps, truncated_index=truncated_time, sketch_w=w)
                e = time.time()
                print("time cost:",e-s)
            for i in range(batch):
                voxel = res_tensor[i].squeeze().cpu().numpy()
                voxel_lst.append(voxel)
                # np.save(os.path.join(save_name, f"{i}.npy"), voxel)
                # # print(voxel)
                # try:
                #     voxel[voxel > 0] = 1
                #     voxel[voxel < 0] = 0
                #     mesh = voxel2mesh(voxel)
                #     mesh.export(os.path.join(save_name, f"{i}.obj"))
                # except Exception as e:
                #     print(str(e))
                index += 1
                res_index += 1
    return voxel_lst

def normalize_pc(pc):
    # normalize pc to [-1, 1]
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc
               
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='generate something')
    parser.add_argument("--occupancy_model_path", type=str, required=True)
    parser.add_argument("--img_feat_list_path", type=str, required=False)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--num_generate", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--truncated_time", type=float, default=0.0)
    parser.add_argument("--data_class", type=str, default="chair")
    parser.add_argument("--text_w", type=float, default=1.0)
    parser.add_argument("--image_path", type=str, default="test.png")
    parser.add_argument("--image_name", type=str2bool, default=False)
    parser.add_argument("--sketch_w", type=float, default=1.0)
    parser.add_argument("--view_information", type=int, default=0)
    parser.add_argument("--detail_view", type=str2bool, default=False)
    parser.add_argument("--rotation", type=float, default=0.)
    parser.add_argument("--elevation", type=float, default=0.)
    parser.add_argument("--kernel_size", type=float, default=4.)
    parser.add_argument("--verbose", type=str2bool, default=False)
    parser.add_argument("--mode", type=str, default="eval_mode")
    parser.add_argument("--cls_mode", type=str, default="chair")
    parser.add_argument("--img_path", type=str, default="")
    parser.add_argument("--vrsketch_path", type=str, default="")

    args = parser.parse_args()
    ensure_directory(args.output_path)

    if args.mode == "eval_mode":
        voxel_lst = eval_model(occupancy_model_path=args.occupancy_model_path,
                                    img_feat_list_path=args.img_feat_list_path, data_root=args.data_root,
                                    output_path=args.output_path, ema=args.ema, steps=args.steps,
                                    num_generate=args.num_generate, truncated_time=args.truncated_time, w=args.sketch_w,
                                    view_information=args.view_information, kernel_size=args.kernel_size,
                                    detail_view=args.detail_view,
                                    rotation=args.rotation, elevation=args.elevation,
                                    cls_mode=args.cls_mode,img_path=args.img_path, vrsketch_path=args.vrsketch_path
                                    )
    elif args.mode == "eval_mode_from_path":
        voxel_lst = eval_model_from_path(occupancy_model_path=args.occupancy_model_path,
                                    img_feat_list_path=args.img_feat_list_path, data_root=args.data_root,
                                    output_path=args.output_path, ema=args.ema, steps=args.steps,
                                    num_generate=args.num_generate, truncated_time=args.truncated_time, w=args.sketch_w,
                                    view_information=args.view_information, kernel_size=args.kernel_size,
                                    detail_view=args.detail_view,
                                    rotation=args.rotation, elevation=args.elevation
                                    )