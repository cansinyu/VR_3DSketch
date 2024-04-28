import numpy as np
from sparsity_network.sparsity_trainer import Sparsity_DiffusionModel
from utils.utils import str2bool, ensure_directory
from utils.sparsity_utils import load_input
from utils.mesh_utils import process_sdf
import argparse
import os
from utils.utils import VIT_MODEL, png_fill_color
import tqdm
import torch 

def normalize_pc(pc):
    # normalize pc to [-1, 1]
    pc = pc - np.mean(pc, axis=0)
    if np.max(np.linalg.norm(pc, axis=1)) < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / np.max(np.linalg.norm(pc, axis=1))
    return pc

def generate_one_mesh(
    model_path: str,
    output_path: str = "./outputs",
    ema: bool = True,
    steps: int = 1000,
    truncated_index: float = 0,
    level: float = 0.0,
    npy_root_data_path: str = None,
    sketch_folder: str = "/public2/home/huyuanqi/data/VRSketch/3DV_VRSketches/point_cloud/aligned_sketch"
):
    discrete_diffusion = Sparsity_DiffusionModel.load_from_checkpoint(model_path).cuda()
    instances_path = npy_root_data_path   # ./Pascal_3D/car_imagenet
    instances_list = os.listdir(instances_path)
    for instances in tqdm.tqdm(instances_list):
        save_new_data_path = os.path.join(output_path, instances)
        if not os.path.exists(save_new_data_path):
            os.makedirs(save_new_data_path)
        npy_list_path = os.path.join(instances_path, instances) # ./Pascal_3D/car_imagenet/n02814533_1015
        file_list = os.listdir(os.path.join(npy_root_data_path, instances))
        npy_files = [file for file in file_list if file.endswith('.npy')]   # .npy list
        sketch_path = os.path.join(sketch_folder, instances+'.npy')
        sketch_pcd = normalize_pc(np.load(sketch_path))
        rgb = np.ones_like(sketch_pcd) * 0.4
        sketch_pcd = torch.from_numpy(np.concatenate((sketch_pcd, rgb),axis=-1))[None].cuda()
        for npy in npy_files:
            root_npy = os.path.join(npy_list_path, npy) # ./Pascal_3D/car_imagenet/n02814533_1015/0.npy
            res = discrete_diffusion.generate_results_from_single_voxel(low_res_voxel=load_input(root_npy), img_condition=sketch_pcd, ema=ema, use_ddim=False, steps=steps, truncated_index=truncated_index)
            name = npy.split(".")[0]
            mesh = process_sdf(res[0], level=level, normalize=True)
            np.savez(os.path.join(save_new_data_path,f"{name}_sdf.npz"), res[0])   # ./Pascal3D_stage2/car_imagenet/n02814533_1015
            mesh.export(os.path.join(save_new_data_path, f"{name}_sdf.obj"))


def generate_meshes(
    model_path: str,
    npy_folder: str,
    batch_size: int,
    output_path: str = "./outputs",
    ema: bool = True,
    steps: int = 1000,
    based_gt: bool = False,
    truncated_index: float = 0,
    level: float = 0.0,
    save_npy: bool = False,
    save_mesh: bool = True,
    start_index: int = 0,
    end_index: int = 100000,
):
    npy_name = npy_folder.split('/')[-1]
    model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
    discrete_diffusion = Sparsity_DiffusionModel.load_from_checkpoint(
        model_path).cuda()

    postfix = f"{model_name}_{model_id}_{ema}_ddpm_{steps}_{npy_name}_{truncated_index}"
    save_path = os.path.join(output_path, postfix)
    if based_gt:
        save_path += "_gt"
    ensure_directory(save_path)
    discrete_diffusion.generate_results_from_folder(
        folder=npy_folder, ema=ema,
        save_path=save_path, batch_size=batch_size, use_ddim=False, steps=steps,
        truncated_index=truncated_index, sort_npy=not based_gt, level=level,
        save_npy=save_npy, save_mesh=save_mesh, start_index=start_index, end_index=end_index)


def generate_meshes_from_img(
    model_path: str,
    npy_folder: str,
    batch_size: int,
    output_path: str = "./outputs",
    ema: bool = True,
    steps: int = 1000,
    based_gt: bool = False,
    truncated_index: float = 0,
    level: float = 0.0,
    save_npy: bool = False,
    save_mesh: bool = True,
    start_index: int = 0,
    end_index: int = 100000,
):
    npy_name = npy_folder.split('/')[-1]
    model_name, model_id = model_path.split('/')[-2], model_path.split('/')[-1]
    discrete_diffusion = Sparsity_DiffusionModel.load_from_checkpoint(
        model_path).cuda()

    postfix = f"{model_name}_{model_id}_{ema}_ddpm_{steps}_{npy_name}_{truncated_index}"
    save_path = os.path.join(output_path, postfix)
    if based_gt:
        save_path += "_gt"
    ensure_directory(save_path)
    discrete_diffusion.generate_results_from_img(
        folder=npy_folder, ema=ema,
        save_path=save_path, batch_size=batch_size, use_ddim=False, steps=steps,
        truncated_index=truncated_index, sort_npy=not based_gt, level=level,
        save_npy=save_npy, save_mesh=save_mesh, start_index=start_index, end_index=end_index)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='generate something')
    parser.add_argument("--generate_method", type=str, default='generate_one_mesh',
                        help="please choose :\n \
                       1. 'generate_one_mesh'\n \
                       2. 'generate_meshes' \
                       ")

    parser.add_argument("--model_path", type=str, default='results/shape_sur_skcond/last.ckpt')  
    parser.add_argument("--output_path", type=str, default="/public2/home/huyuanqi/project/3Dsketch2shape_proj/LAS-Diffusion-main/outputs/encoder_clip_fixedDM_4v_cl_new__epoch=1499.ckpt_test_True_1.3_mine")
    parser.add_argument("--npy_path", type=str, default='mask_chair_new')  # chair_imagenet  mask_car
    parser.add_argument("--ema", type=str2bool, default=True)
    parser.add_argument("--based_gt", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_npy", type=str2bool, default=False)
    parser.add_argument("--save_mesh", type=str2bool, default=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--truncated_time", type=float, default=0.0)
    parser.add_argument("--level", type=float, default=0.0)
    parser.add_argument("--npy_root_data_path", type=str, default="/public2/home/huyuanqi/project/3Dsketch2shape_proj/LAS-Diffusion-main/outputs/encoder_clip_fixedDM_4v_cl_new__epoch=1499.ckpt_test_True_1.3")

    args = parser.parse_args()
    method = (args.generate_method).lower()
    ensure_directory(args.output_path)
    if method == "generate_one_mesh":
        generate_one_mesh(model_path=args.model_path, output_path=args.output_path, steps=args.steps, level=args.level,
                          ema=args.ema, truncated_index=args.truncated_time, npy_root_data_path=args.npy_root_data_path)
    elif method == "generate_meshes":
        generate_meshes(model_path=args.model_path, npy_folder=args.npy_path, output_path=args.output_path, ema=args.ema, steps=args.steps,
                        batch_size=args.batch_size, based_gt=args.based_gt, truncated_index=args.truncated_time, level=args.level, save_npy=args.save_npy, save_mesh=args.save_mesh,
                        start_index=args.start_index, end_index=args.end_index)
    elif method == "generate_meshes_from_img":
        generate_meshes_from_img(model_path=args.model_path, npy_folder=args.npy_path, output_path=args.output_path, ema=args.ema, steps=args.steps,
                        batch_size=args.batch_size, based_gt=args.based_gt, truncated_index=args.truncated_time, level=args.level, save_npy=args.save_npy, save_mesh=args.save_mesh,
                        start_index=args.start_index, end_index=args.end_index)
    else:
        raise NotImplementedError
