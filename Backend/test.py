from eval_all_vrsketch import eval_model
from generate_super_resolution import generate_one_mesh
import argparse
from utils.utils import str2bool, ensure_directory
import os

mode = "eval_mode"
occupancy_model_path = "pretrain_model/epoch=499-v1.ckpt"
data_root = "VR_Sketch/CVPR11.26/step2_data"
ema = True
steps = 50
num_generate = 4
truncated_time = 0.0
sketch_w = 1.0
detail_view = False
rotation = 0.0
elevation = 0.0
view_information = 0
kernel_size = 4.0

generate_method = "generate_one_mesh"
model_path = "pretrain_model/epoch=499.ckpt"
output_path = "output_path"
level = 0.0


if mode == "eval_mode":
    voxel_lst = eval_model(occupancy_model_path=occupancy_model_path,
                           data_root=data_root, ema=ema, steps=steps,
                           num_generate=num_generate, truncated_time=truncated_time, w=sketch_w,
                           view_information=view_information, kernel_size=kernel_size,
                           detail_view=detail_view,
                           rotation=rotation, elevation=elevation,
                           )
else:
    raise NotImplementedError
if len(voxel_lst) > 0:
   print("step 1 sucess")
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
ensure_directory(output_path)

if generate_method == "generate_one_mesh":
    generate_one_mesh(model_path=model_path, npy_path=voxel_lst, output_path=output_path, steps=steps, level=level,
                      ema=ema, truncated_index=truncated_time)
else:
    raise NotImplementedError

print("step 1 sucess")