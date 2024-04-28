import os
import numpy as np
from shutil import copyfile

train_list = "/public2/home/huyuanqi/data/VRSketch/VR_draw/train.txt"
test_list = "/public2/home/huyuanqi/data/VRSketch/VR_draw/test.txt"
data_root = "/public2/home/huyuanqi/data/ShapeNetCore.v1"
save_path = "/public2/home/huyuanqi/data/VRSketch/deepsdf_data"

cls_id_map = {
            "car": "02958343",
            "airplane": "02691156",
            "bench": "02828884",
            "display": "03211117",
            "sofa": "04256520",
            "table": "04379243",
            "watercraft": "04530566",
        }
train_ = []
test_ = []
with open(train_list) as f:
    lines = f.readlines()
    for i in range(len(lines)):
        cls_name = lines[i].split("/")[0]
        ins_name = lines[i].split("/")[1].strip()
        obj_path = os.path.join(cls_id_map[cls_name], ins_name,"model.obj")
        try:
            copyfile(os.path.join(data_root,obj_path), os.path.join(save_path,obj_path))
        except:
            os.makedirs(os.path.join(save_path,cls_id_map[cls_name], ins_name))
            copyfile(os.path.join(data_root,obj_path), os.path.join(save_path,obj_path))
        train_.append(obj_path)
with open(test_list) as f:
    lines = f.readlines()
    for i in range(len(lines)):
        cls_name = lines[i].split("/")[0]
        ins_name = lines[i].split("/")[1].strip()
        obj_path = os.path.join(cls_id_map[cls_name], ins_name,"model.obj")
        try:
            copyfile(os.path.join(data_root,obj_path), os.path.join(save_path,obj_path))
        except:
            os.makedirs(os.path.join(save_path,cls_id_map[cls_name], ins_name))
            copyfile(os.path.join(data_root,obj_path), os.path.join(save_path,obj_path))
        test_.append(os.path.join(obj_path))
a=["02958343","02691156","02828884","03211117","04256520","04379243","04530566"]
for cls_name in a:
    cnt = 0
    lss = os.listdir(os.path.join(data_root, cls_name))
    for ls in lss:
        ls_path = os.path.join(cls_name, ls)
        if cnt >= 800:
            break
        if ls_path not in train_list and ls_path not in test_list:
            # copy to tar
            
            try:
                copyfile(os.path.join(data_root,ls_path,"model.obj"), os.path.join(save_path,ls_path,"model.obj"))
            except:
                os.makedirs(os.path.join(save_path,ls_path))
                copyfile(os.path.join(data_root,ls_path,"model.obj"), os.path.join(save_path,ls_path,"model.obj"))
            cnt += 1
            