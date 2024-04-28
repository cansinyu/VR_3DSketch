from multiprocessing import freeze_support
from flask import Flask, request, jsonify, send_file
import os
import shutil
import subprocess
import time
from flask_cors import CORS
from off2obj import off2obj as pd1
from processing_data1 import process_data as pd2
from processing_data2 import process_data as pd3
from gen_obj import main_process as pd4
from werkzeug.utils import secure_filename
from network.model_trainer import DiffusionModel
from sparsity_network.sparsity_trainer import Sparsity_DiffusionModel

app = Flask(__name__)
CORS(app)

# 提取常量
UPLOAD_FOLDER = "VR_Sketch/CVPR11.26/OFF_data"
ROOT_FOLDER = "VR_Sketch/CVPR11.26"
OUTPUT_PATH = "output_path"

folders_to_check = ["VR_Sketch/CVPR11.26/OFF_data"]

for folder in folders_to_check:
    if not os.path.exists(folder):
        os.makedirs(folder)

# load model
occupancy_model_path = "pretrain_model/occupancy_model.ckpt"
discrete_diffusion = DiffusionModel.load_from_checkpoint(occupancy_model_path).cuda()
sdf_model_path = "pretrain_model/sdf_model.ckpt"
sdf_model = Sparsity_DiffusionModel.load_from_checkpoint(sdf_model_path).cuda()


def run_playfbx_script(obj_file_path, fbx_file_path):
    blender_executable = "/root/autodl-tmp/project/blender-2.79-linux-glibc219-x86_64/blender"
    script_file = "/root/autodl-tmp/project/VR/obj2fbx.py"
    try:
        subprocess.run([blender_executable, '-b', '-P', script_file, '--', obj_file_path, fbx_file_path], check=True)
        print("FBX conversion completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during FBX conversion: {e}")

def find_fbx_files(root):
    fbx_files = []
    for subdir, _, files in os.walk(root):
        for file in files:
            if file.lower().endswith('.fbx'):
                fbx_files.append(os.path.join(subdir, file))
    return fbx_files


@app.route('/get_fbx', methods=['POST'])
def handle_process():
    if 'off' not in request.files or 'txt' not in request.files:
        return jsonify(error="No file part"), 401

    off_file = request.files['off']
    txt_file = request.files['txt']

    if off_file.filename == '' or txt_file.filename == '':
        return jsonify(error="No selected file"), 402

    off_filename = secure_filename(off_file.filename)
    off_path = os.path.join(UPLOAD_FOLDER, off_filename)
    off_file.save(off_path)

    txt_filename = secure_filename(txt_file.filename)
    txt_path = os.path.join(UPLOAD_FOLDER, txt_filename)
    txt_file.save(txt_path)

    print("开始处理......")
    t1 = time.time()
    try:
        pd1(UPLOAD_FOLDER, OUTPUT_PATH)
        t2 = time.time()
        pd2(OUTPUT_PATH)
        shutil.rmtree(os.path.join(OUTPUT_PATH, "step1"))
        t3 = time.time()
        pd3(OUTPUT_PATH)
        shutil.rmtree(os.path.join(OUTPUT_PATH, "step2"))
        t4 = time.time()
        gen_data_root = os.path.join(OUTPUT_PATH, "step3")
        pd4(gen_data_root, OUTPUT_PATH, discrete_diffusion,sdf_model)
        shutil.rmtree(os.path.join(OUTPUT_PATH, "step3"))
        t5 = time.time()
    except Exception as e:
        return jsonify({'error': str(e)}), 405
    os.remove(off_path)
    os.remove(txt_path)
    # 删除上传的原始文件和生成的中间文件
    #shutil.rmtree(os.path.join(ROOT_FOLDER, "OFF_data"))
    # 假设您已经生成了 OBJ 文件，接下来将其转换为 FBX 文件
    obj_file_path = "/root/autodl-tmp/project/VR/output_path/sdf_0.obj"  # 这里填入 OBJ 文件的路径
    fbx_file_path = "/root/autodl-tmp/project/VR/VR_Sketch/CVPR11.26/FBX_data/sdf_0.fbx"  # 保存 FBX 文件的路径
    run_playfbx_script(obj_file_path, fbx_file_path)
    t6 = time.time()
    # 查找生成的 FBX 文件
    fbx_files = find_fbx_files(os.path.join(ROOT_FOLDER, "FBX_data"))
    if not fbx_files:
        return jsonify(error="No FBX file found"), 406
    print(f"pd1:{t2-t1}\n pd2:{t3-t2}\n pd3:{t4-t3}\n pd4:{t5-t4}\n pd5:{t6-t5}\n all:{t6-t1}\n")
    return send_file(fbx_files[0], as_attachment=True)

if __name__ == '__main__':
    freeze_support()
    app.run(host='0.0.0.0', port=6006)