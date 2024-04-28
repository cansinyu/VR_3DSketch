import subprocess

# 定义 Blender 命令
blender_executable_path = "/root/autodl-tmp/project/blender-2.79-linux-glibc219-x86_64/blender"
script_path = "/root/autodl-tmp/project/VR/obj2fbx.py"
obj_file_path = "/root/autodl-tmp/project/VR/output_path/sdf_0.obj"
fbx_file_path = "/root/autodl-tmp/project/VR/VR_Sketch/CVPR11.26/FBX_data/sdf_0.fbx"

# 构建完整命令
command = [
    blender_executable_path,
    "-b",
    "-P", script_path,
    "--",
    obj_file_path,
    fbx_file_path
]

# 运行命令
try:
    subprocess.run(command, check=True)
    print("Blender command executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")
