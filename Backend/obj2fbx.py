import bpy
import sys
import os

# 获取命令行参数
argv = sys.argv
argv = argv[argv.index("--") + 1:]  # 获取所有双破折号之后的参数

if len(argv) < 2:
    print("Usage: blender -b -P this_script.py -- <path_to_obj_file> <path_to_fbx_file>")
    sys.exit(1)

obj_file_path = argv[0]
fbx_file_path = argv[1]

# 删除初始对象
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# 导入 OBJ 文件
bpy.ops.import_scene.obj(filepath=obj_file_path)

# 获取导入的对象
obj = bpy.context.selected_objects[0]

# 切换到编辑模式，并选择所有面
bpy.context.scene.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')

# 执行智能 UV 展开
bpy.ops.uv.smart_project()

# 切换回对象模式
bpy.ops.object.mode_set(mode='OBJECT')

# 导出为 FBX 文件
bpy.ops.export_scene.fbx(filepath=fbx_file_path, use_selection=False)

print("Conversion completed: OBJ to FBX")
