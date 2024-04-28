import os
import shutil
from off2obj import off2obj as pd1                  # off2obj
from processing_data1 import process_data as pd2    # filter obj
from processing_data2 import process_data as pd3    # obj2npy
from gen_obj import main_process as pd4             # npy gen obj

def process(data_root, output_path):
    pd1(data_root, output_path)
    pd2(output_path)
    shutil.rmtree(os.path.join(output_path, "step1"))
    pd3(output_path)
    shutil.rmtree(os.path.join(output_path, "step2"))
    gen_data_root = os.path.join(output_path, "step3")
    pd4(gen_data_root, output_path)
    shutil.rmtree(os.path.join(output_path, "step3"))



if __name__ == '__main__':
    data_root = "VR_Sketch/CVPR11.26/OFF_data"
    output_path = "output_path"
    process(data_root, output_path)