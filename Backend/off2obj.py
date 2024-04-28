import numpy as np
import os
from tqdm import tqdm
import shutil


# import torch

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def off2obj(off_path, output_path):
    off_list = []
    text_list = []
    for root, dirs, files in os.walk(off_path):
        for file in files:
            if file.endswith('.off'):
                off_list.append(os.path.join(root, file))
            if file.endswith('.txt'):
                text_list.append(os.path.join(root, file))

    # print(f"need to slover: {len(off_list)}")
    # print(f"number of txt: {len(text_list)}")

    off_num = range(len(off_list))

    for i in tqdm(off_num, desc="Processing data"):
        off = off_list[i]
        obj = off.rstrip(".off")

        f = open(off)
        length = 0
        while "\n" in f.readline():
            length += 1
        f.close()

        head_idx = 1
        vert = []
        faces = []
        f = open(off)
        out = "# " + obj + "\n"
        for j in range(length):
            line = f.readline().split()

            if j == 0:
                if line[0] == "OFF":
                    head_idx = 1
                elif line[0] != "OFF" and ("OFF" in line[0]):
                    head_idx = 0

            if j > head_idx:
                y = [float(value) for value in line]
                if len(y) == 3:
                    vert.append(y)
                elif len(y) == 4:
                    faces.append(y[1:])

        vert = np.array(vert)
        max_vert = np.max(vert, axis=0)
        min_vert = np.min(vert, axis=0)
        cent_vert = (max_vert + min_vert) / 2
        vert = vert - cent_vert.reshape(1, 3)
        max_abs = np.max(np.abs(vert))
        scale = 0.4 / max_abs
        vert = vert * scale

        for j in range(vert.shape[0]):
            out += "v " + str(vert[j, 0]) + " " + str(vert[j, 1]) + " " + str(vert[j, 2]) + "\n"

        faces = np.array(faces)
        for j in range(faces.shape[0]):
            out += "f " + str(int(faces[j, 0] + 1)) + " " + str(int(faces[j, 1] + 1)) + " " + str(
                int(faces[j, 2] + 1)) + "\n"

        save_path = os.path.join(output_path, 'step1')
        ensure_directory(save_path)
        shutil.copy(text_list[i], save_path)
        new_obj = os.path.join(save_path, off.split('/')[-1]).replace('.off', '.obj')
        w = open(new_obj, "w")
        w.write(out)
        w.close()
        f.close()
        print("Done: " + obj)


def main(root):
    np.random.seed(45)
    # torch.manual_seed(45)
    off_path = root #os.path.join(root, 'OFF_data')
    output_path = os.path.join(root, "OBJ")
    off2obj(off_path, output_path)


if __name__ == "__main__":
    root = "/root/autodl-tmp/data"
    main(root)