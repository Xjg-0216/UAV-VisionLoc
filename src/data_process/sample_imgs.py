import os
import shutil
from tqdm import tqdm
import numpy as np

images_dir = "/media/bh/xujg/UAV-VisionLoc/data/new_dataset/sample2"
output_dir = "/media/bh/xujg/UAV-VisionLoc/data/dataset_wo_orth/test_queries"
os.makedirs(output_dir, exist_ok=True)
images_list = os.listdir(images_dir)
print("total images num: ", len(images_list))
for img_list in tqdm(images_list):
    img_s = os.path.join(images_dir, img_list)
    at = list(map(np.float32, img_list.split('@')[1:7]))
    loc1, loc2, yaw, pitch, roll, height = at[0], at[1], at[2], at[3], at[4], at[5]
    if abs(pitch) < 0.1 and abs(roll) < 0.1 and height > 400 and loc1 > 437200 and loc1 < 438800 and loc2 > 4219400 and loc2 < 4221300:
        img_o = os.path.join(output_dir, img_list)
        shutil.copy(img_s, img_o)
