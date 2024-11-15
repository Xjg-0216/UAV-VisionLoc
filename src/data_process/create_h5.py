'''
Descripttion: 
Author: xujg
version: 
Date: 2024-09-30 10:56:52
LastEditTime: 2024-09-30 10:59:29
'''
import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

def create_h5_file(source_folder, h5_filename, batch_size=1000):
    
    supported_formats = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(source_folder) if f.endswith(supported_formats)]
    total_files = len(files)

    with h5py.File(h5_filename, 'w') as h5f:
        # 先创建占位符数据集
        img_data_dset = None
        img_name_dset = None

        for i in tqdm(range(0, total_files, batch_size)):
            batch_files = files[i:i + batch_size]
            img_data_list = []
            img_name_list = []

            for file in batch_files:
                img_path = os.path.join(source_folder, file)
                
                try:
                    with Image.open(img_path) as img:
                        img_data = np.array(img)

                    # 确保图像数据是3D数组（例如：RGB图像）
                    if img_data.ndim != 3:
                        raise ValueError(f"Image {file} is not a 3D array")
                    
                    img_data_list.append(img_data)
                    file = file.replace("_", "@")[:-4] + "@.jpg"
                    img_name_list.append(file)
                except Exception as e:
                    print(f"Error processing {file}: {e}")

            if img_data_list:
                img_data_array = np.array(img_data_list, dtype=np.uint8)
                img_name_array = np.array(img_name_list, dtype=h5py.string_dtype(encoding='utf-8'))

                if img_data_dset is None:
                    # 在第一次写入时创建数据集，并设置数据集的大小
                    img_data_shape = (total_files,) + img_data_array.shape[1:]
                    img_data_dset = h5f.create_dataset('image_data', shape=img_data_shape, dtype=np.uint8)

                    img_name_dset = h5f.create_dataset('image_name', shape=(total_files,), dtype=h5py.string_dtype(encoding='utf-8'))

                # 写入每一批次数据
                img_data_dset[i:i + len(img_data_array)] = img_data_array
                img_name_dset[i:i + len(img_name_array)] = img_name_array

        print("Image datasets created successfully")


source_folder = '/media/bh/xujg/UAV-VisionLoc/data/dataset_undistort/sample3_orth_queries'
h5_filename = '/media/bh/xujg/UAV-VisionLoc/data/dataset_undistort/train_queries_orth.h5'
create_h5_file(source_folder, h5_filename, batch_size=1000)
