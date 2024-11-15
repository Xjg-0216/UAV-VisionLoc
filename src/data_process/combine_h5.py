import h5py
import numpy as np

def merge_h5_files(file1, file2, output_file):
    with h5py.File(file1, 'r') as h5f1, h5py.File(file2, 'r') as h5f2, h5py.File(output_file, 'w') as h5f_out:
        # 读取第一个文件的数据
        img_data1 = h5f1['image_data'][:]
        img_name1 = h5f1['image_name'][:]
        
        # 读取第二个文件的数据
        img_data2 = h5f2['image_data'][:]
        img_name2 = h5f2['image_name'][:]
        
        # 合并数据
        merged_img_data = np.concatenate((img_data1, img_data2), axis=0)
        merged_img_name = np.concatenate((img_name1, img_name2), axis=0)
        
        # 创建新的数据集并写入合并后的数据
        h5f_out.create_dataset('image_data', data=merged_img_data, dtype=np.uint8)
        h5f_out.create_dataset('image_name', data=merged_img_name, dtype=h5py.string_dtype(encoding='utf-8'))



file1_h5 = "/media/bh/xujg/UAV-VisionLoc/data/dataset_undistort/train_queries_undistort.h5"
file2_h5 = "/media/bh/xujg/UAV-VisionLoc/data/dataset_undistort/train_queries_orth.h5"
file_combine = "/media/bh/xujg/UAV-VisionLoc/data/dataset_undistort/train_queries.h5"

merge_h5_files(file1_h5, file2_h5, file_combine)
