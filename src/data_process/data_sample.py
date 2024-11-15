import os
import random
import shutil
import numpy as np

def select_random_images(src_folder, dst_folder, num_images=500):
    # 创建目标文件夹，如果不存在则创建
    os.makedirs(dst_folder, exist_ok=True)
    
    # 获取源文件夹中的所有图像文件
    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    # 检查是否有足够的图像
    if len(image_files) < num_images:
        print(f"源文件夹中只有 {len(image_files)} 张图像，无法抽取 {num_images} 张。")
        return
    
    # 随机选择指定数量的图像
    selected_images = random.sample(image_files, num_images)
    
    # 复制图像到目标文件夹
    for img in selected_images:
        src_path = os.path.join(src_folder, img)
        dst_path = os.path.join(dst_folder, img)
        shutil.copy(src_path, dst_path)
    
    print(f"已成功抽取 {num_images} 张图像到 {dst_folder}")

def select_height_images(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    image_files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    for img in image_files:
        # at = list(map(np.float32, img.split('@')[1:7]))
        at = list(map(np.float32, img.split('@')[1:7]))
        if at[0] > 437200 and at[0] < 438800 and at[1] > 4220200 and at[1] < 4221900 and at[5] > 300:
            src_path = os.path.join(src_folder, img)
            dst_path = os.path.join(dst_folder, img)
            shutil.copy(src_path, dst_path)
    

# 使用示例
src_folder = '/media/bh/xujg/UAV-VisionLoc/data/dataset_undistort/sample_exp3'  # 替换为源图像文件夹路径
dst_folder = '/media/bh/xujg/UAV-VisionLoc/data/dataset_undistort/test_exp3'  # 替换为目标文件夹路径
select_height_images(src_folder, dst_folder)
