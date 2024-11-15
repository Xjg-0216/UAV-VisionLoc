'''
Descripttion: 裁剪geotiff, 生成数据集
Author: xujg
version: 
Date: 2024-07-02 13:14:21
LastEditTime: 2024-09-15 11:43:50
'''


import rasterio
from rasterio.windows import Window
from PIL import Image
import os
import pyproj
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

#输入文件路径
odm_orthophoto_path = "/media/bh/xujg/crop_orth2.tif"
google_path = "/media/bh/xujg/satellite_2.tif"
# 创建输出目录
output1_dir = "/media/bh/xujg/UAV-VisionLoc/data/test/queries_imgs"
output2_dir = "/media/bh/xujg/UAV-VisionLoc/data/test/database_imgs"             # /database_imgs
os.makedirs(output1_dir, exist_ok=True)
os.makedirs(output2_dir, exist_ok=True)

# 计算给定经度所属的UTM带
def get_utm_zone(lon):
    return int((lon + 180) / 6) + 1

# 将EPSG:3857坐标转换为UTM坐标
def epsg3857_to_utm(x, y):
    wgs84_proj = pyproj.Proj(init='epsg:3857')  # 初始化EPSG:3857投影
    lon, lat = pyproj.transform(wgs84_proj, pyproj.Proj(init='epsg:4326'), x, y)  # 转换到WGS84坐标
    utm_zone = get_utm_zone(lon)
    utm_proj = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    utm_x, utm_y = pyproj.transform(pyproj.Proj(init='epsg:4326'), utm_proj, lon, lat)
    return utm_x, utm_y

# 打开GeoTIFF文件
with rasterio.open(odm_orthophoto_path) as src1, rasterio.open(google_path) as src2:
    # 获取影像的宽度和高度
    width = src1.width
    height = src1.height
    transform = src1.transform
    
    # 确保两个文件的尺寸一致
    if width != src2.width or height != src2.height:
        raise ValueError("两个文件的尺寸不一致")
    
    # 裁剪窗口大小
    window_size = 512
    step_size = 32
    
    # 裁剪并保存图像
    index = 0
    for col in tqdm(range(0, width - window_size + 1, step_size), desc="outer loop"):
        for row in tqdm(range(0, height - window_size + 1, step_size), desc="inner loop", leave=False):
            window = Window(col, row, window_size, window_size)
            transform_window = src1.window_transform(window)
            
            # 获取窗口中心的EPSG:3857坐标
            center_x = col + window_size // 2
            center_y = row + window_size // 2
            center_lon, center_lat = rasterio.transform.xy(transform, center_y, center_x)
            
            try:
                utm_x, utm_y = epsg3857_to_utm(center_lon, center_lat)
            except ValueError as e:
                print(f"Skipping invalid UTM zone for window at col {col}, row {row}: {e}")
                continue


            # 读取第一个文件的裁剪窗口的数据
            window_data1 = src1.read([1, 2, 3], window=window)
            # 读取第二个文件的裁剪窗口的数据
            window_data2 = src2.read([1, 2, 3], window=window)
            
            # 转换第一个文件的数据为PIL图像并保存为JPEG
            img1 = Image.fromarray(window_data1.transpose(1, 2, 0))  # 转置为HWC格式
            filename1 = f"{output1_dir}/@{utm_x:.6f}@{utm_y:.6f}@{index}.jpg"
            img1.save(filename1)
            
            # 转换第二个文件的数据为PIL图像并保存为JPEG
            img2 = Image.fromarray(window_data2.transpose(1, 2, 0))  # 转置为HWC格式
            filename2 = f"{output2_dir}/@{utm_x:.6f}@{utm_y:.6f}@{index}.jpg"
            img2.save(filename2)
            
            index += 1

print("裁剪完成")