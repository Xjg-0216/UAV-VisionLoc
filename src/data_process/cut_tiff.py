'''
Descripttion: 裁剪geotiff, 生成数据集
Author: xujg
version: 
Date: 2024-07-02 13:14:21
LastEditTime: 2024-09-30 13:12:32
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
google_path = "/root/data/1.tif"
# 创建输出目录
output1_dir = "/root/data/train_database"
os.makedirs(output1_dir, exist_ok=True)

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
with rasterio.open(google_path) as src:
    # 获取影像的宽度和高度
    width = src.width
    height = src.height
    transform = src.transform
    
    # 裁剪窗口大小
    window_size = 512
    # window_size = 380
    step_size = 32
    
    # 裁剪并保存图像
    index = 0
    for col in tqdm(range(0, width - window_size + 1, step_size), desc="outer loop"):
        for row in tqdm(range(0, height - window_size + 1, step_size), desc="inner loop", leave=False):
            window = Window(col, row, window_size, window_size)
            transform_window = src.window_transform(window)
            
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
            window_data = src.read([1, 2, 3], window=window)
            # 读取第二个文件的裁剪窗口的数据
            
            # 转换第一个文件的数据为PIL图像并保存为JPEG
            img = Image.fromarray(window_data.transpose(1, 2, 0))  # 转置为HWC格式
            # 调整图像大小
            # img = img.resize((512, 512), Image.LANCZOS)
            filename1 = f"{output1_dir}/@{utm_x:.1f}@{utm_y:.1f}@{index}.jpg"
            img.save(filename1)
            
            
            index += 1

print("裁剪完成")
