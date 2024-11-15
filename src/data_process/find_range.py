import os

def find_lat_lon_extremes(directory):
    lat_values = []
    lon_values = []
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):  # 确保是图像文件
            # 拆分文件名，提取lat和lon
            parts = filename.split('@')
            if len(parts) >= 3:  # 确保有足够的部分
                try:
                    lat = float(parts[1])  # 第一个@后面的部分是lat
                    lon = float(parts[2])  # 第二个@后面的部分是lon
                    lat_values.append(lat)
                    lon_values.append(lon)
                except ValueError:
                    continue  # 忽略无效的数字

    # 计算最小和最大值
    if lat_values and lon_values:
        lat_min = min(lat_values)
        lat_max = max(lat_values)
        lon_min = min(lon_values)
        lon_max = max(lon_values)
        return lat_min, lat_max, lon_min, lon_max
    else:
        return None

# 使用示例
directory = '/media/bh/xujg/UAV-VisionLoc/data/new_dataset/sample1'  # 替换为你的图像目录
extremes = find_lat_lon_extremes(directory)
if extremes:
    lat_min, lat_max, lon_min, lon_max = extremes
    print(f'纬度最小值: {lat_min}, 最大值: {lat_max}')
    print(f'经度最小值: {lon_min}, 最大值: {lon_max}')
else:
    print('未找到有效的图像文件')
