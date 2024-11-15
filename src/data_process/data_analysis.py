import pandas as pd
import os

# 假设 load_data 函数已经定义好，用于加载图像文件夹中的数据，并为数据加上标签
def load_data(folder, label):
    # 初始化空列表用于存储数据
    data = []
    
    # 遍历文件夹中的图像文件
    for filename in os.listdir(folder):
        # 假设文件名中包含了我们需要的元数据，以 '@' 分隔
        # 示例：filename = "image@utm_x@utm_y@yaw@pitch@roll@height.jpg"
        # 去掉扩展名并分割元数据
        info = filename.split('@')
        
        # 如果格式不符合预期，跳过该文件
        if len(info) < 6:
            continue
        
        # 提取元数据
        utm_x, utm_y, yaw, pitch, roll, height = map(float, info[1:7])
        
        # 将数据添加到列表中
        data.append([utm_x, utm_y, yaw, pitch, roll, height, label])
    
    # 将数据转换为 DataFrame
    df = pd.DataFrame(data, columns=['utm_x', 'utm_y', 'yaw', 'pitch', 'roll', 'height', 'label'])
    return df

# 文件夹路径
positive_folder = '/media/bh/xujg/UAV-VisionLoc/result_positive2'
negative_folder = '/media/bh/xujg/UAV-VisionLoc/result_negative2'

# 加载数据并合并
df_pos = load_data(positive_folder, 1)
df_neg = load_data(negative_folder, 0)
df = pd.concat([df_pos, df_neg], ignore_index=True)

# 保存合并后的数据到 CSV 文件
output_csv_path = '/media/bh/xujg/UAV-VisionLoc/data_combined2.csv'
df.to_csv(output_csv_path, index=False)
print(f"数据已保存到 {output_csv_path}")
