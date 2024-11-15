'''
Descripttion: 
Author: xujg
version: 
Date: 2024-08-30 09:34:03
LastEditTime: 2024-09-26 17:35:11
'''


import re
import pandas as pd


file_path = '/mnt/mynewdisk1/datasets/exp4/2024-08-27_10-43-36.log'
excel_file = "/mnt/mynewdisk1/datasets/exp4/output_log_data.xlsx"

# Step 1: 读取日志文件
with open(file_path, 'r') as file:
    log_text = file.read()

# Step 2: 使用正则表达式提取信息
pattern = re.compile(r"Saving image as (.+\.jpg)\n"
                     r"\[INFO\]\[.+?\] Flight attitude: :\(yaw: ([\-\d.]+); pitch: ([\-\d.]+); roll: ([\-\d.]+); height: ([\d.]+)\)\n"
                     r"\[INFO\]\[.+?\] Predict position\(UTM\): \(([\d.]+), ([\d.]+)\)\n"
                     r"\[INFO\]\[.+?\] Real position\(UTM\): \(([\d.]+), ([\d.]+)\)\n"
                     r"\[INFO\]\[.+?\] Real position: \(([\d.]+), ([\d.]+)\)")


# Step 3: 创建一个 DataFrame 来存储提取的数据
matches = pattern.findall(log_text)

# Step 3: 创建一个 DataFrame 来存储提取的数据
columns = ["Image Name", "Yaw", "Pitch", "Roll", "Height", 
           "Predict Position (UTM X)", "Predict Position (UTM Y)",
           "Real Position (UTM X)", "Real Position (UTM Y)", 
           "Real Position (Lat)", "Real Position (Lon)"]

df = pd.DataFrame(matches, columns=columns)

# Step 4: 转换数据类型以进行计算
for col in ["Predict Position (UTM X)", "Predict Position (UTM Y)", "Real Position (UTM X)", "Real Position (UTM Y)"]:
    df[col] = pd.to_numeric(df[col])

# Step 5: 计算 UTM 位置的绝对值之差
df["Difference (UTM X)"] = (df["Predict Position (UTM X)"] - df["Real Position (UTM X)"]).abs()
df["Difference (UTM Y)"] = (df["Predict Position (UTM Y)"] - df["Real Position (UTM Y)"]).abs()

# Step 4: 保存提取的数据到 Excel 文件

df.to_excel(excel_file, index=False)

print(f"Data saved to {excel_file}")