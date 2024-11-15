import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 提取文件名中的参数
def parse_filename(filename):
    parts = filename.split('@')
    return {
        'utm_x': float(parts[1]),
        'utm_y': float(parts[2]),
        'yaw': float(parts[3]),
        'pitch': float(parts[4]),
        'roll': float(parts[5]),
        'height': float(parts[6])  
    }

# 加载数据并标记类别
def load_data(folder_path, label):
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', '.png')):  
            info = parse_filename(file_name)
            info['label'] = label
            data.append(info)
    return pd.DataFrame(data)

# 读取数据
positive_folder = '/media/bh/xujg/UAV-VisionLoc/result_positive1'
negative_folder = '/media/bh/xujg/UAV-VisionLoc/result_negative1'
df_pos = load_data(positive_folder, 1)
df_neg = load_data(negative_folder, 0)
df = pd.concat([df_pos, df_neg])

# 计算相关系数
print(df.corr()['label'])

# 使用随机森林进行特征重要性分析
X = df[['utm_x', 'utm_y', 'yaw', 'pitch', 'roll', 'height']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 输出特征重要性
importance = pd.Series(clf.feature_importances_, index=X.columns)
print("Feature importance:\n", importance.sort_values(ascending=False))

# 绘制散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='utm_x', y='utm_y', hue='label', palette=['blue', 'orange'])
plt.title('Scatter Plot of UTM_x and UTM_y by Label')
plt.xlabel('UTM_x')
plt.ylabel('UTM_y')
plt.legend(title='Label', labels=['Negative', 'Positive'])
plt.show()

# 绘制密度图
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df[df['label'] == 0], x='utm_x', y='utm_y', cmap='Blues', shade=True, alpha=0.5, label='Negative')
sns.kdeplot(data=df[df['label'] == 1], x='utm_x', y='utm_y', cmap='Oranges', shade=True, alpha=0.5, label='Positive')
plt.title('Density Plot of UTM_x and UTM_y by Label')
plt.xlabel('UTM_x')
plt.ylabel('UTM_y')
plt.legend(title='Label')
plt.show()