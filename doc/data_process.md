
# 数据处理


### 1. 谷歌卫星图像下载
目前支持两种方式：
* 1. 通过qgis插件添加谷歌地图底图
  >在qgis中，点击“插件”“管理和安装插件”， 搜索并安装“QuickMapServices”插件。安装完成后，点击“Web”“QuickMapServices”"OSM""Google""Google Satellite"。之后，导出Google地图为GeoTIFF格式，设置地图范围及空间分辨率。

* 2. 通过python代码，`get_google_tiff.py`从谷歌地图服务中获取卫星图像瓦片，并拼接成一个完整的GeoTIFF文件，该py文件主要包含以下内容：
    - URL：使用了 google_satellite_url，通过动态替换 {x}、{y}、{z} 来请求不同位置和缩放级别的瓦片。
    - 经纬度到 Web Mercator 转换：代码将定义的经纬度边界转换为 Web Mercator 投影坐标（EPSG:3857）。
    - 瓦片坐标计算：将 Web Mercator 坐标转换为谷歌地图的瓦片坐标。瓦片坐标是Google Maps API用于表示地理位置和缩放级别的内部表示法。
    - 瓦片下载与拼接：根据计算出来的瓦片坐标范围，逐一请求瓦片，并使用 requests 库下载每一块瓦片，最终通过 PIL 和 GDAL 库将这些瓦片拼接成一个完整的GeoTIFF文件。
    - 输出 GeoTIFF：生成的图像文件是以Web Mercator（EPSG:3857）投影的GeoTIFF格式保存的，并且包含地图的几何变换信息和地理坐标参考系。

### 2. 切割GeoTIFF文件

代码见`src/data_process/cut_tiff.py`，主要功能是对给定的 `GeoTIFF` 文件进行裁剪，生成指定大小的图像数据集，并将每个裁剪窗口对应的图像保存为 `JPEG` 格式文件。裁剪过程中，还计算了每个裁剪窗口的中心坐标，并将其从 `EPSG:3857` 坐标系转换为对应的 `UTM` 坐标，最终将这些 `UTM` 坐标作为文件名的一部分进行保存。

* 文件路径定义：
`google_path`：输入 `GeoTIFF` 文件的路径。
`output_dir`：用于保存裁剪图像的目录路径。如果目录不存在，使用 os.makedirs() 创建该目录。
* 函数 get_utm_zone(lon)：
计算给定经度（lon）所对应的 UTM 带。
根据公式 (lon + 180) / 6 + 1，UTM 带号从 1 到 60 之间变化。
函数 epsg3857_to_utm(x, y)：
该函数的作用是将 EPSG:3857（Web 墨卡托坐标系）的坐标转换为对应的 UTM 坐标系。
首先通过 pyproj 将 EPSG:3857 坐标转换为 WGS84 坐标（经纬度），然后再将经纬度转换为 UTM 坐标。
使用 pyproj.Proj() 创建投影对象，进行投影坐标系之间的转换。
* 打开 GeoTIFF 文件：
使用 rasterio.open() 打开指定的 GeoTIFF 文件，并获取影像的宽度、高度和地理变换参数。
裁剪窗口的设置：
window_size = 512: 裁剪窗口的大小为 512 像素。
step_size = 32: 步长设置为 32 像素，表示每次裁剪的窗口之间重叠 480 像素（512 - 32）。
* 裁剪循环：
使用两层嵌套循环遍历整个影像，按步长进行裁剪：
外层循环遍历影像的宽度（列）。
内层循环遍历影像的高度（行）。
每次迭代生成一个裁剪窗口，并通过 rasterio.windows.Window() 创建裁剪的窗口对象。
* 窗口中心坐标计算：
计算裁剪窗口中心的像素坐标 center_x 和 center_y。
通过 rasterio.transform.xy() 将这些像素坐标转换为 EPSG:3857 坐标（Web 墨卡托）。
* UTM 坐标转换：
使用 epsg3857_to_utm() 将窗口中心的 EPSG:3857 坐标转换为 UTM 坐标。
如果转换失败（可能因为 UTM 带的计算问题），则跳过该窗口。
* 读取与保存图像：
使用 src.read([1, 2, 3], window=window) 读取裁剪窗口内的图像数据（分别读取第 1、2、3 波段，通常代表 RGB 三个颜色通道）。
将裁剪的数据转化为 PIL 图像并保存为 JPEG 格式文件。
使用 UTM 坐标和裁剪索引作为文件名的一部分，例如 @UTM_X@UTM_Y@index.jpg。

针对地理影像进行裁剪，并将裁剪后的图像保存为数据集。
每个裁剪窗口对应一张 512x512 像素的图像，并且图像文件名中包含裁剪窗口的中心坐标（以 UTM 坐标系表示）。
使用步长控制窗口的重叠，最终生成大量重叠的小图像。



### 3. 创建h5文件

代码见`src/data_process/create_h5.py`，主要功能是将一个指定文件夹中的图像文件（支持格式为 .jpg、.jpeg、.png）批量读取，并保存到一个 .h5 文件中，同时保存图像名称。为了避免一次性处理大量图像造成内存问题，代码通过批量处理的方式（batch_size 参数指定）逐步将图像数据保存到 HDF5 文件中。

* 输入参数：
    source_folder: 存放图像文件的文件夹路径。
    h5_filename: 输出的 HDF5 文件名及路径。
    batch_size: 每批次处理的图像数量。

h5文件中数据集名称分别为： `image_name` `image_data`


# 日志分析

在RK3588开发板项目的根目录下会生成以exp开头的日志文件，里面存放着每次飞行时热红外摄像头捕获的图像，飞行姿态信息以及模型预测的定位信息。
代码见`src/data_process/logtoexcel.py`，主要功能是从一个日志文件中提取出特定格式的信息（如图像名称、飞行姿态、预测和实际位置等），然后将提取的数据存储到一个 pandas DataFrame 中，进一步计算预测位置和实际位置的差异，最终将这些数据保存到 Excel 文件中。

* 文件路径定义：
`file_path`: 指定了要读取的日志文件路径。
`excel_file`: 指定了保存提取结果的 Excel 文件路径。
* 读取日志文件：
使用 open(file_path, 'r') 打开日志文件，并读取其中的内容，将其存储到变量 log_text 中。整个文件的内容作为一个字符串处理。
* 正则表达式提取信息：
`pattern` 是定义提取日志文件中所需信息的正则表达式。
* 主要的提取目标：
图像名称：匹配 `"Saving image as XXX.jpg"` 的部分。
飞行姿态（航向、俯仰、滚转、飞行高度）：匹配 `Flight attitude` 信息。
预测位置：匹配 `Predict position(UTM)` 部分。
实际位置（UTM 坐标和经纬度）：匹配 `Real position(UTM)` 和 `Real position` 部分。
* 该正则表达式具体提取了以下内容：
图像名称 (.+\.jpg)
航向 (yaw)、俯仰 (pitch)、滚转 (roll)、飞行高度 (height)
预测位置 (Predict position(UTM) 的 X 和 Y)
实际位置 (Real position(UTM) 的 X 和 Y，以及经纬度）

* 解释正则表达式的各部分：
```bash
Saving image as (.+\.jpg): 匹配图像保存信息，提取图像名称。
Flight attitude: :\(yaw: ([\-\d.]+); pitch: ([\-\d.]+); roll: ([\-\d.]+); height: ([\d.]+)\): 提取飞行姿态中的航向、俯仰、滚转和高度。
Predict position\(UTM\): \(([\d.]+), ([\d.]+)\): 提取预测的 UTM 坐标。
Real position\(UTM\): \(([\d.]+), ([\d.]+)\): 提取实际的 UTM 坐标。
Real position: \(([\d.]+), ([\d.]+)\): 提取实际的经纬度坐标。
```
* 将匹配结果存储到 `DataFrame`：
使用 `pattern.findall(log_text)` 从日志文本中提取所有匹配的结果，返回一个包含匹配数据的列表。
使用 `pandas` 将提取的数据转换为 DataFrame，并为每一列命名，列名包括图像名称、飞行姿态（航向、俯仰、滚转、高度）、预测位置、实际位置以及实际的经纬度。

* 数据类型转换：
对 `Predict Position (UTM X)`、`Predict Position (UTM Y)`、`Real Position (UTM X)` 和 `Real Position (UTM Y)` 这四列进行数据类型转换，将它们转换为数值类型，以便后续进行数值计算。
* 计算 UTM 坐标的差异：
计算预测位置与实际位置之间的差异（绝对值之差），分别计算 UTM X 和 UTM Y 的差异，并将结果存储到新的列 "`Difference (UTM X)`" 和 "`Difference (UTM Y)`" 中。
* 将 DataFrame 保存到 Excel 文件：
使用 `df.to_excel()` 将处理后的 `DataFrame` 保存到指定的 Excel 文件路径。设置 `index=False`，表示不保存 DataFrame 的行索引。

