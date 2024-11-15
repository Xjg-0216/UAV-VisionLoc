from osgeo import gdal, osr
import math
import requests
from io import BytesIO
from PIL import Image
import numpy as np

# Google Satellite XYZ URL
google_satellite_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'

# 设置请求头，模拟浏览器请求
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"
# }

# 定义边界框和缩放级别 (WGS84, EPSG:4326)
upper_left_lon, upper_left_lat = 116.404, 39.915  # 北京
lower_right_lon, lower_right_lat = 116.414, 39.905  # 北京
zoom_level = 18
tile_size = 256

# 将经纬度转换为瓦片坐标
def lonlat_to_tile(lon, lat, zoom):
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

# 获取瓦片坐标
upper_left_tile = lonlat_to_tile(upper_left_lon, upper_left_lat, zoom_level)
lower_right_tile = lonlat_to_tile(lower_right_lon, lower_right_lat, zoom_level)

# 计算目标图像的尺寸
num_tiles_x = lower_right_tile[0] - upper_left_tile[0]
num_tiles_y = lower_right_tile[1] - upper_left_tile[1]
output_width = num_tiles_x * tile_size
output_height = num_tiles_y * tile_size

# 创建 GeoTIFF 文件
output_path = './google_satellite_geotiff_output1.tif'
driver = gdal.GetDriverByName('GTiff')
output_raster = driver.Create(output_path, output_width, output_height, 3, gdal.GDT_Byte)

# 设置地理变换和投影
origin_x = upper_left_lon
origin_y = upper_left_lat

# 设置像素大小，确保像素大小与所需的 `gdalinfo` 输出一致
pixel_width = (lower_right_lon - upper_left_lon) / output_width
pixel_height = (lower_right_lat - upper_left_lat) / output_height
output_raster.SetGeoTransform((origin_x, pixel_width, 0, origin_y, 0, -pixel_height))

# 设置投影为 EPSG:4326 (WGS 84)
srs = osr.SpatialReference()
srs.ImportFromEPSG(4326)
output_raster.SetProjection(srs.ExportToWkt())

# 下载瓦片并写入 GeoTIFF
for x in range(upper_left_tile[0], lower_right_tile[0]):
    for y in range(upper_left_tile[1], lower_right_tile[1]):
        tile_url = google_satellite_url.format(x=x, y=y, z=zoom_level)
        # response = requests.get(tile_url, headers=headers)
        response = requests.get(tile_url)
        
        if response.status_code == 200:
            # 将瓦片图像数据加载到内存并解码
            tile_data = BytesIO(response.content)
            image = Image.open(tile_data)
            tile_array = np.array(image)
            
            # 确保数据维度匹配 RGB 格式
            if tile_array.ndim == 3 and tile_array.shape[2] == 3:
                x_offset = (x - upper_left_tile[0]) * tile_size
                y_offset = (y - upper_left_tile[1]) * tile_size
                
                # 使用 GDAL 写入每个波段的数据
                for i in range(3):
                    band_data = tile_array[:, :, i]
                    output_raster.GetRasterBand(i + 1).WriteArray(band_data, x_offset, y_offset)
            else:
                print(f"Unexpected tile format at {x}, {y}")
        else:
            print(f"Failed to download tile from {tile_url}, HTTP status code: {response.status_code}")

# 关闭并保存 GeoTIFF 文件
output_raster = None

print(f"Layer exported to {output_path}")
