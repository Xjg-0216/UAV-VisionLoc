from osgeo import gdal, osr
import math
import requests
from io import BytesIO
from PIL import Image
import numpy as np

# Google Satellite XYZ URL
google_satellite_url = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}'

# 定义边界框和缩放级别 (WGS84, EPSG:4326)
upper_left_lon, upper_left_lat = 116.404, 39.915  # 北京
lower_right_lon, lower_right_lat = 116.454, 39.855  # 北京
zoom_level = 18
tile_size = 256

# 将经纬度转换为 Web Mercator 投影坐标（EPSG:3857）
def lonlat_to_web_mercator(lon, lat):
    origin_shift = 2 * math.pi * 6378137 / 2.0
    mx = lon * origin_shift / 180.0
    my = math.log(math.tan((90 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    my = my * origin_shift / 180.0
    return mx, my

# 计算 Web Mercator 坐标范围
upper_left_x, upper_left_y = lonlat_to_web_mercator(upper_left_lon, upper_left_lat)
lower_right_x, lower_right_y = lonlat_to_web_mercator(lower_right_lon, lower_right_lat)

print(f"Web Mercator coordinates:")
print(f"Upper Left: {upper_left_x}, {upper_left_y}")
print(f"Lower Right: {lower_right_x}, {lower_right_y}")

# 将 Web Mercator 坐标转换为瓦片坐标
def web_mercator_to_tile(mx, my, zoom):
    origin_shift = 2 * math.pi * 6378137 / 2.0
    xtile = int((mx + origin_shift) / origin_shift / 2 * (2 ** zoom))
    ytile = int((origin_shift - my) / origin_shift / 2 * (2 ** zoom))
    return (xtile, ytile)

# 获取瓦片编号
upper_left_tile = web_mercator_to_tile(upper_left_x, upper_left_y, zoom_level)
lower_right_tile = web_mercator_to_tile(lower_right_x, lower_right_y, zoom_level)

print(f"Tile coordinates:")
print(f"Upper Left Tile: {upper_left_tile}")
print(f"Lower Right Tile: {lower_right_tile}")

# 计算目标图像的尺寸
num_tiles_x = lower_right_tile[0] - upper_left_tile[0]
num_tiles_y = lower_right_tile[1] - upper_left_tile[1]
output_width = num_tiles_x * tile_size
output_height = num_tiles_y * tile_size

print(f"Output dimensions: {output_width} x {output_height}")

# 创建 GeoTIFF 文件
output_path = './google_satellite_geotiff_output_3857.tif'
driver = gdal.GetDriverByName('GTiff')
output_raster = driver.Create(output_path, output_width, output_height, 3, gdal.GDT_Byte)

# 设置地理变换和投影 (Web Mercator, EPSG:3857)
pixel_width = (lower_right_x - upper_left_x) / output_width
pixel_height = (upper_left_y - lower_right_y) / output_height
output_raster.SetGeoTransform((upper_left_x, pixel_width, 0, upper_left_y, 0, -pixel_height))

# 设置投影为 EPSG:3857 (Web Mercator)
srs = osr.SpatialReference()
srs.ImportFromEPSG(3857)
output_raster.SetProjection(srs.ExportToWkt())

# 下载瓦片并写入 GeoTIFF
for x in range(upper_left_tile[0], lower_right_tile[0]):
    for y in range(upper_left_tile[1], lower_right_tile[1]):
        tile_url = google_satellite_url.format(x=x, y=y, z=zoom_level)
        response = requests.get(tile_url)
        
        if response.status_code == 200:
            # 将瓦片图像数据加载到内存并解码
            tile_data = BytesIO(response.content)
            image = Image.open(tile_data)
            tile_array = np.array(image)
            
            # 确保数据维度匹配 RGB 格式
            if tile_array.ndim == 3 and tile_array.shape[2] == 3:
                # 计算偏移，确保瓦片拼接正确
                x_offset = (x - upper_left_tile[0]) * tile_size
                y_offset = (y - upper_left_tile[1]) * tile_size
                
                print(f"Writing tile ({x}, {y}) at offset ({x_offset}, {y_offset}), size {tile_array.shape}")
                
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
