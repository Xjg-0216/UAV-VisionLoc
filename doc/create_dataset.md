# 无人机定位数据集制作

主要分为以下步骤： 

* 通过ODM将拍摄的图像转换成正射地图  

* 通过QGIS将正射地图与谷歌地图对齐  

* 通过python将对齐后的地图进行裁剪   

### ODM


OpenDroneMap (ODM) 是一个开源的软件项目，用于处理从无人机获取的图像数据，并生成高质量的地理空间信息产品。以下是对 OpenDroneMap 的详细介绍：  
* 图像拼接（Orthophoto Generation）：
        ODM 能够将无人机拍摄的重叠图像拼接成高分辨率的正射影像（orthophoto）

* 三维模型重建（3D Reconstruction）：
        将二维图像转换为三维模型，包括点云、网格和纹理模型

* 数字表面模型（DSM）和数字高程模型（DEM）生成：
        ODM 可以生成 DSM 和 DEM，分别表示地表物体和地形的高程数据。

* 多光谱和热成像数据处理：
        除了可见光图像，ODM 还支持多光谱和热成像数据的处理。

ODM项目github地址：[https://github.com/OpenDroneMap/ODM](https://github.com/OpenDroneMap/ODM)
ODM项目文档：[https://docs.opendronemap.org/](https://docs.opendronemap.org/)


#### 1.快速开始

使用docker运行ODM, 首先拉取镜像
~~`sudo docker pull opendronemap/opendronemap` , 此镜像已弃用~~

拉取镜像
```bash
sudo docker pull opendronemap/odm
```
!!! note
    拉取这个镜像需要代理服务器



```bash
xujg@xujg-ps:~$ sudo docker images
REPOSITORY                  TAG       IMAGE ID       CREATED        SIZE
opendronemap/odm            latest    e3d462eb2c32   26 hours ago   1.61GB
hello-world                 latest    feb5d9fea6a5   2 years ago    13.3kB
opendronemap/opendronemap   latest    8c89143494c7   5 years ago    3.04GB

```
示例数据集在： [ https://github.com/OpenDroneMap/odm_data ]( https://github.com/OpenDroneMap/odm_data )

例如pull aukerman数据集
在仓库根目录下运行以下命令：
```bash
sudo docker run -it --rm \
    -v "$(pwd)/images:/code/images" \
    -v "$(pwd)/odm_orthophoto:/code/odm_orthophoto" \
    -v "$(pwd)/odm_texturing:/code/odm_texturing" \
    opendronemap/opendronemap
```
#### 2.输入数据
无人机拍摄的重叠图像。这些图像可以是普通的**可见光图像**、**多光谱图像**或**热成像图像**。图像应具有足够的重叠度，通常建议 60-80% 的前向重叠（沿飞行方向）和侧向重叠（垂直于飞行方向）。
#### 3.输出数据
正射影像 (Orthophoto)：

    高分辨率、地理校正后的正射影像，通常以 GeoTIFF 格式存储。

三维模型 (3D Model)：

    包括点云（Point Cloud）、网格模型（Mesh）和纹理模型（Textured Model），通常以 .obj 或 .ply 格式存储。

数字表面模型 (DSM) 和 数字高程模型 (DEM)：

    表示地表和地形的高程数据，通常以 GeoTIFF 格式存储。

!!! danger
    如果输入图像数据没有EXIF信息， 则需要加地理信息， 否则表面纹理映射，数字高程模型，正射图像都没办法得到


==默认情况下，在项目根目录中加入`geo.txt`(地理信息)， 此时ODM会自动检测到它，如果它有其他名称，则需要指定==
格式如下：
```text
EPSG:4326
stamp66284.jpg 116.3104629517 38.1277542114 115.0
stamp27289.jpg 116.2984924316 38.1408538818 115.0
stamp56536.jpg 116.292350769 38.1299972534 114.0
```

执行ODM命令示例：
```bash
sudo docker run -it --rm     \
-v "/home/xujg/code/project:/datasets"      \
opendronemap/odm     /datasets
```

输出：
```text
该过程完成后，结果将按如下方式组织：

|-- images/
    |-- img-1234.jpg
    |-- ...
|-- opensfm/
    |-- see mapillary/opensfm repository for more info
|-- odm_meshing/
    |-- odm_mesh.ply                    # A 3D mesh
|-- odm_texturing/
    |-- odm_textured_model.obj          # Textured mesh
    |-- odm_textured_model_geo.obj      # Georeferenced textured mesh
|-- odm_georeferencing/
    |-- odm_georeferenced_model.laz     # LAZ format point cloud
|-- odm_orthophoto/
    |-- odm_orthophoto.tif              # Orthophoto GeoTiff
```


### 将正射影像与谷歌地图对齐
使用QGIS等GIS软件，将正射影像与谷歌地图对齐。以下是详细步骤：

##### 1.安装QGIS
下载链接[https://www.qgis.org/en/site/forusers/alldownloads.html#debian-ubuntu](https://www.qgis.org/en/site/forusers/alldownloads.html#debian-ubuntu)  

使用教程: [https://www.osgeo.cn/qgisdoc/docs/user_manual/index.html](https://www.osgeo.cn/qgisdoc/docs/user_manual/index.html)
##### 2.导入正射影像
打开QGIS

* 点击“图层”->“添加图层”->“添加栅格图层”。
* 选择正射影像文件（如 odm_orthophoto.tif）并点击“打开”。

###### 添加谷歌地图底图

* 在QGIS中，点击“插件”->“管理和安装插件”。
* 搜索并安装“QuickMapServices”插件。
* 安装完成后，点击“Web”->“QuickMapServices”->“OSM”->“Google”->“Google Satellite”。

>如果OSM中没有Google， 可以点击OSM中的setting->More services进行下载

###### 可以切换成中文，并设置代理服务器
##### 3.对齐正射影像

###### 设置空间分辨率及重投影Geo-TIFF文件

* 右键该文件-> 打开属性 -> 信息 -> 查看坐标参考系CRS 
* 栅格-> 投影 -> 变形（重投影：
* 在弹出的窗口中，设置以下参数：
    * Input Layer（输入图层）：选择你的栅格图层
    * Source CRS（源CRS）：选择当前栅格数据的CRS
    * Target CRS（目标CRS）：选择EPSG:3857
    * Resampling method（重采样方法）：选择适合你的数据的重采样方法（例如，Bilinear 或 Cubic）
    * 输出文件分辨率选择：1.00000
    * Output file（输出文件）：选择保存重投影栅格的路径和文件名


!!! note 
    使用gdalinfo查看Geotiff文件的信息 `gdalinfo yourfile.tif`
##### 4.导出google地图为GeoTIFF格式

* 选择`google`地图图层，右键导出， 另存为
* 不创建`VRT`
* 填写文件名
* CRS: `EPSG:3857 - WGS  84`
* 范围选择与重采样投影相同的大小
* 分辨率与重投影设置一致，例如都为1， 此时输出的`google.tiff`空间分辨率及尺寸与正射影像的均一致



### 地图裁剪

在导出的`Geotiff`中，`odm_orthophoto.tif`和`google.tif`设置了相同的空间分辨率(1m/pt)， 尺寸大小也相同

使用python进行以下处理：  

* 1.沿`odm_orthophoto.tif`文件裁剪， 像素大小为$512\times512$， 步长为32  

* 2.坐标采用$512\times512$子图中心点的位置坐标，并将`EPSG:3857`转换为`UTM`坐标  

* 3.采用相同的方法对`google.tif`文件进行裁剪  

* 4.数据筛选及清洗  