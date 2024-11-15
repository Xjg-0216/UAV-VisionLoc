import os
import sys
import torch
import parser
import logging
import sklearn
from os.path import join
from datetime import datetime
import numpy as np
import copy
from torch.utils.data import DataLoader
import test
import util
import commons
import datasets_ws
from model import network
from tqdm import tqdm
import h5py
from sklearn.neighbors import NearestNeighbors
import faiss
from utils.plotting import process_results_simulation
import cv2
import math

######################################### SETUP #########################################
def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to a rotation matrix.
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def distort(img_path):
    img = cv2.imread(img_path)
    # Image dimensions
    height, width = img.shape[:2]
    yaw, pitch, roll = np.float32(img_path.split("@")[3]), np.float32(img_path.split("@")[4]), np.float32(img_path.split("@")[5])
    # Get the rotation matrix
    R = euler_to_rotation_matrix(roll, pitch, yaw)

    # Define the source points (four corners of the image)
    src_points = np.array([[0, 0],
                           [width - 1, 0],
                           [width - 1, height - 1],
                           [0, height - 1]], dtype='float32')

    # Apply the rotation to the source points
    dst_points = np.dot(src_points - np.array([width / 2, height / 2]), R[:2, :2].T) + np.array([width / 2, height / 2])
    dst_points = dst_points.astype('float32')

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation
    adjusted_image = cv2.warpPerspective(img, matrix, (width, height))


    # K = np.array([[1.3962*512, 0, 0.5* 512],
    #           [0, 1.3962*512, 0.5*512],
    #           [0, 0, 1]])

    # # 构造透视变换矩阵 H = K * R * K^(-1)
    # H = K @ R @ np.linalg.inv(K)
    # # 对图像应用透视变换
    # adjusted_image = cv2.warpPerspective(img, H, (width, height))
    return adjusted_image

def crop_center(image, crop_width, crop_height):
    """
    Crop the center of the image to the specified width and height.
    """
    h, w = image.shape[:2]
    start_x = w // 2 - crop_width // 2
    start_y = h // 2 - crop_height // 2
    return image[start_y:start_y + crop_height, start_x:start_x + crop_width]

def pre_process(img_path, contrast_factor=1):

    # Read image using OpenCV
    # img = cv2.imread(img_path)
    # distort
    img = distort(img_path)
    # img = cv2.resize(img, (512, 512))
    img = crop_center(img, 512, 512)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert 
    # 调整对比度
    # image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

    # 将图像转换为浮点类型以避免溢出
    img_float = image.astype(np.float32) / 255.0
    
    # 计算图像的平均值
    mean = np.mean(img_float)
    
    # 按照 PyTorch 的方式调整对比度
    image = mean + contrast_factor * (img_float - mean)
    
    # image = np.clip(image, 0, 255)
    image = np.clip(image, 0, 1)
    # 在灰度图像上增加一个维度并复制
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    # 转换为浮点型并归一化
    # image = image.astype(np.float32) / 255.0

    # 归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_rgb = (image - mean) / std
    image_rgb = image_rgb.astype(np.float32)

    # 转换为 (C, H, W) 格式
    return image_rgb.transpose(2, 0, 1), img



args = parser.parse_arguments()


######################################### MODEL #########################################
model = network.GeoLocalizationNet(args)
model = model.to(args.device)
if args.separate_branch:
    model_db = copy.deepcopy(model)

if args.aggregation in ["netvlad", "crn"]:
    args.features_dim *= args.netvlad_clusters

if args.resume is not None:
    logging.info(f"Resuming model from {args.resume}")
    if args.separate_branch:
        model, model_db = util.resume_model_separate(args, model, model_db)
    else:
        model = util.resume_model(args, model)
# Enable DataParallel after loading checkpoint, otherwise doing it before
# would append "module." in front of the keys of the state dict triggering errors
model = torch.nn.DataParallel(model)
if args.separate_branch:
    model_db = torch.nn.DataParallel(model_db)

if args.pca_dim is None:
    pca = None
else:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim
    pca = util.compute_pca(
        args, model, full_features_dim)

######################################### DATASETS #########################################
queries_ds = datasets_ws.SingleBaseDataset(
    args, args.datasets_folder, args.dataset_name, "sample3_sample", True, False)  # loading_queries=False, use_h5_for_queries=True
logging.info(f"Queries set: {queries_ds}")

######################################### TEST on TEST SET #########################################

model = model.eval()


h5_file_path = args.database_array_path
logging.debug("Reading database features array")

## get database features , utms and positives
if os.path.exists(h5_file_path):
    logging.debug("loading Database feature and utms")
    # load local features and utms of database.
    with h5py.File(h5_file_path, 'r') as hf:
        database_features = hf['database_features'][:]
        database_utms = hf['database_utms'][:]

else:
    logging.debug("please extracting database features first.")

with torch.no_grad():

    file_list = sorted(os.listdir(args.img_folder))
    img_list = []
    total, positive, dis = 0, 0, 0
    for path in file_list:
        if img_check(path):
            img_list.append(path)
    for i in tqdm(range(len(img_list))):
        print('infer {}/{}'.format(i+1, len(img_list)), end='\r')

        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print("{} is not found", img_name)
            continue
        input_data, undistort = pre_process(img_path)
        input_data = np.expand_dims(input_data, 0)
        position, idx, distance = model(input_data)
        print("position: ", position)
        # t3 = time()

        # add difference
        pred_x, pred_y = position[0], position[1]
        real_x, real_y = np.float32(img_path.split("@")[1]), np.float32(img_path.split("@")[2])
        total += 1
        diff_position_x = abs(pred_x - real_x)
        diff_position_y = abs(pred_y - real_y)
        s = math.sqrt(diff_position_x * diff_position_x + diff_position_y * diff_position_y)
        dis += s
        if s < 128:
            positive += 1

        if args.img_show or args.img_save:
            print('\n\nIMG: {}'.format(img_name))
            img_p = cv2.imread(img_path)
            draw(img_p, position, distance)
            if args.get_base:
                base_img_path = "/media/bh/xujg/UAV-VisionLoc/data/new_dataset/database2.h5"
                img_base = data_get(base_img_path, idx)
                img_p = concat(img_p, img_base, undistort)

            if args.img_save:
                if s < 128:
                    result_path = os.path.join(args.save_path1, img_name)
                else:
                    result_path = os.path.join(args.save_path2, img_name)
                cv2.imwrite(result_path, img_p)
                print('Position result save to {}'.format(result_path))
            
            if args.img_show:
                cv2.imshow("full post process result", img_p)
                cv2.waitKeyEx(0)

    print("positive: ", positive)
    print("total: ", total)
    print("score: ", positive / total)
    print("average_dis: ", dis / total)



