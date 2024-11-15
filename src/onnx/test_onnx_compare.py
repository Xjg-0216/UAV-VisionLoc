import os
import cv2
import sys
import argparse
import h5py
import faiss
import numpy as np
from time import time
from tqdm import tqdm
import onnxruntime
import math


def calculate_image_coordinates(intrinsics, object_point):
    object_point_homogeneous = np.array([object_point[0], object_point[1], object_point[2], 1])
    image_point_homogeneous = intrinsics @ object_point_homogeneous[:3]
    u = image_point_homogeneous[0] / image_point_homogeneous[2]
    v = image_point_homogeneous[1] / image_point_homogeneous[2]
    return u, v


def calc_dxdy(at):


    # yaw = at[0]
    # pitch = at[2]
    # roll = at[1]
    yaw = at[0]
    pitch = at[2]
    roll = -at[1] if at[1] > 0 else at[1]
    height = at[3]

    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])

    R_roll = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    # 综合旋转矩阵
    R = R_roll @ R_pitch @ R_yaw
    # R = R_yaw @ R_pitch @ R_roll
    # R = R_pitch @ R_roll @ R_yaw
    # R = R_yaw @ R_roll @ R_pitch
    # R = R_pitch @ R_yaw @ R_roll
    # R = R_roll @ R_yaw @ R_pitch
    intrinsics = np.array([[776.65776764287693367805, 0, 320.17987354576234793058],
    [0, 776.65776764287693367805, 243.12414146131357028935],
    [0, 0, 1]])




    # 无倾斜情况下的物体坐标
    initial_point = np.array([0, 0, -height])

    # 倾斜情况下的物体坐标
    tilted_point = R @ initial_point
    print(tilted_point)

    # 计算像素移动
    # u_initial, v_initial = calculate_image_coordinates(intrinsics, initial_point)
    u_tilted, v_tilted = calculate_image_coordinates(intrinsics, tilted_point)
    return u_tilted, v_tilted


def euler_to_rotation_matrix(yaw, pitch, roll):
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y, R_x))


def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


def draw(img, position, distance):
    cv2.putText(img, '{},{}'.format(position.astype("int"), distance.astype("float")),
                (0, 512 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


class ONNXInfer:
    def __init__(self, args):
        self.args = args
        self.load_local_database()
        res = faiss.StandardGpuResources()
        self.faiss_index = faiss.GpuIndexFlatL2(res, args.features_dim)
        self.faiss_index.add(self.database_features)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # self.session = onnxruntime.InferenceSession(args.path_onnx, sess_options=session_options, providers=providers)
        self.session = onnxruntime.InferenceSession(args.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name


    def model_inference(self, input_data):

        # t4 = time()
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        # t5 = time()
        position, idx, distance = self.post_process(outputs[0])
        # t6 = time()
        # print("inference_time: {:.4f} s".format(t5-t4) )
        # print("post_time: {:.4f} s".format(t6-t5) )
        return position, idx, distance

    def post_process(self, result):
        distances, predictions = self.faiss_index.search(
            result, max(self.args.recall_values)
            )
        distance = distances[0]
        prediction = predictions[0]
        sort_idx = np.argsort(distance)
        if self.args.use_best_n == 1:
            best_position = self.database_utms[prediction[sort_idx[0]]]
        else:
            if distance[sort_idx[0]] == 0:
                best_position = self.database_utms[prediction[sort_idx[0]]]
            else:
                mean = distance[sort_idx[0]]
                sigma = distance[sort_idx[0]] / distance[sort_idx[-1]]
                X = np.array(distance[sort_idx[:self.args.use_best_n]]).reshape((-1,))
                weights = np.exp(-np.square(X - mean) / (2 * sigma ** 2))  # gauss
                weights = weights / np.sum(weights)

                x = y = 0
                for p, w in zip(self.database_utms[prediction[sort_idx[:self.args.use_best_n]]], weights.tolist()):
                    y += p[0] * w
                    x += p[1] * w
                best_position = (y, x)
        return best_position, prediction[sort_idx[0]], distance[0]
    
    def load_local_database(self):
        if os.path.exists(self.args.path_local_database):
            print("loading Database feature and utms ...")
            # load local features and utms of database.
            with h5py.File(self.args.path_local_database, 'r') as hf:
                self.database_features = hf['database_features'][:]
                self.database_utms = hf['database_utms'][:]

        else:
            print("please extracting database features first.")
            sys.exit()


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

def distort(img_path, tran):
    img = cv2.imread(img_path)
    # Image dimensions
    height, width = img.shape[:2]
    if len(img_path.split('@')) < 5:
        yaw, pitch, roll = 0, 0, 0
    else:
        at = list(map(np.float32, img_path.split("@")[3:7]))   
        yaw = at[0]
        pitch = at[1]
        roll = at[2]
        if tran:
            u_tilted, v_tilted = calc_dxdy(at)
            M = np.array(np.float32([[1, 0, 320 - u_tilted], [0, 1, 243 - v_tilted]]))
            img = cv2.warpAffine(img, M, (width, height))
    # Get the rotation matrix
    R = euler_to_rotation_matrix(roll, pitch, yaw)

    # Define the source points (four corners of the image)
    src_points = np.array([[0, 0],
                           [width - 1, 0],
                           [width - 1, height - 1],
                           [0, height - 1]], dtype='float32')

    # Apply the rotation to the source points
    # dst_points = np.dot(src_points - np.array([width / 2, height / 2]), R[:2, :2].T) + np.array([width / 2, height / 2])
    dst_points = np.dot(src_points - np.array([320, 243]), R[:2, :2].T) + np.array([320, 243])
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

def pre_process(img_path, contrast_factor, tran, isdistort):

    # Read image using OpenCV
    if not isdistort:
        img = cv2.imread(img_path)
    # distort
    else:
        img = distort(img_path, tran)
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

    # return image_rgb


def data_get(file_path, indice):
    with h5py.File(file_path, 'r') as f:
        image_data = f['image_data'][indice]
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    return image_data

def concat(img1, img2, img3):

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    h3, w3, _ = img3.shape
    new_height = max(h1, h2)

    if h1 < new_height:
        top_pad = (new_height - h1) // 2
        bottom_pad = new_height - h1 - top_pad
        img1 = cv2.copyMakeBorder(img1, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    if h2 < new_height:
        top_pad = (new_height - h2) // 2
        bottom_pad = new_height - h2 - top_pad
        img2 = cv2.copyMakeBorder(img2, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    if h3 < new_height:
        top_pad = (new_height - h3) // 2
        bottom_pad = new_height - h3 - top_pad
        img3 = cv2.copyMakeBorder(img3, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    result = cv2.hconcat([img1, img2, img3])
    return result

def concat2(img1, img2):


    result = cv2.hconcat([img1, img2])
    return result



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, default= "/media/bh/xujg/UAV-VisionLoc/src/uvl.onnx", help='model path, could be .pt or .rknn file')
    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    parser.add_argument('--img_save', action='store_true', default=True, help='save the result')
    parser.add_argument('--get_base', action='store_true', default=True, help='save the result')
    parser.add_argument('--save_path1', default="/media/bh/xujg/UAV-VisionLoc/data/sample_case/case4/out_no_preprocess", help='save the result')
    parser.add_argument('--tran', default=False, help='move point')
    parser.add_argument('--isdistort', default=True, help='image distorted')
    # data params
    parser.add_argument('--img_folder', type=str, default='/media/bh/xujg/UAV-VisionLoc/data/sample_case/case4', help='img folder path')
    parser.add_argument('--path_local_database', type=str, default='/media/bh/xujg/UAV-VisionLoc/data/database_array_1023_sample3_retrain.h5', help='load local features and utms of database')

    parser.add_argument(
        "--features_dim",
        type=int,
        default=1024,
        help="NetVLAD output dims.",
    )
    # retrieval params
    parser.add_argument(
        "--recall_values",
        type=int,
        default=[1, 5, 10, 20],
        nargs="+",
        help="Recalls to be computed, such as R@5.",
    )
    parser.add_argument(
        "--use_best_n",
        type=int,
        default=1,
        help="Calculate the position from weighted averaged best n. If n = 1, then it is equivalent to top 1"
    )

    args = parser.parse_args()

    if args.img_save:
        if not os.path.exists(args.save_path1):
            os.mkdir(args.save_path1)
        # if not os.path.exists(args.save_path2):
        #     os.mkdir(args.save_path2)
    onnx_engine = ONNXInfer(args)
    

    file_list = sorted(os.listdir(args.img_folder))
    img_list = []
    for path in file_list:
        if img_check(path):
            img_list.append(path)

    total, positive, dis = 0, 0, 0
    # run test
    for i in tqdm(range(len(img_list))):
        print('infer {}/{}'.format(i+1, len(img_list)), end='\r')   

        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print("{} is not found", img_name)
            continue
        t1 = time()
        input_data, undistort = pre_process(img_path, 1, args.tran, args.isdistort)
        t2 = time()
        input_data = np.expand_dims(input_data, 0)
        position, idx, distance = onnx_engine.model_inference(input_data)
        print("pre_time: {:.4f} s".format(t2-t1) )
        print("position: ", position)
        # t3 = time()

        # add difference
        pred_x, pred_y = position[0], position[1]
        print(img_path)
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
                base_img_path = "/media/bh/xujg/UAV-VisionLoc/data/new_dataset/database1022_sample3.h5"
                img_base = data_get(base_img_path, idx)
                # img_p = concat(img_p, img_base, undistort)
                img_p = concat2(img_p, img_base)

            if args.img_save:
                # if s < 128:
                result_path = os.path.join(args.save_path1, img_name)
                # else:
                #     result_path = os.path.join(args.save_path2, img_name)
                cv2.imwrite(result_path, img_p)
                print('Position result save to {}'.format(result_path))
            
            if args.img_show:
                cv2.imshow("full post process result", img_p)
                cv2.waitKeyEx(0)

    # print("positive: ", positive)
    # print("total: ", total)
    # print("score: ", positive / total)
    # print("average_dis: ", dis / total)
