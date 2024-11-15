"""
With this script you can evaluate checkpoints or test models from two popular
landmark retrieval github repos.
The first is https://github.com/naver/deep-image-retrieval from Naver labs, 
provides ResNet-50 and ResNet-101 trained with AP on Google Landmarks 18 clean.
$ python eval.py --off_the_shelf=naver --l2=none --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048

The second is https://github.com/filipradenovic/cnnimageretrieval-pytorch from
Radenovic, provides ResNet-50 and ResNet-101 trained with a triplet loss
on Google Landmarks 18 and sfm120k.
$ python eval.py --off_the_shelf=radenovic_gldv1 --l2=after_pool --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048
$ python eval.py --off_the_shelf=radenovic_sfm --l2=after_pool --backbone=resnet101conv5 --aggregation=gem --fc_output_dim=2048

Note that although the architectures are almost the same, Naver's
implementation does not use a l2 normalization before/after the GeM aggregation,
while Radenovic's uses it after (and we use it before, which shows better
results in VG)
"""

import os
import sys
import torch
import parser
import logging
import sklearn
from os.path import join
from datetime import datetime
from torch.utils.model_zoo import load_url
from google_drive_downloader import GoogleDriveDownloader as gdd
import copy
import cv2
import test
import util
import commons
import datasets_ws
from model import network
import faiss
import h5py
import numpy as np
from tqdm import tqdm




def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


def draw(img, position, distance):
    cv2.putText(img, '{},{}'.format(position.astype("int"), distance.astype("float")),
                (0, 512 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


class TorchInfer:
    def __init__(self, args):
        self.args = args
        self.load_local_database()
        res = faiss.StandardGpuResources()
        self.faiss_index = faiss.GpuIndexFlatL2(res, args.features_dim)
        self.faiss_index.add(self.database_features)
        self.model= torch.jit.load(args.model_path)
        self.model.eval()


    def model_inference(self, input_data):

        # t4 = time()
        outputs = self.model(input_data)
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
    return adjusted_image

def crop_center(image, crop_width, crop_height):
    """
    Crop the center of the image to the specified width and height.
    """
    h, w = image.shape[:2]
    start_x = w // 2 - crop_width // 2
    start_y = h // 2 - crop_height // 2
    return image[start_y:start_y + crop_height, start_x:start_x + crop_width]

def pre_process(img_path, contrast_factor=3):

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



    

            



OFF_THE_SHELF_RADENOVIC = {
    "resnet50conv5_sfm": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth",
    "resnet101conv5_sfm": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth",
    "resnet50conv5_gldv1": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth",
    "resnet101conv5_gldv1": "http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth",
}

OFF_THE_SHELF_NAVER = {
    "resnet50conv5": "1oPtE_go9tnsiDLkWjN4NMpKjh-_md1G5",
    "resnet101conv5": "1UWJGDuHtzaQdFhSMojoYVQjmCXhIwVvy",
}

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
model_name = args.resume.split('/')[-2]
args.save_dir = join(
    "test",
    args.save_dir,
    model_name,
    f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
)
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

######################################### MODEL #########################################
model = network.GeoLocalizationNet(args)
model = model.to(args.device)
if args.separate_branch:
    model_db = copy.deepcopy(model)

if args.aggregation in ["netvlad", "crn"]:
    args.features_dim *= args.netvlad_clusters

if args.off_the_shelf.startswith("radenovic") or args.off_the_shelf.startswith("naver"):
    if args.off_the_shelf.startswith("radenovic"):
        pretrain_dataset_name = args.off_the_shelf.split("_")[
            1
        ]  # sfm or gldv1 datasets
        url = OFF_THE_SHELF_RADENOVIC[f"{args.backbone}_{pretrain_dataset_name}"]
        state_dict = load_url(url, model_dir=join(
            "data", "off_the_shelf_nets"))
    else:
        # This is a hacky workaround to maintain compatibility
        sys.modules["sklearn.decomposition.pca"] = sklearn.decomposition._pca
        zip_file_path = join("data", "off_the_shelf_nets",
                             args.backbone + "_naver.zip")
        if not os.path.exists(zip_file_path):
            gdd.download_file_from_google_drive(
                file_id=OFF_THE_SHELF_NAVER[args.backbone],
                dest_path=zip_file_path,
                unzip=True,
            )
        if args.backbone == "resnet50conv5":
            state_dict_filename = "Resnet50-AP-GeM.pt"
        elif args.backbone == "resnet101conv5":
            state_dict_filename = "Resnet-101-AP-GeM.pt"
        state_dict = torch.load(
            join("data", "off_the_shelf_nets", state_dict_filename))
    state_dict = state_dict["state_dict"]
    model_keys = model.state_dict().keys()
    renamed_state_dict = {k: v for k, v in zip(
        model_keys, state_dict.values())}
    model.load_state_dict(renamed_state_dict)
elif args.resume is not None:
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
path_local_database = "/media/bh/xujg/satellite-thermal-geo-localization-main/save/database_array5.h5"
if os.path.exists(path_local_database):
    print("loading Database feature and utms ...")
            # load local features and utms of database.
    with h5py.File(path_local_database, 'r') as hf:
        database_features = hf['database_features'][:]
        database_utms = hf['database_utms'][:]

else:
    print("please extracting database features first.")
    sys.exit()
res = faiss.StandardGpuResources()
faiss_index = faiss.GpuIndexFlatL2(res, 4096)
faiss_index.add(database_features)

img_folder = '/media/bh/xujg/satellite-thermal-geo-localization-main/datasets/data/tmp'
file_list = sorted(os.listdir(img_folder))
img_list = []
for path in file_list:
    if img_check(path):
        img_list.append(path)

# run test
for i in tqdm(range(len(img_list))):
    print('infer {}/{}'.format(i+1, len(img_list)), end='\r')

    img_name = img_list[i]
    img_path = os.path.join(img_folder, img_name)
    if not os.path.exists(img_path):
        print("{} is not found", img_name)
        continue
    input_data, undistort = pre_process(img_path)
    input_data = np.expand_dims(input_data, 0)
    input_data = torch.from_numpy(input_data)
    output = model(input_data.to(args.device))
    output = output.cpu()
    output = output.detach().numpy()
    distances, predictions = faiss_index.search(
            output, 1)
    distance = distances[0]
    prediction = predictions[0]
    sort_idx = np.argsort(distance)
    position = database_utms[prediction[sort_idx[0]]]
    idx = prediction[sort_idx[0]]
    print("position: ", position)

    print('\n\nIMG: {}'.format(img_name))
    img_p = cv2.imread(img_path)
    draw(img_p, position, distance)
    base_img_path = "/media/bh/xujg/satellite-thermal-geo-localization-main/datasets/data/database92.h5"
    img_base = data_get(base_img_path, idx)
    img_p = concat(img_p, img_base, undistort)
    save_path = "/media/bh/xujg/satellite-thermal-geo-localization-main/result_sample1"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    result_path = os.path.join(save_path, img_name)
    cv2.imwrite(result_path, img_p)
    print('Position result save to {}'.format(result_path))