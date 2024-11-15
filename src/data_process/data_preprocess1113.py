'''
Descripttion: 
Author: xujg
version: 
Date: 2024-10-29 16:29:34
LastEditTime: 2024-11-12 15:30:22
'''

import os
import cv2
import numpy as np
from tqdm import tqdm



def crop_center(image, crop_width, crop_height):
    """
    Crop the center of the image to the specified width and height.
    """
    h, w = image.shape[:2]
    start_x = w // 2 - crop_width // 2
    start_y = h // 2 - crop_height // 2
    return image[start_y:start_y + crop_height, start_x:start_x + crop_width]

def get_M(theta, phi, gamma, dx, dy, dz, f):
    
    # Projection 2D -> 3D matrix
    A1 = np.array([ [1, 0, -640/2],
                    [0, 1, -512/2],
                    [0, 0, 1],
                    [0, 0, 1]])
    
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([ [1, 0, 0, 0],
                    [0, np.cos(theta), -np.sin(theta), 0],
                    [0, np.sin(theta), np.cos(theta), 0],
                    [0, 0, 0, 1]])
    
    RY = np.array([ [np.cos(phi), 0, -np.sin(phi), 0],
                    [0, 1, 0, 0],
                    [np.sin(phi), 0, np.cos(phi), 0],
                    [0, 0, 0, 1]])
    
    RZ = np.array([ [np.cos(gamma), -np.sin(gamma), 0, 0],
                    [np.sin(gamma), np.cos(gamma), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    # Composed rotation matrix with (RX, RY, RZ)
    R = np.dot(np.dot(RX, RY), RZ)

    # Translation matrix
    T = np.array([  [1, 0, 0, dx],
                    [0, 1, 0, dy],
                    [0, 0, 1, dz],
                    [0, 0, 0, 1]])

    # Projection 3D -> 2D matrix
    A2 = np.array([ [f, 0, 640/2, 0],
                    [0, f, 512/2, 0],
                    [0, 0, 1, 0]])

    # Final transformation matrix
    return np.dot(A2, np.dot(T, np.dot(R, A1)))

def ImageTransformer(img_path, isdistort=True):
    image = cv2.imread(img_path)
    height, width = image.shape[0], image.shape[1]
    at = list(map(np.float32, img_path.split("@")[3:7]))
    yaw, pitch, roll, H = at[0], at[1], at[2], at[3]
    K = np.array([[776.65776764, 0, 320], [0, 776.65776764, 243], [0, 0, 1]])
    dist = np.array([-0.36861635328211733720, 0.16758271710507155472, 0.00313679920675880367, -0.00006878833592730838, -0.16590988270296352924])
    # self.temp = cv2.undistort(self.image, K, dist)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, image.shape[1::-1], 1, image.shape[1::-1])
    if isdistort:
        image = cv2.undistort(image, K, dist, None, new_K)
    d = 776.65776764
    focal = d / (2 * np.sin(yaw) if np.sin(yaw) != 0 else 1)
    # dz = self.focal * (1.3 * self.H / 452)
    dz = focal * (1.3 * 452 / H)
    mat = get_M(roll, pitch, yaw, 0, 0, dz, focal)
    return cv2.warpPerspective(image, mat, (width, height))



img_root = '/media/bh/xujg/UAV-VisionLoc/data/dataset_undistort/test_exp3'
outimg_root = '/media/bh/xujg/UAV-VisionLoc/data/dataset_undistort/test_exp31'
os.makedirs(outimg_root, exist_ok=True)
imgs_list = os.listdir(img_root)
print(len(imgs_list))
for img_name in tqdm(imgs_list):
    img_old = os.path.join(img_root, img_name)
    output = ImageTransformer(img_old)
    output = crop_center(output, 512, 512)
    parts = img_old.split('@')
    # img_new = os.path.join(outimg_root, f"@{parts[1]}@{parts[2]}@{parts[7]}@.jpg")
    img_new = os.path.join(outimg_root, f"@{parts[1]}@{parts[2]}@.jpg")
    cv2.imwrite(img_new, output)

# # Instantiate the class
# output = ImageTransformer(img_path)
# # Make output dir
# parts = img_path.split('@')
# new_img_path = os.path.join(parts[0], f"@{parts[1]}@{parts[2]}@{parts[7]}@.jpg")
# cv2.imwrite(new_img_path, output)


