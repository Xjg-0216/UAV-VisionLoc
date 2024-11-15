import h5py
import cv2



def data_get(file_path, indice):
    with h5py.File(file_path, 'r') as f:
        image_data = f['image_data'][indice]
        image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    return image_data

idx = [13541, 17223, 12713, 3878, 1138, 8045, 13430, 1807, 16495, 12714, 3880, 3830, 10632, 14126, 7081, 8606, 17224]
base_img_path = "/media/bh/xujg/UAV-VisionLoc/data/new_dataset/database1022_sample3.h5"

for i in idx:
    img_base = data_get(base_img_path, i)
    cv2.imwrite(f"{i}.jpg", img_base)