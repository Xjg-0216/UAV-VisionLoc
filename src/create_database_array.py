import os
import sys
import torch
import parser
import logging
import sklearn
from os.path import join
from datetime import datetime
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
import copy
from torch.utils.data import DataLoader
import test
import util
import commons
import datasets_ws
from model import network
from tqdm import tqdm
import h5py

######################################### SETUP #########################################
args = parser.parse_arguments()
start_time = datetime.now()
model_name = args.resume.split('/')[-2]
args.save_dir = join(
    "logs",
    "create_database",
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
database_ds = datasets_ws.SingleBaseDataset(
    args, args.datasets_folder, args.dataset_name, "", False)
logging.info(f"Train database set: {database_ds}")

######################################### TEST on TEST SET #########################################

# recalls, recalls_str = test.test(args, test_ds, model, test_method = args.test_method, pca = pca, visualize=args.visual_all)
model = model.eval()


h5_file_path = args.database_array_path

with torch.no_grad():
    logging.debug("Extracting database features")
    # For database use "hard_resize", although it usually has no effect because database images have same resolution
    database_ds.test_method = "hard_resize"
    database_dataloader = DataLoader(
        dataset=database_ds,
        num_workers=args.num_workers,
        batch_size=args.infer_batch_size,
        pin_memory=(args.device == "cuda"),
    )
    all_features = np.empty(
        (len(database_ds), args.features_dim), dtype="float32"
    )

    for inputs, indices in tqdm(database_dataloader, ncols=100):

        features = model(inputs.to(args.device))
        features = features.cpu().numpy()
        all_features[indices.numpy(), :] = features
        torch.cuda.empty_cache()  # 清空未使用的显存

    # 检查 HDF5 文件是否已经存在
    if not os.path.exists(h5_file_path):
        with h5py.File(h5_file_path, 'w') as hf:
            hf.create_dataset('database_features', data=all_features)
            hf.create_dataset('database_utms', data=database_ds.data_utms)

            logging.debug(f"database_features size: {all_features.shape}")
            logging.debug(f"database_utms size: {database_ds.data_utms.shape}")
    else:
        logging.debug(f"HDF5 file already exists at {h5_file_path}, skipping saving.")

    del all_features

