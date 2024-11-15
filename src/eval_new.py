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
from sklearn.neighbors import NearestNeighbors
import faiss
from utils.plotting import process_results_simulation

######################################### SETUP #########################################
# 设置 matplotlib 的日志级别
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

args = parser.parse_arguments()
start_time = datetime.now()
model_name = args.resume.split('/')[-2]
args.save_dir = join(
    "logs",
    "eval_new",
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
queries_ds = datasets_ws.SingleBaseDataset(
    args, args.datasets_folder, args.dataset_name, "test_queries_undistort", True, False)  # loading_queries=False, use_h5_for_queries=True
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
    # For database use "hard_resize", although it usually has no effect because database images have same resolution
    queries_ds.test_method = "hard_resize"
    queries_dataloader = DataLoader(
        dataset=queries_ds,
        num_workers=args.num_workers,
        batch_size=args.infer_batch_size,
        pin_memory=(args.device == "cuda"),
    )
    queries_features = np.empty(
        (len(queries_ds), args.features_dim), dtype="float32"
    )

    for inputs, indices in tqdm(queries_dataloader, ncols=100):

        features = model(inputs.to(args.device))
        features = features.cpu().numpy()
        queries_features[indices.numpy(), :] = features

    # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(database_utms)
    positives_per_query = knn.radius_neighbors(
        queries_ds.data_utms,
        radius=args.val_positive_dist_threshold,
        return_distance=False,
    )

    logging.debug("Calculating recalls")
    res = faiss.StandardGpuResources()
    faiss_index = faiss.GpuIndexFlatL2(res, args.features_dim)
    faiss_index.add(database_features)
    distances, predictions = faiss_index.search(
        queries_features, max(args.recall_values)
    )

    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / queries_ds.data_num * 100
    recalls_str = ", ".join(
        [f"R@{val}: {rec:.1f}" for val,
            rec in zip(args.recall_values, recalls)]
    )
    logging.info(f"{recalls_str}")
    if args.use_best_n > 0:

        samples_to_be_used = args.use_best_n
        error_m = []
        position_m = []
        for query_index in tqdm(range(len(predictions))):
            distance = distances[query_index]
            prediction = predictions[query_index]
            sort_idx = np.argsort(distance)
            if args.use_best_n == 1:
                best_position = database_utms[prediction[sort_idx[0]]]
            else:
                if distance[sort_idx[0]] == 0:
                    best_position = database_utms[prediction[sort_idx[0]]]
                else:
                    mean = distance[sort_idx[0]]
                    sigma = distance[sort_idx[0]] / distance[sort_idx[-1]]
                    X = np.array(distance[sort_idx[:samples_to_be_used]]).reshape((-1,))
                    weights = np.exp(-np.square(X - mean) / (2 * sigma ** 2))  # gauss
                    weights = weights / np.sum(weights)

                    x = y = 0
                    for p, w in zip(database_utms[prediction[sort_idx[:samples_to_be_used]]], weights.tolist()):
                        y += p[0] * w
                        x += p[1] * w
                    best_position = (y, x)
            actual_position = queries_ds.data_utms[query_index]
            error = np.linalg.norm((actual_position[0]-best_position[0], actual_position[1]-best_position[1]))
            error_m.append(error)
            position_m.append(actual_position)
        process_results_simulation(error_m, args.save_dir)


