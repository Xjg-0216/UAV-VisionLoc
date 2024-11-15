

from utils.plotting import process_results_simulation
import os
import logging
from os.path import join
from datetime import datetime
import commons
from tqdm import tqdm
import h5py
import numpy as np
import faiss
import time
import parser
import argparse
from PIL import Image, ImageDraw, ImageFont
import datasets_ws
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
import onnxruntime

# 设置 matplotlib 的日志级别
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

class ONNXInfer(object):


    def __init__(self, args):
        self.args = args
        session_options = onnxruntime.SessionOptions()
        # 设置线程数
        # session_options.intra_op_num_threads = 4
        # session_options.inter_op_num_threads = 4
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # self.session = onnxruntime.InferenceSession(args.path_onnx, sess_options=session_options, providers=providers)
        self.session = onnxruntime.InferenceSession(args.resume, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def inference(self, inp_data: np.ndarray):


        results = self.session.run([self.output_name], {self.input_name: inp_data})

        return results[0]


def output_img(path, pil_im, position, save_dir):
    
    im_name = path.split("/")[-1]
    save_path = os.path.join(save_dir, im_name)

    draw = ImageDraw.Draw(pil_im)
    text = str(position)
    text_width = draw.textlength(text)
    width, height = pil_im.size
    x = width - text_width - 20 
    y = height- 20


    draw.text((x, y), text, fill=(255, 0, 0, 0)) 

    pil_im.save(save_path)





if __name__ == "__main__":


    ####################################SETUP##################################
    args = parser.parse_arguments()
    path_local_database = args.database_array_path
    features_dim = args.conv_output_dim


    start_time = datetime.now()
    args.save_dir = join(
        "test",
        "onnx",
        f"{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    commons.setup_logging(args.save_dir)
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")

    ################################## INFERENCE ################################

    logging.info(f"Loading vtl.onnx for ONNX inference...")

    onnx_engine = ONNXInfer(args)


    queries_dataset = datasets_ws.SingleBaseDataset(args, args.datasets_folder, args.dataset_name, "val", True, True)
    logging.info(f"Queries set:  {queries_dataset}")





    queries_features = np.empty((queries_dataset.data_num, features_dim), dtype="float32")
    

    for inputs, indices in tqdm(queries_dataset, ncols=100):

        inputs = inputs.numpy()
        inputs = np.expand_dims(inputs, 0)
        features = onnx_engine.inference(inputs)
        # queries_features[indices.numpy(), :] = features
        queries_features[indices, :] = features


    logging.info(f"Final feature dim: {queries_features.shape[1]}")

    ## get database features , utms and positives
    if os.path.exists(path_local_database):
        logging.debug("loading Database feature and utms")
        # load local features and utms of database.
        with h5py.File(path_local_database, 'r') as hf:
            database_features = hf['database_features'][:]
            database_utms = hf['database_utms'][:]

    else:
        logging.debug("please extracting database features first.")


    # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 25 meters)
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(database_utms)
    positives_per_query = knn.radius_neighbors(
        queries_dataset.data_utms,
        radius=args.val_positive_dist_threshold,
        return_distance=False,
    )

    logging.debug("Calculating recalls")
    res = faiss.StandardGpuResources()
    faiss_index = faiss.GpuIndexFlatL2(res, features_dim)
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
    recalls = recalls / queries_dataset.data_num * 100
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
            actual_position = queries_dataset.data_utms[query_index]
            error = np.linalg.norm((actual_position[0]-best_position[0], actual_position[1]-best_position[1]))
            error_m.append(error)
            position_m.append(actual_position)
        process_results_simulation(error_m, args.save_dir)

