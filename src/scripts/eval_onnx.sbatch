#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --job-name=eval_thermal

eval "$(conda shell.bash hook)"
conda activate VTL

python3 onnx/eval_onnx.py --resume='/media/bh/xujg/UAV-VisionLoc/src/uvl.onnx' --dataset_name=data --datasets_folder=/media/bh/xujg/satellite-thermal-geo-localization-main/datasets --infer_batch_size 16 --use_faiss_gpu --conv_output_dim 1024 --add_bn --val_positive_dist_threshold 128 --database_array_path /media/bh/xujg/UAV-VisionLoc/data/database_array_1023_sample3_retrain.h5