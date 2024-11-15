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


python3 onnx/export_onnx.py --resume='/media/bh/xujg/UAV-VisionLoc/src/logs/default/dataset_undistort-2024-11-13_10-38-17-feb0f2d1-ed7c-4e91-be76-b533629f570b/best_model.pth' --dataset_name=data --datasets_folder=/media/bh/xujg/satellite-thermal-geo-localization-main/datasets --infer_batch_size 16 --use_faiss_gpu --conv_output_dim 1024 --add_bn --val_positive_dist_threshold 128 --G_contrast --database_array_path /media/bh/xujg/UAV-VisionLoc/data/database_array.h5