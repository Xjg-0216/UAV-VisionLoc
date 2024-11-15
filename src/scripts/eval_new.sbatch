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

# python3 eval_new.py --resume='/media/bh/xujg/UAV-VisionLoc/src/logs/default/new_dataset-2024-10-16_12-59-59-1fefe63a-0785-47ec-856e-08af891e12fa/best_model.pth' --dataset_name=new_dataset --datasets_folder=/media/bh/xujg/UAV-VisionLoc/data --infer_batch_size 16 --use_faiss_gpu --conv_output_dim 4096 --add_bn --val_positive_dist_threshold 128 --G_contrast --database_array_path /media/bh/xujg/UAV-VisionLoc/data/database_array_1016_2.h5


python3 eval_new.py --resume='/media/bh/xujg/UAV-VisionLoc/src/logs/default/dataset_undistort-2024-11-13_10-38-17-feb0f2d1-ed7c-4e91-be76-b533629f570b/best_model.pth' --dataset_name=dataset_undistort --datasets_folder=/media/bh/xujg/UAV-VisionLoc/data --infer_batch_size 1 --use_faiss_gpu --conv_output_dim 1024 --add_bn --val_positive_dist_threshold 256 --database_array_path /media/bh/xujg/UAV-VisionLoc/data/database_array_1113.h5