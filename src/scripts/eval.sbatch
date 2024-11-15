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

# python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_thermalmapping_135 --datasets_folder ./datasets --infer_batch_size 16 --use_faiss_gpu --backbone $2 --conv_output_dim 4096 --add_bn --G_contrast

# python3 eval.py --resume='logs/default/'$1'/best_model.pth' --dataset_name=satellite_0_thermalmapping_135 --datasets_folder ./datasets --infer_batch_size 16 --prior_location_threshold=512 --use_faiss_gpu --backbone $2 --conv_output_dim 4096 --add_bn --G_contrast


# python3 eval.py --resume='/root/satellite-thermal-geo-localization-main/logs/default/satellite_0_thermalmapping_135-2024-09-26_01-13-22-a1841775-0fc1-4b5c-a12d-dee3dc4d11df/best_model.pth' --dataset_name=satellite_0_thermalmapping_135 --datasets_folder=/root/data --infer_batch_size 16 --use_faiss_gpu --conv_output_dim 4096 --add_bn --G_contrast

python3 eval.py --resume='/media/bh/xujg/UAV-VisionLoc/src/logs/default/dataset_undistort-2024-11-13_10-38-17-feb0f2d1-ed7c-4e91-be76-b533629f570b/best_model.pth' --dataset_name=dataset_undistort --datasets_folder=/media/bh/xujg/UAV-VisionLoc/data --infer_batch_size 16 --use_faiss_gpu --conv_output_dim 1024 --add_bn --val_positive_dist_threshold 256