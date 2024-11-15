#!/bin/bash 
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=8               # uses 1 compute core per task
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --job-name=train_thermal
#SBATCH --output=train_thermal.out

eval "$(conda shell.bash hook)"
conda activate VTL

# python3 train.py --dataset_name=new_dataset --mining=partial --datasets_folder=/media/bh/xujg/UAV-VisionLoc/data --infer_batch_size 16 --train_batch_size 4 --lr 0.0001 --patience 100 --epochs_num 100 --use_faiss_gpu --conv_output_dim 4096 --add_bn --DA DANN_after --lambda_DA 0.1 --DA_only_positive --G_contrast

python3 train.py --dataset_name=dataset_undistort --mining=partial --datasets_folder=/media/bh/xujg/UAV-VisionLoc/data --infer_batch_size 16 --train_batch_size 4 --lr 0.0001 --patience 100 --epochs_num 100 --use_faiss_gpu --conv_output_dim 4096 --add_bn --DA DANN_after --lambda_DA 0.1 --DA_only_positive --train_positives_dist_threshold 256 --val_positive_dist_threshold 256 --G_contrast