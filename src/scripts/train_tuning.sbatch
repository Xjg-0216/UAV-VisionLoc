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

python3 train.py --dataset_name=new_dataset --mining=partial --datasets_folder=/media/bh/xujg/UAV-VisionLoc/data --infer_batch_size 16 --train_batch_size 4 --lr 0.00004 --patience 100 --epochs_num 150 --use_faiss_gpu --conv_output_dim 4096 --add_bn --DA DANN_after --lambda_DA 0.1 --DA_only_positive --G_contrast --resume=/media/bh/xujg/UAV-VisionLoc/src/logs/default/satellite_0_thermalmapping_135-2024-09-22_01-03-17-0078b365-5242-465f-b59b-f2b181646211/best_model.pth