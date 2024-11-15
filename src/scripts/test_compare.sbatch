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

# 遍历 CASE_NUM 从 1 到 12
for CASE_NUM in {2..2}; do
    # 无预处理
    # python3 test_onnx_compare.py \
    #     --img_folder /media/bh/xujg/UAV-VisionLoc/data/sample_case/case${CASE_NUM} \
    #     --save_path1 /media/bh/xujg/UAV-VisionLoc/data/sample_case/case${CASE_NUM}/out_no_preprocess \
    #     --isdistort False

    # 预处理模式 0
    python3 onnx/test_onnx_compare.py \
        --img_folder /media/bh/xujg/UAV-VisionLoc/data/sample_case/case${CASE_NUM} \
        --save_path1 /media/bh/xujg/UAV-VisionLoc/data/sample_case/case${CASE_NUM}/out_preprocess0 \
        --isdistort True

    # 预处理模式并且使用 tran 参数
    # python3 test_onnx_compare.py \
    #     --img_folder /media/bh/xujg/UAV-VisionLoc/data/sample_case/case${CASE_NUM} \
    #     --save_path1 /media/bh/xujg/UAV-VisionLoc/data/sample_case/case${CASE_NUM}/out_preprocess \
    #     --isdistort True --tran True
done