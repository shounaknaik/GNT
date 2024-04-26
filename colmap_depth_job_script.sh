#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -J colmap_depth
#SBATCH --output=./slurm_outputs/colmap_depth%j.out



#2. Load Cuda12.1
module load cuda12.1/toolkit

#3. Load Cudnn 
# module load cudnn8.5-cuda11.7/8.5.0.96 

conda activate nerf

python3 train.py --config configs/gnt_with_depth.txt --train_scenes chair --eval_scenes chair --use_colmap_depth


