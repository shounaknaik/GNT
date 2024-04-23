#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -J 
#SBATCH --output=slurm_outputs/colmap_depth%j.out



#2. Load Cuda11.7 
module load cuda11.7/toolkit/11.7.1 

#3. Load Cudnn 
module load cudnn8.5-cuda11.7/8.5.0.96 

conda activate nerf
python3 train.py --config configs/gnt_plain.txt --train_scenes chair --eval_scenes chair


