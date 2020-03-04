#!/bin/bash
#SBATCH --job-name=pytorch_BLOCKATTENTION
#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32GB
##SBATCH --exclusive
#SBATCH -o /OSM/CBR/AF_WQ/source/ML/Log/Pytorch/Pytorch_BLOCKATTENTION__%a.txt

#module load python/3.6.1
#module load keras/2.2.4
#module load tensorflow/1.6.0-py36-gpu
module load cuda/9.0.176
module load pytorch/1.1.0-py36-cuda90



python Pytorch_Seq2Seq/main.py


