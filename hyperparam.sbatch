#!/bin/bash

# Generic job script for all experiments.

#SBATCH --gres=gpu:1
#SBATCH --partition=p100_4,v100_sxm2_4,v100_pci_2,k80_4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t168:00:00

module load cuda/9.0.176
module load cudnn/9.0v7.0.5

# Run.
source activate ~/anaconda3/envs/dl4chem-gpu
cd /scratch/onm217/dl4chem-geometry
export IFS=';'
for SUB_FLAGS in $FLAGS
do
	unset IFS
  python PredX_train.py $SUB_FLAGS &
done

wait
