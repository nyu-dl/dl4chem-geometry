#!/bin/bash

# Generic job script for all experiments.

#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH -t168:00:00

/home/em3382/miniconda3/bin/python baseline.py --data COD --num-total-samples 100 --num-parallel-samples 20 --num-threads 20 --savepermol --savedir /scratch/em3382/chemgeom/baselines_default/
