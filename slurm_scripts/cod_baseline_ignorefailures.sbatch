#!/bin/bash

# Generic job script for all experiments.

#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH -t72:00:00

/home/em3382/miniconda3/bin/python baseline.py --data COD --setting ignorefailures --num-total-samples 100 --num-parallel-samples 10 --num-threads 10 --savepermol --savedir /scratch/em3382/molgeom/baseline_ignorefailures_100samples
