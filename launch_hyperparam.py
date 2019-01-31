# create a script that does
import copy
import getpass
import os
import random
import numpy as np
import sys
import subprocess
import pdb
import time
import itertools
import argparse

### EXAMPLE
# python launch_hyperparam.py --data QM9 --num_times 4 --root_ckptdir /scratch/em3382/chemgeom/hyperparam_ckpts --root_eventdir /scratch/em3382/chemgeom/hyperparam_events/ --slurm_dir /scratch/em3382/chemgeom/slurm/

parser = argparse.ArgumentParser(description='Train network')

parser.add_argument('--data', type=str, default='QM9', choices=['COD', 'QM9', 'CSD'])
parser.add_argument('--root_ckptdir', type=str, default='./checkpoints/')
parser.add_argument('--root_eventdir', type=str, default='./events/')
parser.add_argument('--slurm_dir', type=str, default='./slurm/', help='where to write slurm output and err')
parser.add_argument('--num_times', type=int, default=4, help='number of times to launch an experiment')
parser.add_argument('--start', type=int, default=0)

args = parser.parse_args()

flags = [['--dim_h 25', '--dim_h 50'], ['--dim_f 50', '--dim_f 100'], ['--mpnn_steps 3', '--mpnn_steps 5']]
#flags = [['--dim_h 25', '--dim_h 50']] # for debugging

if __name__ == "__main__":

    options = itertools.product(*flags)
    FLAGS = "--data {}".format(args.data)
    for option in options:
        CURR_FLAGS = copy.deepcopy(FLAGS)
        CURR_FLAGS = CURR_FLAGS + " " + ' '.join(option)

        # launch slurm jobs one after another
        slurm_dep = None
        name = ''.join(option).replace(' ', '_').replace('--', '_')
        for i in range(args.start, args.num_times):
            ckptdir = os.path.join(args.root_ckptdir, args.data, "{}_run{}".format(name, i))
            eventdir = os.path.join(args.root_eventdir, args.data, "{}_run{}".format(name, i))

            CURR_SUB_FLAGS = copy.deepcopy(CURR_FLAGS)
            CURR_SUB_FLAGS += " --ckpt {} --eventdir {}".format(ckptdir, eventdir)

            # load from previously created checkpoint
            if i > 0:
                loaddir = os.path.join(os.path.join(args.root_ckptdir, args.data, "{}_run{}".format(name, i-1)), "neuralnet_model.ckpt")
                CURR_SUB_FLAGS += " --loaddir {}".format(loaddir)
            if i > args.start:
                # dependency flag for slurm
                dep_flag = "--dependency=afterany:{}".format(slurm_dep)
            else:
                dep_flag = ""

            # slurm out and err
            slurm_name = "{}_{}_run{}".format(args.data, name, i)
            slurm_out_path = os.path.join(args.slurm_dir, slurm_name + '.out')
            slurm_err_path = os.path.join(args.slurm_dir, slurm_name + '.err')

            my_env = os.environ.copy()
            my_env["FLAGS"] = "{}".format(CURR_SUB_FLAGS)
            my_env["CUDA_VISIBLE_DEVICES"] = "0"

            cmd = 'sbatch -o {} -e {} {} ./hyperparam.sbatch'.format(slurm_out_path, slurm_err_path, dep_flag)

            print (cmd)
            #pdb.set_trace()

            proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, shell=False, env=my_env)
            out, err = proc.communicate()
            slurm_dep = int(out.strip().split()[-1])

            print (out, err)
            print ("launched {}, slurm job id {}".format(i, slurm_dep))
            time.sleep(1.0)
