from __future__ import print_function

import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import sys, gc, os
import PredX_MPNN as MPNN
import sparse
import argparse
import getpass
from test_tube import HyperOptArgumentParser, Experiment
from test_tube.hpc import SlurmCluster

def data_path():
    """Path to data depending on user launching the script"""
    if getpass.getuser() == "mansimov":
        return "/misc/kcgscratch1/ChoGroup/mansimov/seokho_drive_datasets/"
    else:
        return "./"

def train(args, exp=None):
    model = MPNN.Model(args.data, n_max, dim_node, dim_edge, dim_h, dim_f, \
                        batch_size, val_num_samples, \
                        mpnn_steps=args.mpnn_steps, alignment_type=args.alignment_type, tol=args.tol,\
                        use_X=args.use_X, use_R=args.use_R, \
                        virtual_node=args.virtual_node, seed=args.seed, \
                        refine_steps=args.refine_steps)

    #if args.loaddir != None:
    #    model.saver.restore(model.sess, args.loaddir)

    with model.sess:
        if args.test:
            if args.use_val:
                model.test(D1_val, D2_val, D3_val, D4_val, D5_val, molsup_val, \
                            load_path=args.loaddir, tm_v=tm_val, debug=args.debug, \
                            savepred_path=args.savepreddir)
            else:
                model.test(D1_tst, D2_tst, D3_tst, D4_tst, D5_tst, molsup_tst, \
                            load_path=args.loaddir, tm_v=tm_tst, debug=args.debug, \
                            savepred_path=args.savepreddir)
        else:
            model.train(D1_trn, D2_trn, D3_trn, D4_trn, D5_trn, molsup_trn, \
                        D1_val, D2_val, D3_val, D4_val, D5_val, molsup_val, \
                        load_path=args.loaddir, save_path=save_path, event_path=event_path, \
                        tm_trn=tm_trn, tm_val=tm_val, \
                        w_reg=args.w_reg, \
                        debug=args.debug, exp=exp)

def search_train(args):
    exp = Experiment(
        # Location to save the metrics.
        save_dir=args.ckptdir
    )
    exp.argparse(args)
    train(args)
    exp.save()



if __name__ == '__main__':

    hyperparameter_search = True
    if hyperparameter_search:
        parser = HyperOptArgumentParser(strategy='random_search')
    else:
        parser = argparse.ArgumentParser(description='Train network')

    parser.add_argument('--data', type=str, default='QM9', choices=['COD', 'QM9'])
    parser.add_argument('--ckptdir', type=str, default='./checkpoints/')
    parser.add_argument('--eventdir', type=str, default='./events/')
    parser.add_argument('--savepreddir', type=str, default=None,
                        help='path where predictions of the network are save')
    parser.add_argument('--loaddir', type=str, default=None)
    parser.add_argument('--model-name', type=str, default='neuralnet')
    parser.add_argument('--alignment-type', type=str, default='kabsch', choices=['default', 'linear', 'kabsch'])
    parser.add_argument('--virtual-node', action='store_true', help='use virtual node')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--use-val', action='store_true', help='use validation set')
    parser.add_argument('--seed', type=int, default=1334, help='random seed for experiments')
    parser.add_argument('--batch-size', type=int, default=20, help='batch size')
    parser.add_argument('--val-num-samples', type=int, default=10,
                        help='number of samples from prior used for validation')
    parser.add_argument('--tol', type=float, default=1e-5, help='tolerance for masking used in svd calculation')
    parser.add_argument('--use-X', action='store_true', default=False, help='use X as input for posterior of Z')
    parser.add_argument('--use-R', action='store_true', default=True, help='use R(X) as input for posterior of Z')

    if hyperparameter_search:
        parser.add_argument('--nb-trials', type=int, default=5, help='number of hyperparameter combinations')
        parser.opt_range('--refine_steps', type=int, default=0, low=0, high=3, tunable=True,
                            help='number of refinement steps if requested')
        parser.opt_list('--dim-h', type=int, default=50, options=[50, 100, 200], tunable=True,
                            help='dimension of the hidden')
        parser.opt_list('--dim-f', type=int, default=100, options=[50, 100, 200], tunable=True,
                            help='dimension of the hidden')
        parser.opt_range('--mpnn-steps', type=int, default=5, low=0, high=5, tunable=True,
                            help='number of mpnn steps')
        parser.opt_range('--w-reg', type=float, default=1e-3, low=1e-3, high=1e-1, tunable=True,
                            help='weight for conditional prior regularization')
        parser.add_argument('--condaenv-path', type=str)

        args = parser.parse_args()
        cluster = SlurmCluster(
            hyperparam_optimizer=args,
            log_path=args.ckptdir,
            python_cmd='python3'
        )
        # we'll request 10GB of memory per node
        cluster.memory_mb_per_node = 10000

        # set walltime
        cluster.job_time = '96:00:00'

        cluster.load_modules(['cuda/9.0.176', 'cudnn/9.0v7.0.5'])
        cluster.add_command('source activate {}'.format(args.condaenv_path))

    else:

        parser.add_argument('--refine_steps', type=int, default=0, help='number of refinement steps if requested')
        parser.add_argument('--dim-h', type=int, default=50, help='dimension of the hidden')
        parser.add_argument('--dim-f', type=int, default=100, help='dimension of the hidden')
        parser.add_argument('--mpnn-steps', type=int, default=5, help='number of mpnn steps')
        parser.add_argument('--w-reg', type=float, default=1e-3, help='weight for conditional prior regularization')

        args = parser.parse_args()

    if args.data == 'COD':
        n_max = 50
        dim_node = 33
        dim_edge = 15
        if args.virtual_node is True:
            n_max += 1
            dim_edge += 1
        nval = 3000
        ntst = 3000
    elif args.data == 'QM9':
        n_max = 9
        dim_node = 20
        dim_edge = 15
        if args.virtual_node is True:
            n_max += 1
            dim_edge += 1
        ntrn = 100000
        nval = 5000
        ntst = 5000

    dim_h = args.dim_h
    dim_f = args.dim_f
    batch_size = args.batch_size
    val_num_samples = args.val_num_samples

    if not os.path.exists(args.ckptdir):
        os.makedirs(args.ckptdir)
    if not os.path.exists(args.eventdir):
        os.makedirs(args.eventdir)

    save_path = os.path.join(args.ckptdir, args.model_name + '_model.ckpt')
    event_path = os.path.join(args.eventdir, args.model_name)

    if args.virtual_node:
        molvec_fname = data_path() + args.data+'_molvec_'+str(n_max-1)+'_vn.p'
        molset_fname = data_path() + args.data+'_molset_'+str(n_max-1)+'_vn.p'
    else:
        molvec_fname = data_path() + args.data+'_molvec_'+str(n_max)+'.p'
        molset_fname = data_path() + args.data+'_molset_'+str(n_max)+'.p'

    print('::: load data')
    [D1, D2, D3, D4, D5] = pkl.load(open(molvec_fname,'rb'))
    D1 = D1.todense()
    D2 = D2.todense()
    D3 = D3.todense()

    ntrn = len(D5)-nval-ntst

    [molsup, molsmi] = pkl.load(open(molset_fname,'rb'))

    D1_trn = D1[:ntrn]
    D2_trn = D2[:ntrn]
    D3_trn = D3[:ntrn]
    D4_trn = D4[:ntrn]
    D5_trn = D5[:ntrn]
    molsup_trn =molsup[:ntrn]
    D1_val = D1[ntrn:ntrn+nval]
    D2_val = D2[ntrn:ntrn+nval]
    D3_val = D3[ntrn:ntrn+nval]
    D4_val = D4[ntrn:ntrn+nval]
    D5_val = D5[ntrn:ntrn+nval]
    molsup_val =molsup[ntrn:ntrn+nval]
    D1_tst = D1[ntrn+nval:ntrn+nval+ntst]
    D2_tst = D2[ntrn+nval:ntrn+nval+ntst]
    D3_tst = D3[ntrn+nval:ntrn+nval+ntst]
    D4_tst = D4[ntrn+nval:ntrn+nval+ntst]
    D5_tst = D5[ntrn+nval:ntrn+nval+ntst]
    molsup_tst =molsup[ntrn+nval:ntrn+nval+ntst]

    if args.virtual_node:
        tm_trn = np.zeros(D2_trn.shape)
        tm_val = np.zeros(D2_val.shape)
        n_atoms_trn = D2_trn.sum(axis=1)
        n_atoms_val = D2_val.sum(axis=1)

        for i in range(D2_trn.shape[0]):
            tm_trn[i, :n_atoms_trn[i, 0]-1] = 1
        for i in range(D2_val.shape[0]):
            tm_val[i, :n_atoms_val[i, 0]-1] = 1

        if args.test and not args.use_val:
            n_atoms_tst = D2_trn.sum(axis=1)
            tm_tst = np.zeros(D2_tst.shape)
            for i in range(D2_tst.shape[0]):
                tm_tst[i, :n_atoms_val[i, 0]-1] = 1
    else:
        tm_trn, tm_val, tm_tst = None, None, None

    del D1, D2, D3, D4, D5, molsup
    print ('::: num train samples is ')
    print(D1_trn.shape, D3_trn.shape)

    if hyperparameter_search:
        cluster.optimize_parallel_cluster_gpu(train, nb_trials=args.nb_trials, job_name='dl4chem_random_search')
    else:
        train(args)
