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
        if os.uname().nodename == "mansimov-desktop":
            return "./data/"
        else:
            return "/misc/kcgscratch1/ChoGroup/mansimov/seokho_drive_datasets/"
    if getpass.getuser() == "em3382":
        return "/scratch/em3382/seokho_drive_datasets/"
    else:
        return "./"

def train(args, exp=None):

    if args.data == 'COD':
        n_max = 50
        dim_node = 35
        dim_edge = 10
        if args.virtual_node is True:
            n_max += 1
            dim_edge += 1
        nval = 3000
        ntst = 3000
    elif args.data == 'QM9':
        n_max = 9
        dim_node = 22
        dim_edge = 10
        if args.virtual_node is True:
            n_max += 1
            dim_edge += 1
        nval = 5000
        ntst = 5000
    elif args.data == 'CSD':
        n_max = 50
        dim_node = 98
        dim_edge = 10
        if args.virtual_node is True:
            n_max += 1
            dim_edge += 1
        nval = 3000
        ntst = 3000

    dim_h = args.dim_h
    dim_f = args.dim_f
    batch_size = args.batch_size
    val_num_samples = args.val_num_samples

    if not os.path.exists(args.ckptdir):
        os.makedirs(args.ckptdir)

    # create train and valid event dir for tensorboard logits
    args.train_eventdir = args.eventdir.split('/')
    args.train_eventdir.insert(-1, 'train')
    args.train_eventdir = '/'.join(args.train_eventdir)

    args.valid_eventdir = args.eventdir.split('/')
    args.valid_eventdir.insert(-1, 'valid')
    args.valid_eventdir = '/'.join(args.valid_eventdir)

    if not os.path.exists(args.train_eventdir):
        os.makedirs(args.train_eventdir)
    if not os.path.exists(args.valid_eventdir):
        os.makedirs(args.valid_eventdir)

    save_path = os.path.join(args.ckptdir, args.model_name + '_model.ckpt')
    #event_path = os.path.join(args.eventdir, args.model_name)

    if args.virtual_node:
        molvec_fname = data_path() + args.data+'_molvec_'+str(n_max-1)+'_vn.p'
        molset_fname = data_path() + args.data+'_molset_'+str(n_max-1)+'_vn.p'
    else:
        molvec_fname = data_path() + args.data+'_molvec_'+str(n_max)+'.p'
        molset_fname = data_path() + args.data+'_molset_'+str(n_max)+'.p'

    print('::: load data')
    if args.data == 'CSD':
        if args.test is False:
            molset_fname = 'CSD_mol/CSD_molset_50.p'
            D1, D2, D3, D4, D5 = [], [], [], [], []
            for i in range(11):
                with open('CSD_mol/CSD_molvec_50_{}.p'.format(i), 'rb') as f:
                    d1, d2, d3, d4, d5 = pkl.load(f)
                D1.append(d1)
                D2.append(d2)
                D3.append(d3)
                D4.append(d4)
                D5.append(d5)
            from sparse import coo
            D1 = coo.concatenate(D1, 0)
            D2 = coo.concatenate(D2, 0)
            D3 = coo.concatenate(D3, 0)
            D4 = np.concatenate(D4, 0)
            D5 = np.concatenate(D5, 0)
        else:
            if args.use_val is True:
                [D1, D2, D3, D4, D5] = pkl.load(open('CSD_mol/CSD_molvec_val.p', 'rb'))
            else:
                [D1, D2, D3, D4, D5] = pkl.load(open('CSD_mol/CSD_molvec_tst.p', 'rb'))
    else:
        [D1, D2, D3, D4, D5] = pkl.load(open(molvec_fname,'rb'))
    D1 = D1.todense()
    D2 = D2.todense()
    D3 = D3.todense()

    if args.data == 'CSD' and args.test is True:
        if args.use_val is True:
            D1_val, D2_val, D3_val, D4_val, D5_val = D1, D2, D3, D4, D5
            molsup_val = pkl.load(open('CSD_mol/CSD_molset_val.p', 'rb'))
        else:
            D1_tst, D2_tst, D3_tst, D4_tst, D5_tst = D1, D2, D3, D4, D5
            molsup_tst = pkl.load(open('CSD_mol/CSD_molset_tst.p', 'rb'))
        molsup = None
    else:
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
        print ('::: num train samples is ')
        print(D1_trn.shape, D3_trn.shape)

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

    model = MPNN.Model(args.data, n_max, dim_node, dim_edge, dim_h, dim_f, \
                        batch_size, val_num_samples, \
                        mpnn_steps=args.mpnn_steps, alignment_type=args.alignment_type, tol=args.tol,\
                        use_X=args.use_X, use_R=args.use_R, \
                        virtual_node=args.virtual_node, seed=args.seed, \
                        refine_steps=args.refine_steps, refine_mom=args.refine_mom, \
                        prior_T=args.prior_T)
    #if args.loaddir != None:
    #    model.saver.restore(model.sess, args.loaddir)

    if args.savepermol:
        args.savepreddir = os.path.join(args.savepreddir, args.data, "_val_" if args.use_val else "_test_")
        if not os.path.exists(args.savepreddir):
            os.makedirs(args.savepreddir)

    with model.sess:
        if args.test:
            if args.use_val:
                model.test(D1_val, D2_val, D3_val, D4_val, D5_val, molsup_val, \
                            load_path=args.loaddir, tm_v=tm_val, debug=args.debug, \
                            savepred_path=args.savepreddir, savepermol=args.savepermol, useFF=args.useFF)
            else:
                model.test(D1_tst, D2_tst, D3_tst, D4_tst, D5_tst, molsup_tst, \
                            load_path=args.loaddir, tm_v=tm_tst, debug=args.debug, \
                            savepred_path=args.savepreddir, savepermol=args.savepermol, useFF=args.useFF)
        else:
            model.train(D1_trn, D2_trn, D3_trn, D4_trn, D5_trn, molsup_trn, \
                        D1_val, D2_val, D3_val, D4_val, D5_val, molsup_val, \
                        load_path=args.loaddir, save_path=save_path, \
                        train_event_path=args.train_eventdir, valid_event_path=args.valid_eventdir, \
                        log_train_steps=args.log_train_steps, tm_trn=tm_trn, tm_val=tm_val, \
                        w_reg=args.w_reg, \
                        debug=args.debug, exp=exp)

def search_train(args, *extra_args):
    exp = Experiment(
        # Location to save the metrics.
        save_dir=args.ckptdir
    )
    exp.argparse(args)
    train(args, exp)
    exp.save()

def save_func(model):
    model.saver.save()

def load_func(model, loaddir):
    sess = tf.Session()
    saver = tf.train.import_meta_graph('my_test_model-1000.meta')
    saver.restore(model.sess, loaddir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train network')

    parser.add_argument('--data', type=str, default='QM9', choices=['COD', 'QM9', 'CSD'])
    parser.add_argument('--ckptdir', type=str, default='./checkpoints/')
    parser.add_argument('--eventdir', type=str, default='./events/')
    parser.add_argument('--savepreddir', type=str, default=None,
                        help='path where predictions of the network are save')
    parser.add_argument('--savepermol', action='store_true', help='save results per molecule')
    parser.add_argument('--loaddir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='neuralnet')
    parser.add_argument('--alignment_type', type=str, default='kabsch', choices=['default', 'linear', 'kabsch'])
    parser.add_argument('--virtual_node', action='store_true', help='use virtual node')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--use_val', action='store_true', help='use validation set')
    parser.add_argument('--seed', type=int, default=1334, help='random seed for experiments')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--val_num_samples', type=int, default=10,
                        help='number of samples from prior used for validation')
    parser.add_argument('--tol', type=float, default=1e-5, help='tolerance for masking used in svd calculation')
    parser.add_argument('--prior_T', type=float, default=1, help='temperature to use for the prior')
    parser.add_argument('--use_X', action='store_true', default=False, help='use X as input for posterior of Z')
    parser.add_argument('--use_R', action='store_true', default=True, help='use R(X) as input for posterior of Z')
    parser.add_argument('--w_reg', type=float, default=1e-5, help='weight for conditional prior regularization')
    parser.add_argument('--refine_mom', type=float, default=0.99, help='momentum used for refinement')
    parser.add_argument('--refine_steps', type=int, default=0, help='number of refinement steps if requested')
    parser.add_argument('--log_train_steps', type=int, default=100, help='number of steps to log train')
    parser.add_argument('--useFF', action='store_true', help='use force field minimisation if testing')

    parser.add_argument('--dim_h', type=int, default=50, help='dimension of the hidden')
    parser.add_argument('--dim_f', type=int, default=100, help='dimension of the hidden')
    parser.add_argument('--mpnn_steps', type=int, default=5, help='number of mpnn steps')

    args = parser.parse_args()

    train(args)
