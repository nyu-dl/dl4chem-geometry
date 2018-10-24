from __future__ import print_function

import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import sys, gc, os
import PredX_MPNN as MPNN
import sparse
import pdb

# hyper-parameters
#data = 'COD' #'COD' or 'QM9'

import argparse
parser = argparse.ArgumentParser(description='Train student network')

parser.add_argument('--data', type=str, default='COD', choices=['COD','QM9'])
parser.add_argument('--dec', type=str, default='npe', choices=['mpnn','npe','none'])
parser.add_argument('--ckptdir', type=str, default='./checkpoints/')
parser.add_argument('--eventdir', type=str, default='./events/')
parser.add_argument('--loaddir', type=str, default=None)
parser.add_argument('--model-name', type=str, default='test')
parser.add_argument('--alignment-type', type=str, default='kabsch', choices=['default','linear','kabsch'])
parser.add_argument('--debug', action='store_true', help='debug mode')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--use-val', action='store_true', help='use validation set')
parser.add_argument('--dim-h', type=int, default=50, help='dimension of the hidden')
parser.add_argument('--dim-f', type=int, default=100, help='dimension of the hidden')
parser.add_argument('--mpnn-steps', type=int, default=5, help='number of mpnn steps')
parser.add_argument('--mpnn-dec-steps', type=int, default=1, help='number of mpnn steps for decoding')
parser.add_argument('--npe-steps', type=int, default=10, help='number of mpnn steps')
parser.add_argument('--batch-size', type=int, default=20, help='batch size')
parser.add_argument('--tol', type=float, default=1e-5, help='tolerance for masking used in svd calculation')
parser.add_argument('--w-kldz', type=float, default=1., help='weight for kl divergence')
parser.add_argument('--w-pos', type=float, default=1., help='weight for positional loss')
parser.add_argument('--w-prox', type=float, default=0.00001, help='weight for proximity loss')
parser.add_argument('--w-R', type=float, default=1., help='weight for proximity loss iter 0')

args = parser.parse_args()

if args.data == 'COD':
    n_max = 50
    dim_node = 33
    dim_edge = 15
    nval = 3000
    ntst = 3000
elif args.data == 'QM9':
    n_max = 9
    dim_node = 20
    dim_edge = 15
    nval = 5000
    ntst = 5000

dim_h = args.dim_h
dim_f = args.dim_f
batch_size = args.batch_size

load_path = None
save_path = os.path.join(args.ckptdir, args.model_name + '_model.ckpt')
event_path = os.path.join(args.eventdir, args.model_name)
#save_path = args.save_dir+data+'_'+str(n_max)+'_'+str(args.dec)+'_model.ckpt'

print('::: load data')
[D1, D2, D3, D4, D5] = pkl.load(open('./'+args.data+'_molvec_'+str(n_max)+'.p','rb'))
D1 = D1.todense()
D2 = D2.todense()
D3 = D3.todense()

ntrn = len(D5)-nval-ntst

[molsup, molsmi] = pkl.load(open('./'+args.data+'_molset_'+str(n_max)+'.p','rb'))

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

del D1, D2, D3, D4, D5, molsup

model = MPNN.Model(args.data, n_max, dim_node, dim_edge, dim_h, dim_f, batch_size,\
                    args.dec, mpnn_steps=args.mpnn_steps, mpnn_dec_steps=args.mpnn_dec_steps, npe_steps=args.npe_steps, alignment_type=args.alignment_type, tol=args.tol)

if args.loaddir != None:
    model.saver.restore(model.sess, args.loaddir)

with model.sess:
    if args.test:
        if args.use_val:
            model.test(D1_val, D2_val, D3_val, D4_val, D5_val, molsup_val, debug=args.debug)
        else:
            model.test(D1_tst, D2_tst, D3_tst, D4_tst, D5_tst, molsup_tst, debug=args.debug)
    else:
        model.train(D1_trn, D2_trn, D3_trn, D4_trn, D5_trn, molsup_trn, \
                    D1_val, D2_val, D3_val, D4_val, D5_val, molsup_val, \
                    load_path, save_path, event_path, \
                    w_kldz=args.w_kldz, w_pos=args.w_pos, w_prox=args.w_prox, w_R=args.w_R, \
                    debug=args.debug)
    #model.saver.restore( model.sess, save_path )
