from __future__ import print_function

import numpy as np
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem
import tftraj.rmsd as rmsd
import copy
from tensorboardX import SummaryWriter
from tf_rmsd import tf_centroid, tf_centroid_masked, tf_kabsch_rmsd_masked, tf_kabsch_rmsd
import pdb
import rmsd
import glob
import os
import shutil

class Model(object):

    def __init__(self, data, n_max, dim_node, dim_edge, dim_h, dim_f, batch_size, mpnn_steps=5, alignment_type='default', tol=1e-5):

        # hyper-parameters
        self.data = data
        self.mpnn_steps = mpnn_steps
        self.n_max, self.dim_node, self.dim_edge, self.dim_h, self.dim_f, self.batch_size = n_max, dim_node, dim_edge, dim_h, dim_f, batch_size
        self.tol = tol
        if alignment_type == 'linear':
            self.msd_func = self.linear_transform_msd
        elif alignment_type == 'kabsch':
            self.msd_func = self.kabsch_msd
        elif alignment_type == 'default':
            self.msd_func = self.mol_msd

        # variables
        self.G = tf.Graph()
        self.G.as_default()

        self.node = tf.placeholder(tf.float32, [self.batch_size, self.n_max, self.dim_node])
        self.mask = tf.placeholder(tf.float32, [self.batch_size, self.n_max, 1]) # node yes = 1, no = 0
        self.edge = tf.placeholder(tf.float32, [self.batch_size, self.n_max, self.n_max, self.dim_edge])
        self.pos = tf.placeholder(tf.float32, [self.batch_size, self.n_max, 3])
        self.proximity = tf.placeholder(tf.float32, [self.batch_size, self.n_max, self.n_max])
        self.trn_flag = tf.placeholder(tf.bool)

        self.n_atom = tf.reduce_sum( tf.transpose(self.mask, [0, 2, 1]), 2) #[batch_size, 1]

        self.node_embed = self._embed_node(self.node)
        self.edge_2 = tf.concat([self.edge, tf.tile( tf.reshape(self.n_atom, [self.batch_size, 1, 1, 1]), [1, self.n_max, self.n_max, 1] )], 3)

        # p(Z|G) -- prior of Z
        self.priorZ_edge_wgt = self._edge_nn(self.edge_2, name = 'priorZ', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.priorZ_hidden = self._MPNN(self.priorZ_edge_wgt, self.node_embed, name = 'priorZ', reuse = False)
        self.priorZ_out = self._g_nn(self.priorZ_hidden, self.node_embed, 2 * self.dim_h, name = 'priorZ', reuse = False)
        self.priorZ_mu, self.priorZ_lsgms = tf.split(self.priorZ_out, [self.dim_h, self.dim_h], 2)
        self.priorZ_sample = self._draw_sample(self.priorZ_mu, self.priorZ_lsgms)

        # q(Z|R(X),G) -- posterior of Z, used R insted of X as input for simplicity, should be updated
        self.postZ_edge_wgt = self._edge_nn(self.edge_2,  name = 'postZ', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.postZ_hidden = self._MPNN(self.postZ_edge_wgt, tf.concat([self.node_embed, self.pos], 2), name = 'postZ', reuse = False)
        #self.postZ_edge_wgt = self._edge_nn(tf.concat([self.edge_2, tf.reshape(self.proximity, [self.batch_size, self.n_max, self.n_max, 1])], 3),  name = 'postZ', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        #self.postZ_hidden = self._MPNN(self.postZ_edge_wgt, self.node_embed, name = 'postZ', reuse = False)
        self.postZ_out = self._g_nn(self.postZ_hidden, self.node_embed, 2 * self.dim_h, name = 'postZ', reuse = False)
        self.postZ_mu, self.postZ_lsgms = tf.split(self.postZ_out, [self.dim_h, self.dim_h], 2)
        self.postZ_sample = self._draw_sample(self.postZ_mu, self.postZ_lsgms)

        # p(X|Z,G) -- posterior of X
        self.X_edge_wgt = self._edge_nn(self.edge_2, name = 'postX', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.X_hidden = self._MPNN(self.X_edge_wgt, self.postZ_sample + self.node_embed, name = 'postX', reuse = False)
        self.X_pred = self._g_nn(self.X_hidden, self.node_embed, 3, name = 'postX', reuse = False)

        # Prediction of X with p(Z|G) in the test phase
        self.PX_edge_wgt = self._edge_nn(self.edge_2, name = 'postX', reuse = True) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.PX_hidden = self._MPNN(self.PX_edge_wgt, self.priorZ_sample + self.node_embed, name = 'postX', reuse = True)
        self.PX_pred = self._g_nn(self.PX_hidden, self.node_embed, 3, name = 'postX', reuse = True)

        self.saver = tf.train.Saver()
        self.sess = tf.Session()


    def test(self, D1_v, D2_v, D3_v, D4_v, D5_v, MS_v, debug=False):
        # session
        n_batch_val = int(len(D1_v)/self.batch_size)
        np.set_printoptions(precision=5, suppress=True)

        valscores = np.zeros(n_batch_val)
        print ("testing model...")
        for i in range(n_batch_val):
            print (i, n_batch_val)
            start_ = i * self.batch_size
            end_ = start_ + self.batch_size

            D5_batch = self.sess.run(self.PX_pred,
                                feed_dict = {self.node: D1_v[start_:end_], self.mask: D2_v[start_:end_], self.edge: D3_v[start_:end_], self.proximity: D4_v[start_:end_], self.trn_flag: False})

            valres=[]
            for j in range(start_,end_):
                prb_mol = MS_v[j]
                n_est = prb_mol.GetNumAtoms()

                ref_pos = D5_batch[j-start_]
                ref_cf = Chem.rdchem.Conformer(n_est)
                for k in range(n_est):
                    ref_cf.SetAtomPosition(k, ref_pos[k].tolist())

                ref_mol = copy.deepcopy(prb_mol)
                ref_mol.RemoveConformer(0)
                ref_mol.AddConformer(ref_cf)
                valres.append(AllChem.AlignMol(prb_mol, ref_mol))

            valscores[i] = np.mean(valres)
        print ("val scores are {}".format(np.mean(valscores)))


    def train(self, D1_t, D2_t, D3_t, D4_t, D5_t, MS_t, D1_v, D2_v, D3_v, D4_v, D5_v, MS_v,\
            load_path = None, save_path = None, event_path = None, \
            w_reg=1e-3, debug=False):

        # SummaryWriter
        if not debug:
            summary_writer = SummaryWriter(event_path)

        # objective functions
        cost_KLDZ = tf.reduce_mean( tf.reduce_sum( self._KLD(self.postZ_mu, self.postZ_lsgms, self.priorZ_mu, self.priorZ_lsgms), [1, 2]) ) # posterior | prior
        cost_KLD0 = tf.reduce_mean( tf.reduce_sum( self._KLD_zero(self.priorZ_mu, self.priorZ_lsgms), [1, 2]) ) # prior | N(0,1)
        
        cost_X = tf.reduce_mean( self.msd_func(self.X_pred, self.pos, self.mask) )

        cost_op = cost_X + cost_KLDZ + w_reg * cost_KLD0 #hyperparameters!
        train_op = tf.train.AdamOptimizer().minimize(cost_op)

        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()
        if load_path is not None:
            self.saver.restore( self.sess, load_path )

        # session
        n_batch = int(len(D1_t)/self.batch_size)
        n_batch_val = int(len(D1_v)/self.batch_size)
        np.set_printoptions(precision=5, suppress=True)


        # training
        print('::: start training')
        valaggr = np.zeros(500)
        for epoch in range(500):

            [D1_t, D2_t, D3_t, D4_t, D5_t] = self._permutation([D1_t, D2_t, D3_t, D4_t, D5_t])

            trnscores = np.zeros((n_batch, 4))
            for i in range(n_batch):
                start_ = i * self.batch_size
                end_ = start_ + self.batch_size

                trnresult = self.sess.run([train_op, cost_op, cost_X, cost_KLDZ, cost_KLD0],
                                    feed_dict = {self.node: D1_t[start_:end_], self.mask: D2_t[start_:end_], self.edge: D3_t[start_:end_], self.proximity: D4_t[start_:end_], self.pos: D5_t[start_:end_], self.trn_flag: True})

                trnresult = trnresult[1:]
                if debug:
                    print (trnresult)

                # log results
                curr_iter = epoch * n_batch + i
                if not debug:
                    summary_writer.add_scalar("train/cost_op", trnresult[0], curr_iter)
                    summary_writer.add_scalar("train/cost_X", trnresult[1], curr_iter)
                    summary_writer.add_scalar("train/cost_KLDZ", trnresult[2], curr_iter)
                    summary_writer.add_scalar("train/cost_KLD0", trnresult[3], curr_iter)

                assert np.sum(np.isnan(trnresult)) == 0
                trnscores[i,:] = trnresult

            print(np.mean(trnscores,0))


            valscores = np.zeros(n_batch_val)
            for i in range(n_batch_val):
                start_ = i * self.batch_size
                end_ = start_ + self.batch_size

                D5_batch = self.sess.run(self.PX_pred,
                                    feed_dict = {self.node: D1_v[start_:end_], self.mask: D2_v[start_:end_], self.edge: D3_v[start_:end_], self.proximity: D4_v[start_:end_], self.trn_flag: False})

                valres=[]
                for j in range(start_,end_):
                    prb_mol = MS_v[j]
                    n_est = prb_mol.GetNumAtoms()

                    ref_pos = D5_batch[j-start_]
                    ref_cf = Chem.rdchem.Conformer(n_est)
                    for k in range(n_est):
                        ref_cf.SetAtomPosition(k, ref_pos[k].tolist())

                    ref_mol = copy.deepcopy(prb_mol)
                    ref_mol.RemoveConformer(0)
                    ref_mol.AddConformer(ref_cf)
                    valres.append(AllChem.AlignMol(prb_mol, ref_mol))

                valscores[i] = np.mean(valres)

            valaggr[epoch] = np.mean(valscores)
            if not debug:
                summary_writer.add_scalar("val/valscores", np.mean(valscores, 0), epoch)
                summary_writer.add_scalar("val/min_valscores", np.min(valaggr[0:epoch+1]), epoch)

            print('::: training epoch id', epoch, ':: --- val : ', np.mean(valscores, 0), '--- min : ', np.min(valaggr[0:epoch+1]))


            if save_path is not None and not debug:
                self.saver.save( self.sess, save_path )
            # keep track of the best model as well in the separate checkpoint
            # it is done by copying the checkpoint
            if valaggr[epoch] == np.min(valaggr[0:epoch+1]) and not debug:
                for ckpt_f in glob.glob(save_path + '*'):
                    model_name_split = ckpt_f.split('/')
                    model_path = '/'.join(model_name_split[:-1])
                    model_name = model_name_split[-1]
                    best_model_name = model_name.split('.')[0] + '_best.' + '.'.join(model_name.split('.')[1:])
                    full_best_model_path = os.path.join(model_path, best_model_name)
                    full_model_path = ckpt_f
                    shutil.copyfile(full_model_path, full_best_model_path)

    def do_mask(self, vec, m):
        return tf.boolean_mask(vec, tf.reshape(tf.greater(m, tf.constant(0.5)), [self.n_max,]) )

    def kabsch_msd(self, frames, targets, masks):
        losses = []
        for i in range(self.batch_size):
            frame = frames[i]
            target = targets[i]
            mask = masks[i]
            target_cent = target - tf_centroid_masked(target, mask, self.tol)
            frame_cent = frame - tf_centroid_masked(frame, mask, self.tol)
            losses.append(tf_kabsch_rmsd_masked(tf.stop_gradient(target_cent), frame_cent, mask, self.tol))
        loss = tf.stack(losses, 0)
        return loss

    def mol_msd(self, frames, targets, masks):
        frames -= tf.reduce_mean(frames, axis = 1, keepdims = True)
        targets -= tf.reduce_mean(targets, axis = 1, keepdims = True)

        loss = tf.stack([rmsd.squared_deviation( self.do_mask(frames[i], masks[i]), self.do_mask(targets[i], masks[i]) ) for i in range(self.batch_size)], 0)
        return loss / tf.reduce_sum(masks, axis=[1,2])

    def linear_transform_msd(self, frames, targets, masks):
        def linearly_transform_frames(padded_frames, padded_targets):
            s, u, v = tf.svd(padded_frames)
            tol = 1e-7
            atol = tf.reduce_max(s) * tol
            s = tf.boolean_mask(s, s > atol)
            s_inv = tf.diag(1. / s)
            pseudo_inverse = tf.matmul(v, tf.matmul(s_inv, u, transpose_b=True))

            weight_matrix = tf.matmul(padded_targets, pseudo_inverse)
            transformed_frames = tf.matmul(weight_matrix, padded_frames)
            return transformed_frames

        padding = tf.constant([[0, 0], [0, 0], [0, 1]])
        padded_frames = tf.pad(frames, padding, 'constant', constant_values=1)
        padded_targets = tf.pad(targets, padding, 'constant', constant_values=1)

        mask_matrices = []
        for i in range(self.batch_size):
            mask_matrix = tf.diag(tf.reshape(masks[i], [-1]))
            mask_matrices.append(mask_matrix)
        #mask_matrix = tf.diag(tf.reshape(masks, [self.batch_size, -1]))
        mask_tensor = tf.stack(mask_matrices)
        masked_frames = tf.matmul(mask_tensor, padded_frames)
        masked_targets = tf.matmul(mask_tensor, padded_targets)
        transformed_frames = []
        for i in range(self.batch_size):
            transformed_frames.append(linearly_transform_frames(masked_frames[i], masked_targets[i]))
        transformed_frames = tf.stack(transformed_frames)
        #transformed_frames = linearly_transform_frames(masked_frames, masked_targets)
        loss = tf.losses.mean_squared_error(transformed_frames, masked_targets)

        return loss

    def _permutation(self, set):

        permid = np.random.permutation(len(set[0]))
        for i in range(len(set)):
            set[i] = set[i][permid]

        return set


    def _draw_sample(self, mu, lsgms):

        epsilon = tf.random_normal(tf.shape(lsgms), 0., 1.)
        sample = tf.multiply(tf.exp(0.5 * lsgms), epsilon)
        sample = tf.add(mu, sample)
        sample = tf.multiply(sample, self.mask)

        return sample


    def _embed_node(self, inp): #[batch_size, n_max, dim_node]

        inp = tf.reshape(inp, [self.batch_size * self.n_max, int(inp.shape[2])])

        inp = tf.layers.dense(inp, self.dim_h, activation = tf.nn.sigmoid)
        inp = tf.layers.dense(inp, self.dim_h, activation = tf.nn.tanh)

        inp = tf.reshape(inp, [self.batch_size, self.n_max, self.dim_h])
        inp = tf.multiply(inp, self.mask)

        return inp


    def _edge_nn(self, inp, name='', reuse=True): #[batch_size, n_max, n_max, dim_edge]

        with tf.variable_scope('edge_nn'+name, reuse=reuse):

            inp = tf.reshape(inp, [self.batch_size * self.n_max * self.n_max, int(inp.shape[3])])

            inp = tf.layers.dense(inp, 2 * self.dim_h, activation = tf.nn.sigmoid)
            inp = tf.layers.dense(inp, self.dim_h * self.dim_h, activation = tf.nn.tanh)

            inp = tf.reshape(inp, [self.batch_size, self.n_max, self.n_max, self.dim_h, self.dim_h])

        return inp


    def _msg_nn(self, wgt, node, name='', reuse=True):

        wgt = tf.reshape(wgt, [self.batch_size * self.n_max, self.n_max * self.dim_h, self.dim_h])
        node = tf.reshape(node, [self.batch_size * self.n_max, self.dim_h, 1])

        msg = tf.matmul(wgt, node)
        msg = tf.reshape(msg, [self.batch_size, self.n_max, self.n_max, self.dim_h])
        msg = tf.transpose(msg, perm = [0, 2, 3, 1])
        msg = tf.reduce_mean(msg, 3) / self.n_max

        return msg


    def _update_GRU(self, msg, node, name='', reuse=True):

        with tf.variable_scope('update_GRU'+name, reuse=reuse):

            msg = tf.reshape(msg, [self.batch_size * self.n_max, 1, self.dim_h])
            node = tf.reshape(node, [self.batch_size * self.n_max, self.dim_h])

            cell = tf.nn.rnn_cell.GRUCell(self.dim_h)
            _, node_next = tf.nn.dynamic_rnn(cell, msg, initial_state = node)

            node_next = tf.reshape(node_next, [self.batch_size, self.n_max, self.dim_h])
            node_next = tf.multiply(node_next, self.mask)

        return node_next


    def _MPNN(self, edge_wgt, node_hidden_0, name='', reuse=True):

        for i in range(self.mpnn_steps): #hyperparameters!

            mv_0 = self._msg_nn(edge_wgt, node_hidden_0)
            node_hidden_0 = self._update_GRU(mv_0, node_hidden_0, name=name, reuse=(i+reuse)!=0)#[batch_size, n_max, dim_h]

        return node_hidden_0


    def _g_nn(self, inp, node, outdim, name='', reuse=True): #[batch_size, n_max, -]

        with tf.variable_scope('g_nn'+name, reuse=reuse):

            inp = tf.concat([inp, node], 2)

            inp = tf.reshape(inp, [self.batch_size * self.n_max, int(inp.shape[2])])
            inp = tf.layers.dropout(inp, rate = 0.2, training = self.trn_flag)
            inp = tf.layers.dense(inp, self.dim_f, activation = tf.nn.sigmoid)
            inp = tf.layers.dropout(inp, rate = 0.2, training = self.trn_flag)
            #inp = tf.layers.dense(inp, self.dim_f, activation = tf.nn.sigmoid)
            inp = tf.layers.dense(inp, outdim)

            inp = tf.reshape(inp, [self.batch_size, self.n_max, outdim])
            inp = tf.multiply(inp, self.mask)

        return inp


    def _pos_to_proximity(self, pos, reuse=True): #[batch_size, n_max, 3]

        with tf.variable_scope('pos_to_proximity', reuse=reuse):

            pos_1 = tf.expand_dims(pos, axis = 2)
            pos_2 = tf.expand_dims(pos, axis = 1)

            pos_sub = tf.subtract(pos_1, pos_2)
            proximity = tf.square(pos_sub)
            proximity = tf.reduce_sum(proximity, 3)
            proximity = tf.sqrt(proximity + 1e-5)

            proximity = tf.reshape(proximity, [self.batch_size, self.n_max, self.n_max])
            proximity = tf.multiply(proximity, self.mask)
            proximity = tf.multiply(proximity, tf.transpose(self.mask, perm = [0, 2, 1]))

            proximity = tf.matrix_set_diag(proximity, [[0] * self.n_max] * self.batch_size)

        return proximity


    def _KLD(self, mu0, lsgm0, mu1, lsgm1):# [batch_size, n_max, dim_h]
        
        var0 = tf.exp(lsgm0)
        var1 = tf.exp(lsgm1)
        a = tf.div( var0 + 1e-5, var1 + 1e-5)    
        b = tf.div( tf.square( tf.subtract(mu1, mu0) ), var1 + 1e-5)
        c = tf.log( tf.div(var1 + 1e-5, var0 + 1e-5 ) + 1e-5)
        
        kld = 0.5 * tf.reduce_sum(a + b - 1 + c, 2, keepdims = True) * self.mask
        
        return kld 
    

    def _KLD_zero(self, mu0, lsgm0):# [batch_size, n_max, dim_h]
        
        a = tf.exp(lsgm0) + tf.square(mu0)
        b = 1 + lsgm0
        
        kld = 0.5 * tf.reduce_sum(a - b, 2, keepdims = True) * self.mask
        
        return kld 