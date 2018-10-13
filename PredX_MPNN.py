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

class Model(object):

    def __init__(self, data, n_max, dim_node, dim_edge, dim_h, dim_f, batch_size, dec, alignment_type='default'):

        # hyper-parameters
        self.dec = dec
        self.data = data
        self.n_max, self.dim_node, self.dim_edge, self.dim_h, self.dim_f, self.batch_size = n_max, dim_node, dim_edge, dim_h, dim_f, batch_size

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

        self.n_atom = tf.reduce_sum( tf.transpose(self.mask, [0, 2, 1]), 2) #[batch_size, 1]
        self.n_atom_pair = self.n_atom * (self.n_atom - 1)

        self.node_embed = self._embed_node(self.node)
        self.edge_2 = tf.concat([self.edge, tf.tile( tf.reshape(self.n_atom, [self.batch_size, 1, 1, 1]), [1, self.n_max, self.n_max, 1] )], 3)

        # p(R|G)
        self.proximity_edge_wgt = self._edge_nn(self.edge_2, name = 'proximity', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.proximity_hidden = self._MPNN(self.proximity_edge_wgt, self.node_embed, name = 'proximity', reuse = False)
        self.proximity_pred = self._f_nn(tf.concat([self.proximity_hidden, self.node_embed], 2), self.edge_2, name = 'proximity', reuse = False) #[batch_size, n_max, n_max]
        #self.edge_2 = tf.concat([self.edge_2, tf.reshape(self.proximity_pred, [self.batch_size, self.n_max, self.n_max, 1])], 3)

        # q(Z|R,G)
        self.Z_edge_wgt = self._edge_nn(self.edge_2,  name = 'vaeZ', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.Z_hidden = self._MPNN(self.Z_edge_wgt, self.node_embed, name = 'vaeZ', reuse = False)
        self.Z_out = tf.layers.dense(self.Z_hidden, 2 * self.dim_h)
        self.Z_mu, self.Z_lsgms = tf.split(self.Z_out, [self.dim_h, self.dim_h], 2)
        self.Z_sample = self._draw_sample(self.Z_mu, self.Z_lsgms)

        # p(X|Z,G)
        self.X_edge_wgt = self._edge_nn(self.edge_2, name = 'vaeX', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.X_hidden = self._MPNN(self.X_edge_wgt, self.Z_sample + self.node_embed, name = 'vaeX', reuse = False)

        self.pos_list=[]
        self.prox_list=[]

        self.pos_init = self._g_nn(self.X_hidden, 3, name = 'vaeX', reuse = False)
        #self.pos_init -= tf.reduce_mean(self.pos_init, axis = 1, keepdims = True)

        self.pos_list.append(self.pos_init)
        self.prox_list.append(self._pos_to_proximity(self.pos_list[-1], reuse = False))

        if dec == 'mpnn':
            # MPNN-based PE
            self.PE_edge_wgt = self._edge_nn(tf.reshape(self.proximity_pred, [self.batch_size, self.n_max, self.n_max, 1]), name = 'PE', reuse = False)
            for i in range(1):
                self.PE_hidden = self._MPNN(self.PE_edge_wgt, self._embed_pos(self.pos_list[-1], reuse = (i!=0)), name = 'PE', reuse = (i!=0))
                self.pos_list.append(self.pos_list[-1] + self._g_nn(self.PE_hidden, 3, name = 'PE', reuse = (i!=0)))
                self.prox_list.append(self._pos_to_proximity(self.pos_list[-1], reuse = True))
        elif dec == 'npe':
            for i in range(1):   #hyperparameters!
                self.pos_list.append(self._NPE(self.pos_list[-1], self.prox_list[-1], self.proximity_pred, i!=0))
                self.prox_list.append(self._pos_to_proximity(self.pos_list[-1], reuse = True))

        self.saver = tf.train.Saver()
        self.sess = tf.Session()


    def train(self, D1_t, D2_t, D3_t, D4_t, D5_t, MS_t, D1_v, D2_v, D3_v, D4_v, D5_v, MS_v, load_path = None, save_path = None):

        # SummaryWriter
        summary_writer = SummaryWriter(save_path.split('/')[0] + '/' + save_path.split('/')[1] + '/' + self.dec, 'events')

        # objective functions
        cost_R = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.proximity_pred, self.proximity), [1, 2]) ) / 2

        cost_KLDZ = tf.reduce_mean( tf.reduce_sum( self._KLD(self.Z_mu, self.Z_lsgms), [1, 2]) )

        cost_pos_list = [tf.reduce_mean( self.msd_func(x, self.pos, self.mask) ) for x in self.pos_list]
        #cost_pos_list = [tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x, self.pos), [1, 2]) ) for x in self.pos_list]
        cost_prox_list = [tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x, self.proximity), [1, 2]) ) / 2 for x in self.prox_list]
        cost_reg_list = [tf.reduce_mean(tf.reduce_sum(tf.square(x), [1, 2]))  for x in self.pos_list]

        cost_pos = cost_pos_list[-1]#tf.add_n(cost_pos_list)/len(cost_pos_list)#
        cost_prox = tf.add_n(cost_prox_list)/len(cost_prox_list)
        cost_reg = cost_reg_list[0]

        #cost_pre = cost_R #+ 1. * cost_KLDZ + 1e-2 * cost_prox_list[0] + 1e-5 * cost_reg_list[0] #hyperparameters!
        #train_pre = tf.train.AdamOptimizer().minimize(cost_pre)
        cost_op = cost_KLDZ + cost_pos + 0.1 * cost_prox + 0. * cost_reg + 10. * cost_R #hyperparameters!
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

            trnscores = np.zeros((n_batch, 6))
            for i in range(n_batch):
                start_ = i * self.batch_size
                end_ = start_ + self.batch_size

                """
                D5_batch = self.sess.run(self.pos_list[-1],
                                    feed_dict = {self.node: D1_t[start_:end_], self.mask: D2_t[start_:end_], self.edge: D3_t[start_:end_], self.proximity: D4_t[start_:end_]})

                for j in range(start_,end_):
                    prb_mol = MS_t[j]
                    n_est = prb_mol.GetNumAtoms()

                    ref_pos = D5_batch[j-start_]
                    ref_cf = Chem.rdchem.Conformer(n_est)
                    for k in range(n_est):
                        ref_cf.SetAtomPosition(k, ref_pos[k].tolist())

                    ref_mol = copy.deepcopy(prb_mol)
                    ref_mol.RemoveConformer(0)
                    ref_mol.AddConformer(ref_cf)
                    RMS = AllChem.AlignMol(prb_mol, ref_mol)

                    D5_t[j][:n_est] =  np.array(prb_mol.GetConformer(0).GetPositions())
                """

                #trnresult = self.sess.run([train_op, cost_op, cost_KLDZ, cost_pos, cost_prox, cost_reg, cost_R],
                #                    feed_dict = {self.node: D1_t[start_:end_], self.mask: D2_t[start_:end_], self.edge: D3_t[start_:end_], self.proximity: D4_t[start_:end_], self.pos: D5_t[start_:end_]})
                #masks_np = np.sum(np.sum(D2_t[start_:end_], axis=1), axis=1)
                #print (masks_np)
                #print ((masks_np == 0).sum(), (masks_np == 1).sum())
                trnresult = self.sess.run([train_op, cost_op, cost_KLDZ, cost_pos, cost_prox, cost_reg, cost_R, self.pos_list[-1]],
                                    feed_dict = {self.node: D1_t[start_:end_], self.mask: D2_t[start_:end_], self.edge: D3_t[start_:end_], self.proximity: D4_t[start_:end_], self.pos: D5_t[start_:end_]})

                pred_pos = trnresult[-1]
                trnresult = trnresult[:-1]
                """
                ground_pos = D5_t[start_:end_]
                ground_mask = D2_t[start_:end_]
                np_rmsd = np.zeros((pred_pos.shape[0],))

                for ii in range(pred_pos.shape[0]):
                    mask_ii = int(ground_mask[ii].sum())
                    ground_pos_ii = ground_pos[ii][:mask_ii]
                    pred_pos_ii = pred_pos[ii][:mask_ii]
                    np_rmsd[ii] = rmsd.kabsch_rmsd(ground_pos_ii - rmsd.centroid(ground_pos_ii), pred_pos_ii - rmsd.centroid(pred_pos_ii))

                print (trnresult[3], np_rmsd.mean())
                print ('--------------')
                """
                # log results
                curr_iter = epoch * n_batch + i
                summary_writer.add_scalar("train/cost_op", trnresult[1], curr_iter)
                summary_writer.add_scalar("train/cost_KLDZ", trnresult[2], curr_iter)
                summary_writer.add_scalar("train/cost_pos", trnresult[3], curr_iter)
                summary_writer.add_scalar("train/cost_prox", trnresult[4], curr_iter)
                summary_writer.add_scalar("train/cost_reg", trnresult[5], curr_iter)
                summary_writer.add_scalar("train/cost_R", trnresult[6], curr_iter)


                assert np.sum(np.isnan(trnresult[1:])) == 0
                trnscores[i,:] = trnresult[1:]

            print(np.mean(trnscores,0))


            valscores = np.zeros(n_batch_val)
            for i in range(n_batch_val):
                start_ = i * self.batch_size
                end_ = start_ + self.batch_size

                D5_batch = self.sess.run(self.pos_list[-1],
                                    feed_dict = {self.node: D1_v[start_:end_], self.mask: D2_v[start_:end_], self.edge: D3_v[start_:end_], self.proximity: D4_v[start_:end_]})

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
            summary_writer.add_scalar("val/valscores", np.mean(valscores, 0), epoch)
            summary_writer.add_scalar("val/min_valscores", np.min(valaggr[0:epoch+1]), epoch)

            print('::: training epoch id', epoch, ':: --- val : ', np.mean(valscores, 0), '--- min : ', np.min(valaggr[0:epoch+1]))


            if epoch % 10 == 0:
                if save_path is not None:
                    self.saver.save( self.sess, save_path )

            if epoch > 30 and np.min(valaggr[0:epoch-30]) < np.min(valaggr[epoch-30:epoch+1]) and valaggr[epoch] < np.min(valaggr[0:epoch]) * 1.01:
                print('::: terminate')
                if save_path is not None:
                    self.saver.save( self.sess, save_path )

                break

    def do_mask(self, vec, m):
        return tf.boolean_mask(vec, tf.reshape(tf.greater(m, tf.constant(0.5)), [self.n_max,]) )

    """
    def kabsch_msd(self, frames, targets, masks):
        masks_int = tf.cast(tf.reduce_sum(masks, axis=[1,2]), tf.int32)
        loss = tf.stack([tf_kabsch_rmsd(targets[i][:masks_int[i]] - tf_centroid(targets[i][:masks_int[i]]), frames[i][:masks_int[i]] - tf_centroid(frames[i][:masks_int[i]])) for i in range(self.batch_size)], 0)
        return loss
    """
    def kabsch_msd(self, frames, targets, masks):
        losses = []
        for i in range(self.batch_size):
            frame = frames[i]
            target = targets[i]
            mask = masks[i]
            target_cent = target - tf_centroid_masked(target, mask)
            frame_cent = frame - tf_centroid_masked(frame, mask)
            losses.append(tf_kabsch_rmsd_masked(target_cent, frame_cent, mask))
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


    def _embed_pos(self, inp, reuse=True): #[batch_size, n_max, dim_node]

        with tf.variable_scope('embed_pos', reuse=reuse):
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

        for i in range(5): #hyperparameters!

            mv_0 = self._msg_nn(edge_wgt, node_hidden_0)
            node_hidden_0 = self._update_GRU(mv_0, node_hidden_0, name=name, reuse=(i+reuse)!=0)#[batch_size, n_max, dim_h]

        return node_hidden_0


    def _g_nn(self, inp, outdim, name='', reuse=True): #[batch_size, n_max, -]

        with tf.variable_scope('g_nn'+name, reuse=reuse):

            inp = tf.reshape(inp, [self.batch_size * self.n_max, int(inp.shape[2])])
            inp = tf.layers.dropout(inp, rate = 0.2)
            inp = tf.layers.dense(inp, self.dim_f, activation = tf.nn.sigmoid)
            inp = tf.layers.dropout(inp, rate = 0.2)
            #inp = tf.layers.dense(inp, self.dim_f, activation = tf.nn.sigmoid)
            inp = tf.layers.dense(inp, outdim)

            inp = tf.reshape(inp, [self.batch_size, self.n_max, outdim])
            inp = tf.multiply(inp, self.mask)

        return inp


    def _f_nn(self, inp, edge, name='', reuse=True): #[batch_size, n_max, dim_h], [batch_size, n_max, dim_edge]

        with tf.variable_scope('f_nn'+name, reuse=reuse):

            nhf_1 = tf.expand_dims(inp, axis = 2)
            nhf_2 = tf.expand_dims(inp, axis = 1)
            pairwise_add = tf.add(nhf_1, nhf_2) #[batch_size, n_max, n_max, dim_h]
            pairwise_mul = tf.multiply(nhf_1, nhf_2) #[batch_size, n_max, n_max, dim_h]
            inp = tf.concat([pairwise_add, pairwise_mul, edge], 3) #pairwise_mul, #[batch_size, n_max, n_max, 2 * dim_h + dim_edge]

            inp = tf.reshape(inp, [self.batch_size * self.n_max * self.n_max, int(inp.shape[3])])
            inp = tf.layers.dropout(inp, rate = 0.2)
            inp = tf.layers.dense(inp, self.dim_f, activation = tf.nn.sigmoid)
            inp = tf.layers.dropout(inp, rate = 0.2)
            #inp = tf.layers.dense(inp, self.dim_f, activation = tf.nn.sigmoid)
            inp = tf.layers.dense(inp, 1)
            inp = tf.exp(inp)

            inp = tf.reshape(inp, [self.batch_size, self.n_max, self.n_max])
            inp = tf.multiply(inp, self.mask)
            inp = tf.multiply(inp, tf.transpose(self.mask, perm = [0, 2, 1]))

            inp = tf.matrix_set_diag(inp, [[0] * self.n_max] * self.batch_size)

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


    def _NPE(self, pos, pos_proximity, ref_proximity, reuse=True): #[batch_size, n_max, 3], [batch_size, n_max, n_max]

        with tf.variable_scope('NPE', reuse=reuse):

            diff_proximity = tf.subtract(ref_proximity, pos_proximity)

            delta = tf.reshape(tf.reduce_mean( tf.square( diff_proximity ) , [1, 2]), [self.batch_size, 1]) / self.n_atom_pair

            w1 = tf.concat([delta, self.n_atom], 1) #[batch_size, 1]
            w1 = tf.layers.dense(w1, 10, activation = tf.nn.sigmoid)
            w1 = tf.layers.dense(w1, 1)
            w1 = tf.exp(w1)
            w1 = tf.reshape(w1, [self.batch_size, 1, 1])

            wgts = tf.div(diff_proximity, pos_proximity + 1e-5)
            wgts = tf.multiply(wgts, self.mask) # [batch_size, n_max, n_max]
            wgts = tf.multiply(wgts, tf.transpose(self.mask, perm = [0, 2, 1]))
            wgts = tf.reshape(wgts, [self.batch_size, self.n_max, self.n_max, 1])

            pos_1 = tf.expand_dims(pos, axis = 2)
            pos_2 = tf.expand_dims(pos, axis = 1)
            pos_sub = tf.subtract(pos_1, pos_2) # [batch_size, n_max, n_max, 3]

            pos_diff = tf.multiply(wgts, pos_sub) # [batch_size, n_max, n_max, 3]
            pos_diff = tf.transpose(pos_diff, [0, 1, 3, 2])
            pos_diff = w1 * tf.reduce_mean(pos_diff, 3)# [batch_size, n_max, 3]

            pos = pos + pos_diff
            pos = tf.multiply(pos, self.mask)

        return pos


    def _KLD(self, mu, lsgm):# [batch_size, n_max, dim_h]

        a = tf.exp(lsgm) + tf.square(mu)
        b = 1. + lsgm

        kld = 0.5 * tf.reduce_sum(a - b, 2, keepdims = True) * self.mask

        return kld


    def next_pos(self, pos_input, proximity_pred_input, mask_ref):

        return self.sess.run(self.pos_o, feed_dict = {self.pos_i: pos_input, self.proximity_pred_i: proximity_pred_input, self.mask: mask_ref})
