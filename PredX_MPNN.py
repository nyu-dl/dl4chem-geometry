from __future__ import print_function

import numpy as np
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem
import copy

class Model(object):

    def __init__(self, data, n_max, dim_node, dim_edge, dim_h, dim_f, batch_size):
         
        # hyper-parameters
        self.data = data
        self.n_max, self.dim_node, self.dim_edge, self.dim_h, self.dim_f, self.batch_size = n_max, dim_node, dim_edge, dim_h, dim_f, batch_size

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

        # q(Z|R,G)
        self.Z_edge_wgt = self._edge_nn(self.edge_2,  name = 'vaeZ', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.Z_hidden = self._MPNN(self.Z_edge_wgt, self.node_embed, name = 'vaeZ', reuse = False)
        self.Z_out = tf.layers.dense(self.Z_hidden, 2 * self.dim_h)
        self.Z_mu, self.Z_lsgms = tf.split(self.Z_out, [self.dim_h, self.dim_h], 2)
        self.Z_sample = self._draw_sample(self.Z_mu, self.Z_lsgms)

        # p(X|Z,G)
        self.X_edge_wgt = self._edge_nn(self.edge_2, name = 'vaeX', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.X_hidden = self._MPNN(self.X_edge_wgt, self.Z_sample + self.node_embed, name = 'vaeX', reuse = False)
        self.pos_0 = self._g_nn(self.X_hidden, 3, name = 'vaeX', reuse = False)
        self.pos_0_prox = self._pos_to_proximity(self.pos_0, False)
        
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        
    
    def train(self, D1_t, D2_t, D3_t, D4_t, D5_t, MS_t, D1_v, D2_v, D3_v, D4_v, D5_v, MS_v, load_path = None, save_path = None):

        # objective functions
        cost_KLDZ = tf.reduce_mean( tf.reduce_sum( self._KLD_zero(self.Z_mu, self.Z_lsgms), [1, 2]) ) # posterior - prior
        cost_proximity = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.pos_0_prox, self.proximity), [1, 2]) ) / 2  
        cost_reg = tf.reduce_mean(tf.reduce_sum(tf.square(self.pos_0), [1, 2]))              
        
        cost_npe = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.pos_0, self.pos), [1, 2]) )
                
        cost = cost_KLDZ + cost_npe + 0 * cost_proximity + 0 * cost_reg

        train_op = tf.train.AdamOptimizer().minimize(cost)


        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()
        if load_path is not None:
            self.saver.restore( self.sess, load_path )

        # session 
        print('::: start training')
        n_batch = int(len(D1_t)/self.batch_size)
        n_batch_val = int(len(D1_v)/self.batch_size)
        np.set_printoptions(precision=5, suppress=True)
        
        valaggr = np.zeros(500)
        for epoch in range(500):
        
            [D1_t, D2_t, D3_t, D4_t, D5_t, MS_t] = self._permutation([D1_t, D2_t, D3_t, D4_t, D5_t, MS_t])
            
            trnscores = np.zeros((n_batch, 3))
            for i in range(n_batch):
                start_ = i * self.batch_size
                end_ = start_ + self.batch_size
        
                D5_batch = self.sess.run(self.pos_0,
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

                trnresult = self.sess.run([train_op, cost, cost_KLDZ, cost_npe],
                                    feed_dict = {self.node: D1_t[start_:end_], self.mask: D2_t[start_:end_], self.edge: D3_t[start_:end_], self.proximity: D4_t[start_:end_], self.pos: D5_t[start_:end_]})
 
                assert np.sum(np.isnan(trnresult[1:])) == 0
                trnscores[i,:] = trnresult[1:]
            
            print(np.mean(trnscores,0))
            
            
            valscores = np.zeros(n_batch_val)
            for i in range(n_batch_val): 
                start_ = i * self.batch_size
                end_ = start_ + self.batch_size
                     
                D5_batch = self.sess.run(self.pos_0,
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
            print('::: training epoch id', epoch, ':: --- val : ', np.mean(valscores, 0), '--- min : ', np.min(valaggr[0:epoch+1]))


            if epoch > 20 and np.min(valaggr[0:epoch-20]) < np.min(valaggr[epoch-20:epoch+1]) and valaggr[epoch] < np.min(valaggr[0:epoch]) * 1.01:
                print('::: terminate')
                if save_path is not None:
                    self.saver.save( self.sess, save_path )
               
                break
       
            
    def _permutation(self, set):
    
        permid = np.random.permutation(len(set[0]))
        for i in range(len(set)):
            set[i] = set[i][permid]
    
        return set
    

    def _draw_sample(self, mu, lsgms):
    
        epsilon = tf.random_normal(tf.shape(lsgms), 0, 1)
        sample = tf.multiply(tf.exp(0.5*lsgms), epsilon)
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
    
        for i in range(5):
        
            mv_0 = self._msg_nn(edge_wgt, node_hidden_0)
            node_hidden_0 = self._update_GRU(mv_0, node_hidden_0, name=name, reuse=(i+reuse)!=0)#[batch_size, n_max, dim_h]
     
        return node_hidden_0
        
        
    def _g_nn(self, inp, outdim, name='', reuse=True): #[batch_size, n_max, -]
    
        with tf.variable_scope('g_nn'+name, reuse=reuse):
  
            inp = tf.reshape(inp, [self.batch_size * self.n_max, int(inp.shape[2])])
            inp = tf.layers.dropout(inp, rate = 0.2)
            inp = tf.layers.dense(inp, self.dim_f, activation = tf.nn.sigmoid)
            inp = tf.layers.dropout(inp, rate = 0.2)
            inp = tf.layers.dense(inp, self.dim_f, activation = tf.nn.sigmoid)
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
            inp = tf.layers.dense(inp, self.dim_f, activation = tf.nn.sigmoid)
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
       

    def _KLD_zero(self, mu0, lsgm0):# [batch_size, n_max, dim_h]
        
        a = tf.exp(lsgm0) + tf.square(mu0)
        b = 1 + lsgm0
        
        kld = 0.5 * tf.reduce_sum(a - b, 2, keepdims = True) * self.mask
        
        return kld 
        
        
    def next_pos(self, pos_input, proximity_pred_input, mask_ref):
    
        return self.sess.run(self.pos_o, feed_dict = {self.pos_i: pos_input, self.proximity_pred_i: proximity_pred_input, self.mask: mask_ref}) 