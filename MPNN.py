import numpy as np
import tensorflow as tf

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
        
        self.proximity = tf.placeholder(tf.float32, [self.batch_size, self.n_max, self.n_max])
        
        self.n_atom = tf.reduce_sum( tf.transpose(self.mask, [0, 2, 1]), 2) #[batch_size, 1]
        self.n_atom_pair = self.n_atom * (self.n_atom - 1)
        
        self.node_embed = self._embed_node(self.node)
        self.edge_2 = tf.concat([self.edge, tf.tile( tf.reshape(self.n_atom, [self.batch_size, 1, 1, 1]), [1, self.n_max, self.n_max, 1] )], 3)

        # p(Z|G)
        self.priorZ_edge_wgt = self._edge_nn(self.edge_2, name = 'priorZ', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.priorZ_hidden = self._MPNN(self.priorZ_edge_wgt, self.node_embed, name = 'priorZ', reuse = False)
        self.priorZ_out = self._g_nn(self.priorZ_hidden, self.node, 2 * self.dim_h, name = 'priorZ', reuse = False)
        self.priorZ_mu, self.priorZ_lsgms = tf.split(self.priorZ_out, [self.dim_h, self.dim_h], 2)
        self.priorZ_sample = self._draw_sample(self.priorZ_mu, self.priorZ_lsgms)
        self.priorZ_embed = self._embed_latent(self.priorZ_sample, name = 'Z', reuse=False)
        
        # q(Z|R,G)
        self.postZ_edge_wgt = self._edge_nn(tf.concat([self.edge_2, tf.reshape(self.proximity, [self.batch_size, self.n_max, self.n_max, 1])], 3),  name = 'postZ', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.postZ_hidden = self._MPNN(self.postZ_edge_wgt, self.node_embed, name = 'postZ', reuse = False)
        self.postZ_out = self._g_nn(self.postZ_hidden, self.node, 2 * self.dim_h, name = 'postZ', reuse = False)
        self.postZ_mu, self.postZ_lsgms = tf.split(self.postZ_out, [self.dim_h, self.dim_h], 2)
        self.postZ_sample = self._draw_sample(self.postZ_mu, self.postZ_lsgms)
        self.postZ_embed = self._embed_latent(self.postZ_sample, name = 'Z', reuse=True)

        # p(R|G)
        self.proximity_edge_wgt = self._edge_nn(self.edge_2, name = 'proximity', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        self.proximity_hidden = self._MPNN(self.proximity_edge_wgt, self.node_embed, name = 'proximity', reuse = False)
        self.proximity_pred = self._f_nn(tf.concat([self.proximity_hidden, self.node_embed], 2), self.edge_2, name = 'proximity', reuse = False) #[batch_size, n_max, n_max] 
        self.val_proximity_pred = self.proximity_pred
        
        # p(X|Z,G) with NPE
        self.postX_edge_wgt = self._edge_nn(self.edge_2, name = 'postX', reuse = False) #[batch_size, n_max, n_max, dim_h, dim_h]
        
        self.postX_hidden = self._MPNN(self.postX_edge_wgt, self.postZ_embed, name = 'postX', reuse = False)
        self.pos_0 = self._g_nn(self.postX_hidden, self.node, 3, name = 'postX', reuse = False)
        self.pos_proximity_list=[]
        self.pos_proximity_list.append(self._pos_to_proximity(self.pos_0, reuse = False))
        for i in range(100):
            self.pos_0 = self._NPE(self.pos_0, self.pos_proximity_list[-1], self.proximity_pred, i!=0)#  
            self.pos_proximity_list.append(self._pos_to_proximity(self.pos_0, reuse = True)) 

        self.val_postX_hidden = self._MPNN(self.postX_edge_wgt, self.priorZ_embed, name = 'postX', reuse = True)
        self.val_pos_0 = self._g_nn(self.val_postX_hidden, self.node, 3, name = 'postX', reuse = True)
        self.val_pos_proximity_list=[]
        self.val_pos_proximity_list.append(self._pos_to_proximity(self.val_pos_0, reuse = True))
        for i in range(100):
            self.val_pos_0 = self._NPE(self.val_pos_0, self.val_pos_proximity_list[-1], self.val_proximity_pred, reuse = True)#  
            self.val_pos_proximity_list.append(self._pos_to_proximity(self.val_pos_0, reuse = True)) 
        
        
        # functions for further NPE
        self.pos_i = tf.placeholder(tf.float32, [self.batch_size, self.n_max, 3])
        self.proximity_pred_i = tf.placeholder(tf.float32, [self.batch_size, self.n_max, self.n_max])
        self.proximity_pos_i = self._pos_to_proximity(self.pos_i, reuse = True)
        
        self.pos_o = self._NPE(self.pos_i, self.proximity_pos_i, self.proximity_pred_i, reuse = True)
        
        self.cost_pos_i = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.proximity_pos_i, self.proximity), [1, 2]) ) / 2


        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        
    
    def train(self, D1_t, D2_t, D3_t, D4_t, D1_v, D2_v, D3_v, D4_v, load_path = None, save_path = None):

        alpha = 10.
        beta = 10.

        # objective functions
        cost_KLDZ = tf.reduce_mean( tf.reduce_sum( self._KLD(self.postZ_mu, self.postZ_lsgms, self.priorZ_mu, self.priorZ_lsgms), [1, 2]) ) # posterior - prior
        cost_KLD0 = tf.reduce_mean( tf.reduce_sum( self._KLD_zero(self.priorZ_mu, self.priorZ_lsgms), [1, 2]) ) # posterior - prior
        
        cost_proximity = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.proximity_pred, self.proximity), [1, 2]) ) / 2                
        cost_npe_list = [tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.pos_proximity_list[i], self.proximity), [1, 2]) ) / 2 for i in range(len(self.pos_proximity_list))]
        cost_npe = tf.add_n(cost_npe_list[1:])/100
        cost_npe_last = cost_npe_list[-1]
        cost_npe_first = cost_npe_list[0]
        
        cost = cost_proximity + alpha * (cost_KLDZ + cost_npe_first + 0.01 * cost_KLD0 ) + beta * cost_npe #

        val_cost_proximity = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.val_proximity_pred, self.proximity), [1, 2]) ) / 2                
        val_cost_npe_list = [tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.val_pos_proximity_list[i], self.proximity), [1, 2]) ) / 2 for i in range(len(self.val_pos_proximity_list))]
        val_cost_npe = tf.add_n(val_cost_npe_list[1:])/100
        val_cost_npe_last = val_cost_npe_list[-1]
        val_cost_npe_first = val_cost_npe_list[0]
        
        val_cost = val_cost_proximity + alpha *  (cost_KLDZ + val_cost_npe_first) + beta * val_cost_npe  # 

        train_op = tf.train.AdamOptimizer().minimize(cost)
        

        self.sess.run(tf.global_variables_initializer())
        self.sess.graph.finalize()
        if load_path is not None:
            self.saver.restore( self.sess, load_path )

        # session 
        print '::: start training'
        n_batch = int(len(D1_t)/self.batch_size)
        n_batch_val = int(len(D1_v)/self.batch_size)
        np.set_printoptions(precision=5, suppress=True)
        
        valaggr = np.zeros(500)
        for epoch in range(500):
        
            [D1_t, D2_t, D3_t, D4_t] = self._permutation([D1_t, D2_t, D3_t, D4_t])
            
            for i in range(n_batch):
                start_ = i * self.batch_size
                end_ = start_ + self.batch_size
        
                trnresult = self.sess.run([train_op, cost, cost_KLDZ, cost_proximity, cost_npe_first, cost_npe_last, cost_KLD0],
                                    feed_dict = {self.node: D1_t[start_:end_], self.mask: D2_t[start_:end_], self.edge: D3_t[start_:end_], self.proximity: D4_t[start_:end_]})
                
                assert np.sum(np.isnan(trnresult[1:])) == 0
            
            valscores = np.zeros((n_batch_val, 5))
            valNPE = np.zeros((n_batch_val, 4))
            for i in range(n_batch_val): 
                start_ = i * self.batch_size
                end_ = start_ + self.batch_size
                     
                valresult = self.sess.run([val_cost, cost_KLDZ, val_cost_proximity, val_cost_npe_first, val_cost_npe_last, self.val_pos_0, self.val_proximity_pred],
                                    feed_dict = {self.node: D1_v[start_:end_], self.mask: D2_v[start_:end_], self.edge: D3_v[start_:end_], self.proximity: D4_v[start_:end_]})
        
                valscores[i, :] = valresult[:5]
                [val_a, val_b] = valresult[-2:]
                                    
                for j in range(4):
                    for _ in range(100):
                        val_a = self.next_pos(val_a, val_b, D2_v[start_:end_])
                        
                    valNPE[i, j] = self.eval_pos(val_a, D4_v[start_:end_], D2_v[start_:end_])
              
            valaggr[epoch] = np.mean(valNPE[:, -1])
            
            print '::: training epoch id', epoch, ':: --- val : ', np.mean(valscores, 0)
            print '::: NPE : ', np.mean(valNPE, 0), ':: --- aggr : ', valaggr[epoch]

            if epoch > 20 and np.min(valaggr[0:epoch-20]) < np.min(valaggr[epoch-20:epoch+1]) and valaggr[epoch] < np.min(valaggr[0:epoch]) * 1.01:
                print '::: terminate'
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
        inp = tf.layers.dropout(inp, rate = 0.2)
        inp = tf.layers.dense(inp, self.dim_h, activation = tf.nn.tanh)
    
        inp = tf.reshape(inp, [self.batch_size, self.n_max, self.dim_h])
        inp = tf.multiply(inp, self.mask) 
           
        return inp
    

    def _embed_latent(self, inp, name='', reuse=True): #[batch_size, n_max, dim_node]
    
        with tf.variable_scope('embed_latent'+name, reuse=reuse): 
           
            inp = tf.reshape(inp, [self.batch_size * self.n_max, int(inp.shape[2])])
            inp = tf.layers.dense(inp, self.dim_h, activation = tf.nn.tanh)
            inp = tf.reshape(inp, [self.batch_size, self.n_max, self.dim_h])
            inp = tf.multiply(inp, self.mask) 
           
        return inp
        
                   
    def _edge_nn(self, inp, name='', reuse=True): #[batch_size, n_max, n_max, dim_edge]
        
        with tf.variable_scope('edge_nn'+name, reuse=reuse):
        
            inp = tf.reshape(inp, [self.batch_size * self.n_max * self.n_max, int(inp.shape[3])])
            
            inp = tf.layers.dense(inp, 2 * self.dim_h, activation = tf.nn.sigmoid)
            inp = tf.layers.dropout(inp, rate = 0.2)
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
        
        
    def _g_nn(self, inp, node, outdim, name='', reuse=True): #[batch_size, n_max, -]
    
        with tf.variable_scope('g_nn'+name, reuse=reuse):
    
            #inp = tf.concat([inp, node], 2)
        
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
        
        
    def next_pos(self, pos_input, proximity_pred_input, mask_ref):
    
        return self.sess.run(self.pos_o, feed_dict = {self.pos_i: pos_input, self.proximity_pred_i: proximity_pred_input, self.mask: mask_ref}) 
    
    def eval_pos(self, pos_input, proximity_ref, mask_ref):
    
        return self.sess.run(self.cost_pos_i, feed_dict = {self.pos_i: pos_input, self.proximity: proximity_ref, self.mask: mask_ref})