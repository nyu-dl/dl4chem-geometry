import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import sys, gc
import MPNN as MPNN
import sparse

# hyper-parameters
data = 'COD'
n_max = 30
dim_node = 33
dim_edge = 15
dim_h = 50
dim_f = 250
batch_size = 20
ntrn = int(40000 * 0.9)
nval = int(40000 * 0.1)
ntst = int(0.1 * 40000)

load_path = None
save_path = './'+data+'_NPE_newmodel_pred_'+str(n_max)+'_epoch.ckpt'

print '::: load data'
[D1, D2, D3, D4] = pkl.load(open('./'+data+'_molvec_'+str(n_max)+'.p','rb'))
D1 = D1.todense()
D2 = D2.todense()
D3 = D3.todense()

D1_trn = D1[:ntrn]
D2_trn = D2[:ntrn]
D3_trn = D3[:ntrn]
D4_trn = D4[:ntrn]
D1_val = D1[40000-nval:40000]
D2_val = D2[40000-nval:40000]
D3_val = D3[40000-nval:40000]
D4_val = D4[40000-nval:40000]
D1_tst = D1[40000:40000+ntst]
D2_tst = D2[40000:40000+ntst]
D3_tst = D3[40000:40000+ntst]
D4_tst = D4[40000:40000+ntst]
del D1, D2, D3, D4

model = MPNN.Model(data, n_max, dim_node, dim_edge, dim_h, dim_f, batch_size)
with model.sess:
    model.train(D1_trn, D2_trn, D3_trn, D4_trn, D1_val, D2_val, D3_val, D4_val, load_path, save_path)
    #model.saver.restore( model.sess, save_path )

    n_batch_tst = int(len(D1_tst)/model.batch_size)

    aggr_pos = []
    for iterid in range(10):
        print iterid

        all_pos_pred = []
        for i in range(n_batch_tst):
            start_ = i*batch_size
            end_ = start_+batch_size

            [pos_pred_, proximity_pred] = model.sess.run([model.val_pos_0, model.val_proximity_pred],
                                feed_dict = {model.node: D1_tst[start_:end_], model.mask: D2_tst[start_:end_], model.edge: D3_tst[start_:end_], model.proximity: D4_tst[start_:end_]})

            val_mask = D2_tst[start_:end_]
            for j in range(400):
                pos_pred_ = model.next_pos(pos_pred_, proximity_pred, val_mask)

            all_pos_pred.append(pos_pred_)

        aggr_pos.append(np.concatenate(all_pos_pred, 0))

    pkl.dump(aggr_pos, open('./'+data+'_NPE_newmodel_pred_'+str(n_max)+'.p','wb'))
