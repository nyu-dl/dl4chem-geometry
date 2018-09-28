import tensorflow as tf
import tftraj.rmsd as rmsd
import pickle as pkl
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

data = 'QM9'
n_max = 9
n_batch = 5

print('::: load data')
#[D1, D2, D3, D4, D5] = pkl.load(open('./'+data+'_molvec_'+str(n_max)+'.p','rb'))
#D1 = D1.todense()
#D2 = D2.todense()
#D3 = D3.todense()
[suppl, molsmi] = pkl.load(open('./QM9_molset_'+str(n_max)+'.p','rb'))

#https://towardsdatascience.com/tensorflow-rmsd-using-tensorflow-for-things-it-was-not-designed-to-do-ada4c9aa0ea2
def mol_msd(frames, targets, masks):
    frames -= tf.reduce_mean(frames, axis=1, keep_dims=True)
    targets -= tf.reduce_mean(targets, axis=1, keep_dims=True)
    
    return tf.stack([rmsd.squared_deviation(frames[i], targets[i]) for i in range(frames.get_shape()[0])],0)

pos_ref=[]
pos_uff=[]
#pos_mask=[]
for t in range(15, 15+n_batch):
    print(t)
    posa=np.zeros((9,3))
    mol_ref=suppl[t]
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol_ref)
    Chem.rdmolops.AssignStereochemistry(mol_ref)
    n_est = mol_ref.GetNumHeavyAtoms()
    
    posa[:n_est]=mol_ref.GetConformer(0).GetPositions()
    pos_ref.append(np.copy(posa))

    ## baseline force field part
    mol_baseUFF = Chem.AddHs(mol_ref)
    AllChem.EmbedMolecule(mol_baseUFF, AllChem.ETKDG())
    AllChem.UFFOptimizeMoleculeConfs(mol_baseUFF, confId=0)
    mol_baseUFF=Chem.RemoveHs(mol_baseUFF)
    
    posa[:n_est]=mol_baseUFF.GetConformer(0).GetPositions()
    pos_uff.append(np.copy(posa))
    
    RMSD_UFF = AllChem.AlignMol(mol_baseUFF, mol_ref) 
    
    #pos_mask.append(np.copy(D2[t]))


#objective: align frame(pos_uff) to fit target(pos_ref)

pos_ref = np.array(pos_ref)
pos_uff = np.array(pos_uff)
#pos_mask = np.array(pos_mask)
print(pos_ref.shape, pos_uff.shape)

frame = tf.Variable(pos_uff, dtype=tf.float32)
target = tf.placeholder(tf.float32, [n_batch, n_max, 3])
#mask = tf.placeholder(tf.float32, [n_batch, n_max, 1])
msd = mol_msd(target, frame)

loss = tf.reduce_mean(msd)
train = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(500):
    res = sess.run([train, msd], feed_dict = {target: pos_ref})
    if step % 10==0:
        print(step, res[1])