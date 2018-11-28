import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import copy, csv
from rdkit import Chem
from rdkit.Chem import AllChem
import pdb
import pickle as pkl
import os

import argparse
parser = argparse.ArgumentParser(description='Run baseline model')

parser.add_argument('--data', type=str, default='COD', choices=['COD','QM9'], help='which dataset to use')
parser.add_argument('--use-val', action='store_true', help='use validation set instead of test set')
parser.add_argument('--savedir', type=str, default='./', help='save directory of results')

args = parser.parse_args()

if args.data == "COD":
    n_max=50
    nval=3000
    ntst=3000

elif args.data == "QM9":
    n_max=9
    nval=5000
    ntst=5000

[suppl, molsmi] = pkl.load(open('./'+str(args.data)+'_molset_'+str(n_max)+'.p','rb'))
assert (len(suppl) == len(molsmi))
ntrn = len(suppl) - nval - ntst
# use validation set
if args.use_val:
    suppl = suppl[ntrn:ntrn+nval]
    molsmi = molsmi[ntrn:ntrn+nval]
    nmols = nval
# use test set
else:
    suppl = suppl[ntrn+nval:ntrn+nval+ntst]
    molsmi = molsmi[ntrn+nval:ntrn+nval+ntst]
    nmols = ntst

print (':::{}'.format("val" if args.use_val else "test"))
# save uff and mmff results
uff = []
mmff = []
for t in range(nmols):
    if t % 10 == 0 :
        print (t, nmols, molsmi[t])
    mol_ref=copy.deepcopy(suppl[t])

    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol_ref)
    Chem.rdmolops.AssignStereochemistry(mol_ref)

    mol_smi = Chem.MolFromSmiles(molsmi[t])

    n_est = mol_ref.GetNumHeavyAtoms()

    ttest_uff = []
    ttest_mmff = []
    for repid in range(10):
        mol_init_1=Chem.AddHs(mol_ref)
        mol_init_1.RemoveConformer(0)
        AllChem.EmbedMolecule(mol_init_1)
        
        try:
            ## baseline force field part with UFF
            mol_baseUFF = copy.deepcopy(mol_init_1)
            AllChem.UFFOptimizeMoleculeConfs(mol_baseUFF, confId=0)
            mol_baseUFF=Chem.RemoveHs(mol_baseUFF)
            RMS_UFF = AllChem.AlignMol(mol_baseUFF, mol_ref)
            
            ttest_uff.append([n_est, RMS_UFF])
        except:
            continue
            
        try:
            ## baseline force field part with MMFF
            mol_baseMMFF = copy.deepcopy(mol_init_1)
            AllChem.MMFFOptimizeMoleculeConfs(mol_baseMMFF, confId=0)
            mol_baseMMFF=Chem.RemoveHs(mol_baseMMFF)
            RMS_MMFF = AllChem.AlignMol(mol_baseMMFF, mol_ref)
            
            ttest_mmff.append([n_est, RMS_MMFF])
        except:
            continue
            
    #print (len(ttest))
    if len(ttest_uff) > 0:
        mean_ttest, std_ttest = np.mean(ttest_uff, 0), np.std(ttest_uff, 0)
        uff.append([mean_ttest[1], std_ttest[1], len(ttest_uff)])
        
    if len(ttest_mmff) > 0:
        mean_ttest, std_ttest = np.mean(ttest_mmff, 0), np.std(ttest_mmff, 0)
        mmff.append([mean_ttest[1], std_ttest[1], len(ttest_mmff)])

print ("UFF results")
print (np.mean(np.array(uff)[:,0]), np.mean(np.array(uff)[:,1]))
print ("MMFF results")
print (np.mean(np.array(mmff)[:,0]), np.mean(np.array(mmff)[:,1]))

f_name = '_val_' if args.use_val else '_test_'
f_name = args.data + f_name
pkl.dump(np.array(uff), open(os.path.join(args.savedir, f_name + 'uff.p'), 'wb'))
pkl.dump(np.array(mmff), open(os.path.join(args.savedir, f_name + 'mmff.p'), 'wb'))
