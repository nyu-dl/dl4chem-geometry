import numpy as np
import pickle
from rdkit import Chem

n_min = 2
n_max = 50
smilist, mollist = [], []
suppl = Chem.SDMolSupplier('CSD.sdf')
j = 0
k = 0
for i, mol in enumerate(suppl):
    try:
        Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
        Chem.rdmolops.AssignStereochemistry(mol)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        na = mol.GetNumHeavyAtoms()
        pos = mol.GetConformer().GetPositions()
        if na==pos.shape[0] and na>=n_min and na<=n_max:
            smilist.append(smiles)
            mollist.append(mol)
            j += 1
        k += 1
    except:
        continue

print('j = {}'.format(j))
print('k = {}'.format(k))
print('i = {}'.format(i))
smilist=np.array(smilist)
mollist=np.array(mollist)
with open('CSD_molset_all.p','wb') as f:
    pickle.dump([mollist, smilist], f)
