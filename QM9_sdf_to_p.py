from rdkit import Chem
from rdkit.Chem import AllChem
import csv
import numpy as np
import pickle as pkl

data = 'QM9'
n_min = 2
n_max = 9
suppl = Chem.SDMolSupplier('./QM9_gdb9.sdf')

smilist=[]
mollist=[]
for i, mol in enumerate(suppl):
    try:
        Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
        Chem.rdmolops.AssignStereochemistry(mol)

        smi=Chem.MolToSmiles(mol, isomericSmiles=True)
        if '.' in smi:
            continue
    except:
        continue
        
    na = mol.GetNumHeavyAtoms()
    pos = mol.GetConformer().GetPositions()
    if na==pos.shape[0] and na>=n_min and na<=n_max:
        mollist.append(mol)
        smilist.append(smi)
        print(len(smilist), i)

smilist=np.array(smilist)
mollist=np.array(mollist)
permid=np.random.permutation(len(mollist))
mollist=mollist[permid]
smilist=smilist[permid]
pkl.dump([mollist, smilist], open(data+'_molset_all.p','wb'))