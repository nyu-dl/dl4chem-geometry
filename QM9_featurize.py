from __future__ import print_function

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pickle as pkl
import copy
import sparse

def to_onehot(val, cat, etc=0):

    onehot=np.zeros(len(cat))
    for ci, c in enumerate(cat):
        if val == c:
            onehot[ci]=1

    if etc==1 and np.sum(onehot)==0:
        print(val)

    return onehot


def atomFeatures(a, ri, ri_a):

    def _ringSize_a(a, rings):
        onehot = np.zeros(6)
        aid = a.GetIdx()
        for ring in rings:
            if aid in ring and len(ring) <= 8:
                onehot[len(ring) - 3] += 1

        return onehot

    v1 = to_onehot(a.GetSymbol(), ['C','N','O','F'], 1)
    v2 = to_onehot(a.GetHybridization(), [Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3], 1)

    v3 = [a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(), a.GetTotalNumHs(), int(a.GetIsAromatic())]
    v4 = _ringSize_a(a, ri_a)

    v5 = np.zeros(3)
    try:
        tmp = to_onehot(a.GetProp('_CIPCode'), ['R','S'], 1)
        v5[0] = tmp[0]
        v5[1] = tmp[1]
    except:
        v5[2]=1

    v5 = v5[:2]

    return np.concatenate([v1,v2,v3,v4,v5], axis=0)


def bondFeatures(bbs, ri, samering, shortpath):

    if len(bbs)==1:
        v1 = to_onehot(bbs[0].GetBondType(), [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC], 1)
        v2 = to_onehot(str(bbs[0].GetStereo()), ['STEREOZ', 'STEREOE','STEREOANY','STEREONONE'], 1)
        v3 = [int(bbs[0].GetIsConjugated()), shortpath]
    else:
        v1 = np.zeros(4)
        v2 = np.zeros(4)
        v3 = [0, shortpath]

    v4 = samering
    v2 = v2[:3]


    return np.concatenate([v1,v2,v3,v4], axis=0)


data='QM9'
n_min=2
n_max=9
atom_dim=20
edge_dim=15

[mollist, smilist] = pkl.load(open('./'+data+'_molset_all.p','rb'))

D1 = []
D2 = []
D3 = []
D4 = []
D5 = []
mollist2 = []
smilist2 = []
for i in range(len(mollist)):
    if i % 500 == 0:
        print (i, len(mollist))
    smi = smilist[i]
    mol = mollist[i]

    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    Chem.rdmolops.AssignStereochemistry(mol)

    if mol.GetNumHeavyAtoms() < n_min or mol.GetNumHeavyAtoms() > n_max:
        print('error')
        break

    n = mol.GetNumAtoms()
    ri = mol.GetRingInfo()
    ri_a = ri.AtomRings()

    pos = mol.GetConformer().GetPositions()
    assert n==pos.shape[0]

    mollist2.append(mol)
    smilist2.append(smi)

    node = np.zeros((n_max, atom_dim))
    mask = np.zeros((n_max, 1))

    for j in range(n):
        atom = mol.GetAtomWithIdx(j)
        node[j, :]=atomFeatures(atom, ri, ri_a)
        mask[j, 0]=1

    edge = np.zeros((n_max, n_max, edge_dim))
    for j in range(n-1):
        for k in range(j+1, n):
            molpath = Chem.GetShortestPath(mol, j, k)
            shortpath = len(molpath) - 1
            assert shortpath>0

            samering = np.zeros(6)
            for alist in ri_a:
                if j in alist and k in alist and len(alist) <= 8:
                    samering[len(alist) - 3] += 1

            bond = [mol.GetBondBetweenAtoms(molpath[mm], molpath[mm+1]) for mm in range(shortpath)]
            edge[j, k, :] = bondFeatures(bond, ri, samering, shortpath)
            edge[k, j, :] = bondFeatures(bond, ri, samering, shortpath)

    proximity = np.zeros((n_max, n_max))
    proximity[:n, :n] = euclidean_distances(pos)

    pos2 = np.zeros((n_max, 3))
    pos2[:n] = pos

    D1.append(np.array(node, dtype=int))
    D2.append(np.array(mask, dtype=int))
    D3.append(np.array(edge, dtype=int))
    D4.append(np.array(proximity))
    D5.append(np.array(pos2))

    if len(D1)==110000:
        break

D1 = np.array(D1, dtype=int)
D2 = np.array(D2, dtype=int)
D3 = np.array(D3, dtype=int)
D4 = np.array(D4)
D5 = np.array(D5)

print([D1.shape, D2.shape, D3.shape, D4.shape, D5.shape])
print([np.sum(np.isnan(D1)), np.sum(np.isnan(D2)), np.sum(np.isnan(D3)), np.sum(np.isnan(D4))])
print([D1.nbytes, D3.nbytes])

D1 = sparse.COO.from_numpy(D1)
D2 = sparse.COO.from_numpy(D2)
D3 = sparse.COO.from_numpy(D3)
print([D1.nbytes, D3.nbytes])

pkl.dump([D1, D2, D3, D4, D5], open(data+'_molvec_'+str(n_max)+'.p','wb'))

mollist2 = np.array(mollist2)
smilist2 = np.array(smilist2)

pkl.dump([mollist2, smilist2], open(data+'_molset_'+str(n_max)+'.p','wb'))
