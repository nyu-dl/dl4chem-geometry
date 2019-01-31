from __future__ import print_function

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import pickle as pkl
import copy
import sparse
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--virtual-node', action='store_true')
parser.add_argument('--loaddir', type=str, default='./')
parser.add_argument('--savedir', type=str, default='./')

args = parser.parse_args()

def to_onehot(val, cat, etc=0):

    onehot=np.zeros(len(cat))
    for ci, c in enumerate(cat):
        if val == c:
            onehot[ci]=1

    if etc==1 and np.sum(onehot)==0:
        print(val)

    return onehot


def atomFeatures(a, ri_a):

    def _ringSize_a(a, rings):
        onehot = np.zeros(6)
        aid = a.GetIdx()
        for ring in rings:
            if aid in ring and len(ring) <= 8:
                onehot[len(ring) - 3] += 1

        return onehot

    v1 = to_onehot(a.GetSymbol(), ['C','N','O','F','Cl','Br','I','S','B','Si','P','Te','Se','Ge','As',
          'Zn', 'Bi', 'Ga', 'Nb', 'Y', 'Sn', 'U', 'Tb', 'Mn', 'Ta', 'Ni', 'Sc', 'Pb', 'La', 'Re', 'Pt',
          'Cu', 'Sb', 'Tl', 'Eu', 'Tm', 'Th', 'Cr', 'Ce', 'Lu', 'Pu', 'Dy', 'S', 'Co', 'Cd', 'In', 'Nd',
          'Li', 'Be', 'Al', 'K', 'Yb', 'Zr', 'V', 'Sm', 'Np', 'Rb', 'Ag', 'Os', 'W', 'Ho', 'Mo', 'Fe',
          'Pr', 'Pd', 'Mg', 'Er', 'Tc', 'Xe', 'Hg', 'Gd', 'Hf', 'Rh', 'Ti', 'Ir', 'Ru', 'Au'], 1)
    v2 = to_onehot(str(a.GetHybridization()), ['SP','SP2','SP3','SP3D','SP3D2', 'UNSPECIFIED'], 1)

    v3 = [a.GetAtomicNum(), a.GetDegree(), a.GetFormalCharge(), a.GetTotalNumHs(), atom.GetImplicitValence(), a.GetNumRadicalElectrons(), int(a.GetIsAromatic())]
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


def bondFeatures(bbs, samering, shortpath):

    if len(bbs)==1:
        v1 =  to_onehot(str(bbs[0].GetBondType()), ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'], 1)
        v2 = to_onehot(str(bbs[0].GetStereo()), ['STEREOZ', 'STEREOE','STEREOANY','STEREONONE'], 1)
        v2 = v2[:2]
        v3 = [int(bbs[0].GetIsConjugated()), int(bbs[0].IsInRing()), samering, shortpath]
    else:
        v1 = np.zeros(4)
        v2 = np.zeros(2)
        v3 = [0, 0, samering, shortpath]

    return np.concatenate([v1,v2,v3], axis=0)


data='CSD'
n_min=2
n_max=50
atom_dim=98
edge_dim=10
virtual_node = args.virtual_node
print(virtual_node, flush=True)
if virtual_node:
    edge_dim += 1

[mollist, smilist] = pkl.load(open(args.loaddir+data+'_molset_all.p','rb'))

D1 = []
D2 = []
D3 = []
D4 = []
D5 = []
mollist2 = []
smilist2 = []
print(len(mollist), flush=True)
for i in range(len(mollist)):
    if i % 1000 == 0: print(i, flush=True)

    smi = smilist[i]
    mol = mollist[i]
    if '.' in smi: continue

    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    Chem.rdmolops.AssignStereochemistry(mol)

    if mol.GetNumHeavyAtoms() < n_min or mol.GetNumHeavyAtoms() > n_max:
        print('error')
        break

    n = mol.GetNumAtoms()
    ri = mol.GetRingInfo()
    ri_a = ri.AtomRings()

    pos = mol.GetConformer().GetPositions()

    if virtual_node:
        pos = np.vstack([pos, np.zeros(3)])
        assert n == pos.shape[0] - 1
    else:
        assert n==pos.shape[0]

    mollist2.append(mol)
    smilist2.append(smi)

    if virtual_node:
        node = np.zeros((n_max+1, atom_dim))
        mask = np.zeros((n_max+1, 1))
    else:
        node = np.zeros((n_max, atom_dim))
        mask = np.zeros((n_max, 1))

    for j in range(n):
        atom = mol.GetAtomWithIdx(j)
        node[j, :]=atomFeatures(atom, ri_a)
        mask[j, 0]=1
    if virtual_node:
        mask[n, 0] = 1

    if virtual_node:
        edge = np.zeros((n_max+1, n_max+1, edge_dim))
        edge[:n, n, 0] = 1
        edge[n, :n, 0] = 1
    else:
        edge = np.zeros((n_max, n_max, edge_dim))

    for j in range(n-1):
        for k in range(j+1, n):
            molpath = Chem.GetShortestPath(mol, j, k)
            shortpath = len(molpath) - 1
            assert shortpath>0
            samering = 0
            for alist in ri_a:
                if j in alist and k in alist:
                    samering = 1

            bond = [mol.GetBondBetweenAtoms(molpath[mm], molpath[mm+1]) for mm in range(shortpath)]

            if virtual_node:
                edge[j, k, :] = np.pad(bondFeatures(bond, samering, shortpath), (1, 0), 'constant')
                edge[k, j, :] = edge[j, k, :]
            else:
                edge[j, k, :] = bondFeatures(bond, samering, shortpath)
                edge[k, j, :] = edge[j, k, :]

    if virtual_node:
        proximity = np.zeros((n_max+1, n_max+1))
        proximity[:n+1, :n+1] = euclidean_distances(pos)

        pos2 = np.zeros((n_max+1, 3))
        pos2[:n+1] = pos
    else:
        proximity = np.zeros((n_max, n_max))
        proximity[:n, :n] = euclidean_distances(pos)

        pos2 = np.zeros((n_max, 3))
        pos2[:n] = pos

    D1.append(np.array(node, dtype=np.int8))
    D2.append(np.array(mask, dtype=np.int8))
    D3.append(np.array(edge, dtype=np.int8))
    D4.append(np.array(proximity))
    D5.append(np.array(pos2))

    #if len(D1)==66000:
    #    break

D1 = np.array(D1, dtype=np.int8)
D2 = np.array(D2, dtype=np.int8)
D3 = np.array(D3, dtype=np.int8)
D4 = np.array(D4)
D5 = np.array(D5)

print([D1.shape, D2.shape, D3.shape, D4.shape, D5.shape])
print([np.sum(np.isnan(D1)), np.sum(np.isnan(D2)), np.sum(np.isnan(D3)), np.sum(np.isnan(D4))])
print([D1.nbytes, D3.nbytes])

D1 = sparse.COO.from_numpy(D1)
D2 = sparse.COO.from_numpy(D2)
D3 = sparse.COO.from_numpy(D3)
print([D1.nbytes, D3.nbytes])

if virtual_node:
    molvec_fname = args.savedir + data+'_molvec_'+str(n_max)+'_vn'
    molset_fname = args.savedir + data + '_molset_' + str(n_max) + '_vn.p'
else:
    molvec_fname = args.savedir + data+'_molvec_'+str(n_max)
    molset_fname = args.savedir + data + '_molset_' + str(n_max) + '.p'

print(molvec_fname)
print(molset_fname)

chunk_size = int((len(D1)-1)/10)
for i in range(11):
    if i == 10:
        d1 = D1[i*chunk_size:]
        d2 = D2[i*chunk_size:]
        d3 = D3[i*chunk_size:]
        d4 = D4[i*chunk_size:]
        d5 = D5[i*chunk_size:]
    else:
        d1 = D1[i*chunk_size:(i+1)*chunk_size]
        d2 = D2[i*chunk_size:(i+1)*chunk_size]
        d3 = D3[i*chunk_size:(i+1)*chunk_size]
        d4 = D4[i*chunk_size:(i+1)*chunk_size]
        d5 = D5[i*chunk_size:(i+1)*chunk_size]
    with open(molvec_fname + '_{}.p'.format(i), 'wb') as f:
        pkl.dump([d1, d2, d3, d4, d5], f)


mollist2 = np.array(mollist2)
smilist2 = np.array(smilist2)

with open(molset_fname,'wb') as f:
    pkl.dump([mollist2, smilist2], f)
