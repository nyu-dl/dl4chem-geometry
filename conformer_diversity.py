import numpy as np
import argparse, os, pickle
from rdkit.Chem import rdMolAlign

parser = argparse.ArgumentParser()
parser.add_argument('--loaddir')
parser.add_argument('--savedir')
args = parser.parse_args()

if not os.path.exists(args.savedir):
    os.mkdir(args.savedir)

num_mols = 0
mean_rmsds, std_rmsds = [], []

for j in range(1, num_mols+1):
    print(j, flush=True)
    with open(os.path.join(args.loaddir, 'mol_{}'.format(j)), 'rb') as f:
        conformers = pickle.load(f)['pred_mmff']

    mol_rmsds = []
    for i in range(len(conformers) - 1):
        rmsds = []
        rdMolAlign.AlignMolConformers(conformers[i], conformers[i+1:], RMSlist=rmsds)
        mol_rmsds.extend(rmsds)

    mol_rmsds = np.array(mol_rmsds)
    mean_rmsd = mol_rmsds.mean()
    std_rmsd = mol_rmsds.std()
    mean_rmsds.append(mean_rmsd)
    std_rmsds.append(std_rmsd)

np.save(os.path.join(args.savedir, 'mean_rmsds.npy'), np.array(mean_rmsds))
np.save(os.path.join(args.savedir, 'std_rmsds.npy'), np.array(std_rmsds))