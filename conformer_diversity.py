import numpy as np
import argparse, copy, os, pickle
from rdkit.Chem import rdMolAlign
from rdkit import Chem

parser = argparse.ArgumentParser()
parser.add_argument('--loaddir')
parser.add_argument('--savedir')
parser.add_argument('--data', choices=['CSD', 'COD', 'QM9'])
parser.add_argument('--results-type', choices=['etkdg_mmff', 'mpnn', 'mpnn_mmff'])
args = parser.parse_args()

root = './'
if not os.path.exists(args.savedir):
    os.mkdir(args.savedir)

num_conformers = [100]
data = args.data
if data == 'CSD':
    data_path = root + 'CSD_mol/CSD_molset_tst.p'
    with open(data_path, 'rb') as f:
        all_mols = pickle.load(f)
    ntst = 3000
else:
    if data == 'COD':
        data_path = root + 'COD_molset_50.p'
        ntst = 3000
    elif data == 'QM9':
        data_path = root + 'QM9_molset_9.p'
        ntst = 5000
    with open(data_path, 'rb') as f:
        all_mols = pickle.load(f)[0]
    all_mols = all_mols[-ntst:]
all_mols = np.array(all_mols)

def get_num_atoms(coords):
    return int(np.any(coords != 0, axis=1).sum())

mean_rmsds, median_rmsds, std_rmsds, heavy_atoms = [], [], [], []

for j in range(1, ntst+1):
    print(j, flush=True)
    try:
        if args.results_type == 'mpnn_mmff':
            with open(os.path.join(args.loaddir, 'mol_{}.p'.format(j)), 'rb') as f:
                info = pickle.load(f)
                mols = info['pred_mmff']
                ha = info['n_heavy_atoms']
        else:
            if args.results_type == 'mpnn':
                with open(os.path.join(args.loaddir, 'mol_{}_neuralnet.p'.format(j)), 'rb') as f:
                    info = pickle.load(f)
                    coords = info['pred']
                ha = get_num_atoms(coords[0])
            elif args.results_type == 'etkdg_mmff':
                with open(os.path.join(args.loaddir, 'mol_{}.p'.format(j)), 'rb') as f:
                    info = pickle.load(f)
                    coords = info['pred_mmff']
                ha = info['n_heavy_atoms']
            mol = all_mols[j]
            mol.RemoveAllConformers()
            mols = []
            for conformer_coords in coords:
                num_atoms = int(np.all(conformer_coords != 0, axis=1).sum())
                conformer = Chem.Conformer(num_atoms)
                for atom_num in range(num_atoms):
                    conformer.SetAtomPosition(atom_num, conformer_coords[atom_num].tolist())
                current_mol = copy.deepcopy(mol)
                current_mol.AddConformer(conformer)
                mols.append(current_mol)

        mol_rmsds = []
        rmsds = []
        for i in range(len(mols) - 1):
            for k in range(i, len(mols)):
                rmsd = rdMolAlign.AlignMol(mols[k], mols[i])
                rmsds.append(rmsd)
            #rdMolAlign.AlignMolConformers(mol, confIds=range(i, len(conformers)), RMSlist=rmsds)

        mol_rmsds = np.array(rmsds)
        mean_rmsd = mol_rmsds.mean()
        median_rmsd = np.median(mol_rmsds)
        std_rmsd = mol_rmsds.std()
        mean_rmsds.append(mean_rmsd)
        median_rmsds.append(median_rmsd)
        std_rmsds.append(std_rmsd)
        heavy_atoms.append(ha)
    except:
        print('Failed', flush=True)

np.save(os.path.join(args.savedir, 'mean_rmsds.npy'), np.array(mean_rmsds))
np.save(os.path.join(args.savedir, 'std_rmsds.npy'), np.array(std_rmsds))
np.save(os.path.join(args.savedir, 'median_rmsds.npy'), np.array(median_rmsds))
np.save(os.path.join(args.savedir, 'heavy_atoms.npy'), np.array(heavy_atoms))
