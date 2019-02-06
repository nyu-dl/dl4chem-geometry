from rdkit.Chem import Descriptors, rdmolfiles
import pickle
import numpy as np
import pandas as pd

def get_statistics(molset_fname):
    with open(molset_fname, 'rb') as f:
        mols = pickle.load(f)[0]

    dataset_distinct_atoms = {}
    num_distinct_atoms = []
    num_bonds = []
    num_rotatable_bonds = []
    molecular_mass = []
    contains_symmetric_pair = []

    for mol in mols:
        atoms = mol.GetAtoms()
        symbols = [atom.GetSymbol() for atom in atoms]
        dataset_distinct_atoms.update(symbols)
        num_distinct_atoms.append(len({symbols}))
        num_bonds.append(len(mol.GetBonds()))
        num_rotatable_bonds.append(Descriptors.NumRotatableBonds(mol))
        molecular_mass.append(Descriptors.HeavyAtomMolWt(mol))
        canonical_ranking = rdmolfiles.CanonicalRankAtoms(mol, breakTies=False)
        symmetric_pair = int(len(canonical_ranking) != len(set(canonical_ranking)))
        contains_symmetric_pair.append(symmetric_pair)

    num_distinct_atoms_dataset = len(dataset_distinct_atoms)
    num_distinct_atoms = np.array(num_distinct_atoms)
    num_distinct_atoms = (num_distinct_atoms.mean(), num_distinct_atoms.std())
    num_bonds = np.array(num_bonds)
    num_bonds = (num_bonds.mean(), num_bonds.std())
    num_rotatable_bonds = np.array(num_rotatable_bonds)
    num_rotatable_bonds = (num_rotatable_bonds.mean(), num_rotatable_bonds.std())
    molecular_mass = np.array(molecular_mass)
    molecular_mass = (molecular_mass.mean(), molecular_mass.std())
    contains_symmetric_pair = np.array(contains_symmetric_pair).mean()

    return [num_distinct_atoms_dataset, num_distinct_atoms, num_bonds, num_rotatable_bonds, molecular_mass, contains_symmetric_pair]

if __name__ == '__main__':
    dataset_fnames = ['QM9_molset_all.p', 'COD_molset_all.p', 'CSD_molset_all.p']
    df = pd.DataFrame(columns = ['num_distinct_atoms_dataset', 'num_distinct_atoms', 'num_bonds', 'num_rotatable_bonds',
                                 'molecular_mass', 'contains_symmetric_pair'])
    for fname in dataset_fnames:
        df.append([fname.split('_')[0]] + get_statistics(molset_fname=fname))
    df.to_csv('statistics.csv')