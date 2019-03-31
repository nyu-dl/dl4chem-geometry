import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import copy, csv
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalForceFields
from rdkit.Chem import rdDistGeom as molDG
from rdkit.Chem.Pharm3D.EmbedLib import EmbedMol
import rdkit.DistanceGeometry as DG
import rdkit.Geometry as Geometry
from rdkit.Chem.rdtrajectory import Snapshot, Trajectory
import pdb
import pickle as pkl
import os
import time
import argparse
import getpass
import logging

from collections import OrderedDict

parser = argparse.ArgumentParser(description='Run baseline model')

parser.add_argument('--data', type=str, default='COD', choices=['COD','QM9', 'CSD'], help='which dataset to use')
parser.add_argument('--use-val', action='store_true', help='use validation set instead of test set')
parser.add_argument('--savedir', type=str, default='./', help='save directory of results')
parser.add_argument('--nn-path', type=str, default=None, help='path to neural net results')
parser.add_argument('--max-random-seed', type=int, default=100000, help='range of random seed to sample conformers from')
parser.add_argument('--max-attempts', type=int, default=6, help='max attempts for embedding bounds matrix')
parser.add_argument('--max-iters', type=int, default=200, help='max iters for MMFF and UFF (use default rdkit values)')
parser.add_argument('--min-mol-id', type=int, default=0, help='run experiment starting from this molecule id')
parser.add_argument('--max-mol-id', type=int, default=-1, help='run experiment up to this molecule id')
parser.add_argument('--n-confs', type=int, default=100, help='number of conformations to use')
parser.add_argument('--savepermol', action='store_true', help='save embed and mmff and uff results per molecule')
parser.add_argument('--ignore-saved', action='store_true', help='ignore molecules that already been saved')

args = parser.parse_args()

def data_path():
    """Path to data depending on user launching the script"""
    if getpass.getuser() == "mansimov":
        if os.uname().nodename == "mansimov-desktop":
            return "./data/"
        else:
            return "/misc/kcgscratch1/ChoGroup/mansimov/seokho_drive_datasets/"
    if getpass.getuser() == "em3382":
        return "/scratch/em3382/seokho_drive_datasets/"
    else:
        return "./"

# setup logger settings
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

if args.data == "COD":
    n_max=50
    nval=3000
    ntst=3000

elif args.data == "QM9":
    n_max=9
    nval=5000
    ntst=5000

elif args.data == 'CSD':
    n_max = 50
    nval = 3000
    ntst = 3000

if args.data == 'CSD':
    if args.use_val:
        suppl = pkl.load(open('CSD_mol/CSD_molset_val.p','rb'))
        if args.max_mol_id == -1:
            args.max_mol_id = nval
    else:
        suppl = pkl.load(open('CSD_mol/CSD_molset_tst.p','rb'))
        if args.max_mol_id == -1:
            args.max_mol_id = ntst
else:
    [suppl, molsmi] = pkl.load(open(data_path()+str(args.data)+'_molset_'+str(n_max)+'.p','rb'))
    assert (len(suppl) == len(molsmi))
    ntrn = len(suppl) - nval - ntst
    # use validation set

    if args.use_val:
        suppl = suppl[ntrn:ntrn+nval]
        molsmi = molsmi[ntrn:ntrn+nval]
        if args.max_mol_id == -1:
            args.max_mol_id = nval
    # use test set
    else:
        suppl = suppl[ntrn+nval:ntrn+nval+ntst]
        molsmi = molsmi[ntrn+nval:ntrn+nval+ntst]
        if args.max_mol_id == -1:
            args.max_mol_id = ntst


data_split = '_val_' if args.use_val else '_test_'

if args.savepermol:
    # create dir
    dir_name = os.path.join(args.savedir, args.data, data_split)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if args.savepermol:
        if not os.path.exists(os.path.join(dir_name, "mols")):
            os.makedirs(os.path.join(dir_name, "mols"))

logger.info ('running loop starting from {} up to {}'.format(args.min_mol_id, args.max_mol_id))
#trajectories = []

for t in range(args.min_mol_id, args.max_mol_id):
    # load neural net predictions
    if args.nn_path:
        mol_nn_pred = pkl.load(open('{}/{}/{}/mol_{}_neuralnet.p'.format(args.nn_path, args.data, data_split, t), 'rb'))
        mol_nn_pred = mol_nn_pred['pred']

    # check if already exists
    if args.ignore_saved:
        if os.path.isfile(os.path.join(dir_name, "mols", 'mol_{}.p'.format(t))):
            continue

    n_confs = args.n_confs

    # get info about molecule
    mol_ref=copy.deepcopy(suppl[t])
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol_ref)
    Chem.rdmolops.AssignStereochemistry(mol_ref)

    #mol_smi = Chem.MolFromSmiles(molsmi[t])

    n_est = mol_ref.GetNumHeavyAtoms()
    n_rot_bonds = AllChem.CalcNumRotatableBonds(mol_ref)

    try:
        logger.info ('Molecule {} SMILES string'.format(t))
        logger.info (molsmi[t])
    except:
        pass

    # get original bounds matrix
    mol_init_hs = copy.deepcopy(mol_ref)
    #mol_init_hs = copy.deepcopy(mol_ref)
    mol_init_hs.RemoveAllConformers()

    # embed/uff/mmff stats
    ttest_uff = []
    pred_uff = []
    ttest_mmff = []
    pred_mmff = []
    ttest_embed = []
    pred_embed = []

    for repid in range(n_confs):
        logger.info ('conformer # {}'.format(repid))

        mol_init_hs = copy.deepcopy(mol_ref)
        mol_init_hs.RemoveAllConformers()

        # use neural net predictions to modify bounds matrix
        if args.nn_path:
            coords = mol_nn_pred[repid]

        # add the coordinates into the conformer
        if coords is None:
            logger.info ('failed conformer # {}'.format(repid))
            continue

        conf = Chem.Conformer(n_est)
        conf.SetId(0)
        for i in range(n_est):
            conf.SetAtomPosition(i, coords[i].tolist())

        mol_init_hs.AddConformer(conf)

        mol_init_hs = Chem.AddHs(mol_init_hs, addCoords=True)

        mol_init_embed = copy.deepcopy(mol_init_hs)
        # some weird issue
        # Can't kekulize mol.  Unkekulized atoms: 6 7 8
        try:
            mol_init_embed=Chem.RemoveHs(mol_init_embed)
        except:
            logger.info ('Cant kekulize mol issue')
            continue

        RMS_EMBED = AllChem.AlignMol(mol_init_embed, mol_ref)
        pred_embed.append(mol_init_embed)
        ttest_embed.append(RMS_EMBED)

        # run MMFF/UFF on top of it
        try:
            ## baseline force field part with UFF
            mol_baseUFF = copy.deepcopy(mol_init_hs)
            uff_out = AllChem.UFFOptimizeMolecule(mol_baseUFF, maxIters=args.max_iters)
            mol_baseUFF=Chem.RemoveHs(mol_baseUFF)
            RMS_UFF = AllChem.AlignMol(mol_baseUFF, mol_ref)
            pred_uff.append(mol_baseUFF)
            ttest_uff.append(RMS_UFF)
        except:
            continue
        try:
            ## baseline force field part with MMFF
            mol_baseMMFF = copy.deepcopy(mol_init_hs)
            mmff_out = AllChem.MMFFOptimizeMolecule(mol_baseMMFF, maxIters=args.max_iters)
            mol_baseMMFF=Chem.RemoveHs(mol_baseMMFF)
            RMS_MMFF = AllChem.AlignMol(mol_baseMMFF, mol_ref)
            pred_mmff.append(mol_baseMMFF)
            ttest_mmff.append(RMS_MMFF)
        except:
            continue

    # save results per molecule
    if args.savepermol:
        mol_info = {'n_heavy_atoms': n_est, 'n_rot_bonds': n_rot_bonds}
        if len(ttest_embed) > 0:
            mol_info["embed"] = ttest_embed
            mol_info["pred_embed"] = pred_embed

        if len(ttest_mmff) > 0:
            mol_info["mmff"] = ttest_mmff
            mol_info["pred_mmff"] = pred_mmff

        if len(ttest_uff) > 0:
            mol_info["uff"] = ttest_uff
            mol_info["pred_uff"] = pred_uff

        pkl.dump(mol_info, \
            open(os.path.join(dir_name, "mols", 'mol_{}.p'.format(t)), 'wb'))
