import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import copy, csv
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit.Geometry as Geometry
import pdb
import pickle as pkl
import os
import getpass
import logging
import scipy
logging.basicConfig(level=logging.INFO)


import argparse
parser = argparse.ArgumentParser(description='Run baseline model')

parser.add_argument('--data', type=str, default='COD', choices=['COD','QM9'], help='which dataset to use')
parser.add_argument('--use_val', action='store_true', help='use validation set instead of test set')
parser.add_argument('--norm', action='store_true', help='normalize neural net predictions (subtract mean)')
parser.add_argument('--savedir', type=str, default='./', help='save directory of results')
parser.add_argument('--nn_path', type=str, default=None, help='path to neural net results')
parser.add_argument('--saveall', action='store_true', help='save mmff and uff rmsd per molecule')
parser.add_argument('--debug', action='store_true', help='debug the results')
parser.add_argument('--savepermol', action='store_true', help='save mmff and uff rmsd per molecule in separate repos')
parser.add_argument('--max_iters', type=int, default=200, help='number of iterations for baselines to run')
parser.add_argument('--num_reps', type=int, default=10, help='number of repetitions')

args = parser.parse_args()

if args.data == "COD":
    n_max=50
    nval=3000
    ntst=3000

elif args.data == "QM9":
    n_max=9
    nval=5000
    ntst=5000

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

[suppl, molsmi] = pkl.load(open(data_path()+str(args.data)+'_molset_'+str(n_max)+'.p','rb'))
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

logging.info (':::{}'.format("val" if args.use_val else "test"))
# save uff and mmff results
uff = []
mmff = []
if args.saveall:
    all_uff = []
    all_mmff = []

subdata_name = '_val_' if args.use_val else '_test_'
f_name = 'rep_{}_iter_{}'.format(args.num_reps, args.max_iters)
args.savedir = os.path.join(args.savedir, args.data, subdata_name, f_name)
if not os.path.exists(args.savedir):
    os.makedirs(args.savedir)
    if args.savepermol:
        os.makedirs(os.path.join(args.savedir, "mols"))

if args.nn_path:
    args.nn_path = os.path.join(args.nn_path, args.data, '_val_' if args.use_val else '_test_')
uff_converged = []
mmff_converged = []

for t in range(nmols):
    if t % 10 == 0 :
        logging.info("{}, {}, {}".format(t, nmols, molsmi[t]))
    mol_ref=copy.deepcopy(suppl[t])

    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol_ref)
    Chem.rdmolops.AssignStereochemistry(mol_ref)

    mol_smi = Chem.MolFromSmiles(molsmi[t])

    n_est = mol_ref.GetNumHeavyAtoms()
    n_rot_bonds = AllChem.CalcNumRotatableBonds(mol_ref)

    ttest_uff = []
    pred_uff = []
    ttest_mmff = []
    pred_mmff = []
    embed_success = 0

    if args.nn_path:
        nn_mol = pkl.load(open(os.path.join(args.nn_path, "mol_{}_neuralnet.p".format(t)), 'rb'))
        nn_pred = nn_mol['pred']

    for repid in range(args.num_reps):
        if args.debug:
            if repid % 10 == 0:
                logging.info ("repid {} ; mol id {}".format(repid, t))
        if args.nn_path:
            n_est = mol_ref.GetNumAtoms()
            coord_map = {}
            for k in range(n_est):
                nn_pred_k = nn_pred[repid][k].tolist()
                point_k = Geometry.Point3D(*nn_pred_k)
                coord_map[k] = point_k

            mol_init_1=Chem.AddHs(mol_ref)
            mol_init_1.RemoveConformer(0)
            embed_result = AllChem.EmbedMolecule(mol_init_1, coordMap=coord_map,\
                            useRandomCoords=True, enforceChirality=False,\
                            ignoreSmoothingFailures=True)
            if embed_result == 0:
                embed_success += 1
            else:
                pdb.set_trace()

        else:
            mol_init_1=Chem.AddHs(mol_ref)
            mol_init_1.RemoveConformer(0)
            AllChem.EmbedMolecule(mol_init_1)

        try:
            ## baseline force field part with UFF
            mol_baseUFF = copy.deepcopy(mol_init_1)
            uff_out = AllChem.UFFOptimizeMolecule(mol_baseUFF, maxIters=args.max_iters)
            uff_converged.append(1 - uff_out)
            mol_baseUFF=Chem.RemoveHs(mol_baseUFF)
            RMS_UFF = AllChem.AlignMol(mol_baseUFF, mol_ref)
            pred_uff.append(mol_baseUFF)
            #pred_uff.append(mol_baseUFF.GetConformer().GetPositions())
            ttest_uff.append([n_est, RMS_UFF])
        except:
            continue

        try:
            ## baseline force field part with MMFF
            mol_baseMMFF = copy.deepcopy(mol_init_1)
            mmff_out = AllChem.MMFFOptimizeMolecule(mol_baseMMFF, maxIters=args.max_iters)
            mmff_converged.append(1 - mmff_out)
            mol_baseMMFF=Chem.RemoveHs(mol_baseMMFF)
            RMS_MMFF = AllChem.AlignMol(mol_baseMMFF, mol_ref)
            pred_mmff.append(mol_baseMMFF)
            #pred_mmff.append(mol_baseMMFF.GetConformer().GetPositions())
            ttest_mmff.append([n_est, RMS_MMFF])
        except:
            continue

    if args.debug:
        logging.info ('mol id {}'.format(t))
        logging.info ('num of successful embeddings {}'.format(embed_success))
        logging.info ('num of uff mols {}, mmff mols {}'.format(len(ttest_uff), len(ttest_mmff)))

    if args.saveall:
        if len(ttest_uff) > 0:
            all_uff.append(np.array(ttest_uff))
        if len(ttest_mmff) > 0:
            all_mmff.append(np.array(ttest_mmff))

    # save results per molecule
    if args.savepermol:
        mol_info = {'n_heavy_atoms': n_est, 'n_rot_bonds': n_rot_bonds}
        if len(ttest_mmff) > 0:
            mol_info["mmff"] = np.array(ttest_mmff)
            mol_info["pred_mmff"] = np.array(pred_mmff)
        if len(ttest_uff) > 0:
            mol_info["uff"] = np.array(ttest_uff)
            mol_info["pred_uff"] = np.array(pred_uff)
        pkl.dump(mol_info, \
            open(os.path.join(args.savedir, "mols", 'mol_{}.p'.format(t)), 'wb'))
        #pkl.dump({'mmff': np.array(ttest_mmff), 'uff': np.array(ttest_uff), 'n_heavy_atoms': n_est},\
        #        open(os.path.join(args.savedir, "mols", 'mol_{}.p'.format(t)), 'wb'))

    if len(ttest_uff) > 0:
        mean_ttest, std_ttest = np.mean(ttest_uff, 0), np.std(ttest_uff, 0)
        uff.append([mean_ttest[1], std_ttest[1], len(ttest_uff)])

    if len(ttest_mmff) > 0:
        mean_ttest, std_ttest = np.mean(ttest_mmff, 0), np.std(ttest_mmff, 0)
        mmff.append([mean_ttest[1], std_ttest[1], len(ttest_mmff)])

uff_converged = np.array(uff_converged)
mmff_converged = np.array(mmff_converged)
logging.info ("UFF results")
logging.info ("{}, {}".format(np.mean(np.array(uff)[:,0]), np.mean(np.array(uff)[:,1])))
logging.info ("Percent converged {}".format(float(uff_converged.sum()) / float(uff_converged.shape[0])))
logging.info ("MMFF results")
logging.info ("{}, {}".format(np.mean(np.array(mmff)[:,0]), np.mean(np.array(mmff)[:,1])))
logging.info ("Percent converged {}".format(float(mmff_converged.sum()) / float(mmff_converged.shape[0])))
pkl.dump(np.array(uff), open(os.path.join(args.savedir, 'uff.p'), 'wb'))
pkl.dump(np.array(mmff), open(os.path.join(args.savedir, 'mmff.p'), 'wb'))
if args.saveall:
    pkl.dump(np.array(all_uff), open(os.path.join(args.savedir, 'all_uff.p'), 'wb'))
    pkl.dump(np.array(all_mmff), open(os.path.join(args.savedir, 'all_mmff.p'), 'wb'))
