import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import copy, csv
from rdkit import Chem
from rdkit.Chem import AllChem  
    
n_max=30
ntrn=40000
ntst=4000

f = open('COD_NPE_res_newmodel_log_'+str(n_max)+'.csv', 'wb')
wr = csv.writer(f)
    
poslist=pkl.load(open('./COD_NPE_newmodel_pred_'+str(n_max)+'.p','rb'))
print len(poslist[0])
assert len(poslist[0])==ntst
[suppl, molsmi] = pkl.load(open('./COD_molset_'+str(n_max)+'.p','rb'))

print ':::test'
for t in range(ntst):
    
    mol_ref=copy.deepcopy(suppl[ntrn+t])
    
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol_ref)
    Chem.rdmolops.AssignStereochemistry(mol_ref)

    mol_smi = Chem.MolFromSmiles(molsmi[ntrn+t])
    
    n_est = mol_ref.GetNumHeavyAtoms()
    
    ttest = []
    for repid in range(10):
        
        for _ in range(5):
            try:
                    
                mol_init_1=Chem.AddHs(mol_ref)
                mol_init_1.RemoveConformer(0)
                AllChem.EmbedMolecule(mol_init_1)
                
                ## proposed model
                NPE_pos = poslist[repid][t]
                NPE_cf=Chem.rdchem.Conformer(n_est)
                for i in range(n_est):
                    NPE_cf.SetAtomPosition(i, [float(NPE_pos[i,0]),float(NPE_pos[i,1]),float(NPE_pos[i,2])])
        
                mol_neural=copy.deepcopy(mol_ref)
                mol_neural.RemoveConformer(0)
                mol_neural.AddConformer(NPE_cf)#
                RMS_neural = AllChem.GetBestRMS(mol_neural, mol_ref) 

                ## baseline force field part
                mol_baseUFF = copy.deepcopy(mol_init_1)
                AllChem.UFFOptimizeMoleculeConfs(mol_baseUFF, confId=0)
                mol_baseUFF=Chem.RemoveHs(mol_baseUFF)
                RMS_UFF = AllChem.GetBestRMS(mol_baseUFF, mol_ref)   
        
                mol_baseMMFF = copy.deepcopy(mol_init_1)
                AllChem.MMFFOptimizeMoleculeConfs(mol_baseMMFF, confId=0)
                mol_baseMMFF=Chem.RemoveHs(mol_baseMMFF)
                RMS_MMFF = AllChem.GetBestRMS(mol_baseMMFF, mol_ref) 
               
                ttest.append([n_est, RMS_neural, RMS_UFF, RMS_MMFF])
                
                break
                
            except:
                continue
                
    
    if len(ttest)>1:
        wr.writerow(np.concatenate([np.array([molsmi[ntrn+t]]), np.mean(ttest,0), np.std(ttest,0)])) 