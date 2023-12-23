# %%
import numpy as np
import pandas as pd
import sys
from .create_drug_feat import *
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch
import os
import random
import pickle
# %%
def map_indices(row, drug2index, CL2index):
    drug, cl = row.split('_')
    drug = int(drug)
    cl = int(cl)
    drug_index = drug2index[drug]  # Replace 'Unknown' with a default value as needed
    cl_index = CL2index[cl]  # Replace 'Unknown' with a default value as needed
    return f"{drug_index}_{cl_index}"
def process_CL_drug(train_idx):
    drug = [int(idx.split('_')[0]) for idx in train_idx]
    CL = [int(idx.split('_')[1]) for idx in train_idx]
    return drug, CL
def train_val_test_split(df):
    # Splitting into train_val and test sets (80% train_val, 20% test)
    train_val, test = train_test_split(df, test_size=0.2, random_state=42)

    # Further splitting train_val into train and val sets (82% train, 18% val of train_val)
    train, val = train_test_split(train_val, test_size=0.8, random_state=42)

    # Extracting the indices
    train_index = train.index
    val_index = val.index
    test_index = test.index

    return train_index, val_index, test_index

def preprocess_cell_data():
    saving_root = '/home/yurui/GDSC_2/GDSC2_dataset/'
    if not os.path.exists('/home/yurui/GDSC_2/GDSC2_dataset/index2CL.pickle'):
        print('Preprocessing cell line data...')
        GDSC_df = pd.read_csv('/home/yurui/GDSC_2/GDSC2_curve_data/GDSC2_curve_estimate_4_para.csv')
        drug_meta = pd.read_csv('/home/yurui/GDSC_2/GDSC2_dataset/meta_data/drug_meta.csv')
        cell_meta = pd.read_csv('/home/yurui/GDSC_2/GDSC2_dataset/meta_data/cell_meta.csv')
        GE_data =pd.read_csv('/home/yurui/GDSC_2/GDSC2_dataset/CL_feature/GE_final.csv')
        GE_df = GE_data.drop_duplicates(subset='COSMIC_ID', keep='first').set_index('COSMIC_ID')
        GE_df.reset_index(inplace=True)
        drug_meta.set_index('DRUG_ID', inplace=True)
        GDSC_df = GDSC_df.merge(drug_meta, on='DRUG_ID')
        GDSC_df = GDSC_df[GDSC_df['COSMIC_ID'].isin(GE_df['COSMIC_ID'])]
        GDSC_df['DRUGID_COSMICID'] = GDSC_df['DRUG_ID'].astype(str) + '_' + GDSC_df['COSMIC_ID'].astype(str)
        GDSC_df['DRUGID_COSMICID_MAXC'] = GDSC_df['DRUGID_COSMICID'] + '_' + GDSC_df['MAX_CONC'].astype(str)
        GDSC_df = GDSC_df.groupby('DRUGID_COSMICID').first().reset_index()
        mut_cols = [col for col in GDSC_df.columns if 'mut' in col]
        chr_cols = [col for col in GDSC_df.columns if 'chr' in col]
        cna_cols = [col for col in GDSC_df.columns if 'cna' in col]
        GDSC_df_copy = GDSC_df.copy()
        result_df = GDSC_df_copy[GDSC_df.columns[:26]]
        # result_df.set_index('DRUGID_COSMICID', inplace=True)
        mut_df = GDSC_df_copy[['COSMIC_ID'] + mut_cols].drop_duplicates().reset_index(drop=True)
        mut_df.set_index('COSMIC_ID', inplace=True)
        duplicated_indices = mut_df.index.duplicated(keep='first')
        mut_df = mut_df[~duplicated_indices]
        chr_df = GDSC_df_copy[['COSMIC_ID'] + chr_cols].drop_duplicates().reset_index(drop=True)
        chr_df.set_index('COSMIC_ID', inplace=True)
        cna_df = GDSC_df_copy[['COSMIC_ID'] + cna_cols].drop_duplicates().reset_index(drop=True)
        cna_df.set_index('COSMIC_ID', inplace=True)
        GE_df = GE_df.set_index('COSMIC_ID').loc[cna_df.index]
        ## Get the dictionary of the raw number and the index
        index2CL = {GE_df.index.get_loc(CL): CL for CL in GE_df.index}
        CL2index = {index2CL[index]:index for index in index2CL.keys()}
        index2drug = {id:drug for id,drug in enumerate(GDSC_df.DRUG_ID.unique())}
        drug2index = {index2drug[index]:index for index in index2drug.keys()}
        cna_df = cna_df.reindex(sorted(CL2index, key=CL2index.get))
        GE_df = GE_df.reindex(sorted(CL2index, key=CL2index.get))
        chr_df = chr_df.reindex(sorted(CL2index, key=CL2index.get))
        mut_df = mut_df.reindex(sorted(CL2index, key=CL2index.get))
        mut_tensor = torch.tensor(mut_df.values, dtype=torch.long)
        chr_tensor = torch.tensor(chr_df.values, dtype=torch.long)
        cna_tensor = torch.tensor(cna_df.values, dtype=torch.long)
        GE_tensor = torch.tensor(GE_df.values, dtype=torch.float)
        cell_meta = pd.read_csv('/home/yurui/GDSC_2/GDSC2_dataset/meta_data/cell_meta.csv')
        cell_meta.TCGA_DESC.fillna('UNCLASSIFIED', inplace=True)
        cell_meta = cell_meta[['COSMIC_ID', 'CANCER_TYPE']]
        cell_meta.set_index('COSMIC_ID', inplace=True)
        # cell_meta = cell_meta.loc[CL2index.keys()]
        cell_meta = cell_meta.reindex(sorted(CL2index, key=CL2index.get))
        cell_meta.loc[cell_meta.CANCER_TYPE == -1, 'CANCER_TYPE'] = 31
        CL_type_tensor = torch.tensor(cell_meta.CANCER_TYPE.values, dtype=torch.long)
        result_df = result_df.copy()
        result_df.loc[:,'drugindex_CLindex'] = result_df['DRUGID_COSMICID'].apply(map_indices, args=(drug2index, CL2index))
        result_df.set_index('drugindex_CLindex', inplace=True)
        y_cols = [f'norm_cells_{i+1}' for i in range(7)]
        label_tensor = torch.tensor(result_df[y_cols].values, dtype=torch.float)
        print('Saving the data...')
        result_df.to_csv(saving_root + 'DRP_data/DRP_df.csv')
        with open(saving_root + 'index2CL.pickle', 'wb') as handle:
            pickle.dump(index2CL, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(saving_root + 'CL2index.pickle', 'wb') as handle:
            pickle.dump(CL2index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(saving_root + 'index2drug.pickle', 'wb') as handle:
            pickle.dump(index2drug, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(saving_root + 'drug2index.pickle', 'wb') as handle:
            pickle.dump(drug2index, handle, protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(mut_tensor, saving_root + 'CL_feature/mut_tensor.pt')
        torch.save(chr_tensor, saving_root + 'CL_feature/chr_tensor.pt')
        torch.save(cna_tensor, saving_root + 'CL_feature/cna_tensor.pt')
        torch.save(GE_tensor, saving_root + 'CL_feature/GE_tensor.pt')
        torch.save(CL_type_tensor, saving_root + 'CL_feature/CL_type_tensor.pt')
        torch.save(label_tensor, saving_root + 'label_tensor.pt')
        
    else:
        with open(saving_root + 'index2CL.pickle', 'rb') as handle:
            index2CL = pickle.load(handle)

        with open(saving_root + 'CL2index.pickle', 'rb') as handle:
            CL2index = pickle.load(handle)

        with open(saving_root + 'index2drug.pickle', 'rb') as handle:
            index2drug = pickle.load(handle)

        with open(saving_root + 'drug2index.pickle', 'rb') as handle:
            drug2index = pickle.load(handle)
        mut_tensor = torch.load(saving_root + 'CL_feature/mut_tensor.pt')
        chr_tensor = torch.load(saving_root + 'CL_feature/chr_tensor.pt')
        cna_tensor = torch.load(saving_root + 'CL_feature/cna_tensor.pt')
        GE_tensor = torch.load(saving_root + 'CL_feature/GE_tensor.pt')
        CL_type_tensor = torch.load(saving_root + 'CL_feature/CL_type_tensor.pt')
        label_tensor = torch.load(saving_root + 'label_tensor.pt')
        result_df = pd.read_csv(saving_root + 'DRP_data/DRP_df.csv', index_col=0)
    return mut_tensor, chr_tensor, cna_tensor, GE_tensor, CL_type_tensor, label_tensor, result_df, drug2index, CL2index
# %%
def process_drug_feat(drug2index):
    drug_atom_feat, drug_bond_feat =  load_drug_feat()
    drug_atom_feat = {drug2index[drug]: drug_atom_feat[drug] for drug in drug2index.keys()}
    drug_bond_feat = {drug2index[drug]: drug_bond_feat[drug] for drug in drug2index.keys()}
    return drug_atom_feat, drug_bond_feat
def process_DRP_data(GDSC_df, drug2index, CL2index):
    GDSC_df.loc[:,'drugindex_CLindex'] = GDSC_df['DRUGID_COSMICID'].apply(map_indices, args=(drug2index, CL2index))
    GDSC_df.set_index('drugindex_CLindex', inplace=True)
    return GDSC_df
## Random choose rows for train-val and test by 8-2
# def train_val_test(GDSC_df):
#     random.seed(1234)
#     train_val_idx = random.sample(list(GDSC_df.index), int(len(GDSC_df.index)*0.8))
#     test_idx = [idx for idx in GDSC_df.index if idx not in train_val_idx]
#     train_idx = random.sample(train_val_idx, int(len(train_val_idx)*0.8))
#     val_idx = [idx for idx in train_val_idx if idx not in train_idx]
#     return train_idx, val_idx, test_idx
class DRP_dataset(Dataset):
    def __init__(self, mut_tensor, chr_tensor, cna_tensor, GE_tensor, CL_type_tensor, drug_atom_feat, drug_bond_feat, 
                 result_tensor, drug_idx, CL_idx):
        # Cell line features
        self.mut_tensor = mut_tensor
        self.chr_tensor = chr_tensor
        self.cna_tensor = cna_tensor
        self.GE_tensor = GE_tensor
        self.CL_type_tensor = CL_type_tensor

        # Drug features
        self.drug_atom_feat = drug_atom_feat
        self.drug_bond_feat = drug_bond_feat

        # Labels
        self.result_tensor = result_tensor

        # Indices
        self.drug_idx = drug_idx
        self.CL_idx = CL_idx

    def __len__(self):
        return len(self.CL_idx)

    def __getitem__(self, idx):
        # Fetch cell line features
        CL_ids = self.CL_idx[idx]
        drug_ids = self.drug_idx[idx]
        mut = self.mut_tensor[CL_ids]
        chr = self.chr_tensor[CL_ids]
        cna = self.cna_tensor[CL_ids]
        GE = self.GE_tensor[CL_ids]
        CL_type_tensor = self.CL_type_tensor[CL_ids]
        # Fetch drug features
        drug_atom = self.drug_atom_feat[drug_ids]
        drug_bond = self.drug_bond_feat[drug_ids]

        # Fetch labels
        label = self.result_tensor[idx]

        return CL_ids, mut, chr, cna, GE, drug_ids, drug_atom, drug_bond, label, CL_type_tensor
def collate_fn(batch_tuple):
    CL_id, mut, chr, cna, GE, drug, drug_atom_reprs, drug_bond_reprs, label, CL_type_tensor = map(list, zip(*batch_tuple))
    drug_atom_reprs = Batch.from_data_list(drug_atom_reprs)
    drug_bond_reprs = Batch.from_data_list(drug_bond_reprs)
    return {'cell_id': torch.tensor(CL_id),
            'drug_id': torch.tensor(drug),
            'mut': torch.stack(mut, dim = 0),
            'chr': torch.stack(chr, dim = 0),
            'cna': torch.stack(cna, dim = 0),
            'GE': torch.stack(GE),
            'drug_atom_repr': drug_atom_reprs, 'drug_bond_repr': drug_bond_reprs, 
            'label': torch.stack(label),
            'CL_type': torch.stack(CL_type_tensor, dim = 0)}
    
if __name__ == '__main__':
    print('Begin loading the cell data')
    mut_tensor, chr_tensor, cna_tensor, GE_tensor, CL_type_tensor, label_tensor, result_df, drug2index, CL2index = preprocess_cell_data()
    print('Begin loading the drug data')
    drug_atom_feat, drug_bond_feat = process_drug_feat(drug2index)
    print('Begin train-val-test split')
    train_idx, val_idx, test_idx = train_val_test_split(result_df)
    print(train_idx[0])
    drug_idx, CL_idx = process_CL_drug(train_idx)
    DRP_trainset = DRP_dataset(mut_tensor, chr_tensor, cna_tensor, GE_tensor, CL_type_tensor, drug_atom_feat, drug_bond_feat, result_tensor=label_tensor, drug_idx = drug_idx, CL_idx = CL_idx)
    train_loader = DataLoader(DRP_trainset, batch_size= 1024, shuffle=True, collate_fn=collate_fn)
    batch_sample = next(iter(train_loader))
    print(batch_sample['label'].shape)


