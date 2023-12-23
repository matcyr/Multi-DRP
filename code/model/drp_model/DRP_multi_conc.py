import torch
import torch.nn as nn
import os
import sys
from model.drug_model.drug_encoder import *
from model.cell_model.cell_encoder import *



class DRP_multi_view(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.dim_drug = model_config.get('drug_embed_dim')
        self.layer_cell = model_config.get('cell_layer_num')
        self.layer_drug = model_config.get('drug_layer_num') + 1
        self.dim_cell = model_config.get('cell_embed_dim')
        self.dropout_ratio = model_config.get('dropout_rate')
        self.view_dim = model_config.get('view_dim')
        self.use_regulizer = model_config.get('use_regulizer')
        self.use_regulizer_pathway = model_config.get('use_drug_path_way')
        # drug graph branch
        self.GNN_drug = drug_hier_encoder(model_config)
        ## cell non-graph feature. 
        self.ge_encoder = GE_vae(model_config)
        self.mut_encoder = cell_cnn(model_config, 310)
        self.cna_encoder = cell_cnn(model_config, 425)
        self.chr_encoder = cell_cnn(model_config, 338)
        # if self.use_cnn == 'True':
        #     self.mut_encoder = cell_cnn(model_config, 2560)
        #     self.cnv_encoder = cell_cnn(model_config, 2816)
        
        self.dose_embedding = nn.Embedding(7, 128)
        self.ge_vae_emb = nn.Sequential(
            nn.Linear(self.dim_cell, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024, 2*self.dim_cell),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )
        self.mut_emb_layer = nn.Sequential(
            nn.Linear(self.dim_cell, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024, self.dim_cell),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )
        self.cna_emb_layer = nn.Sequential(
            nn.Linear(self.dim_cell, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024, self.dim_cell),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )
        self.chr_emb_layer = nn.Sequential(
            nn.Linear(self.dim_cell, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(1024, self.dim_cell),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_ratio),
        )
        if self.use_regulizer == 'True':
            self.cell_regulizer = nn.Linear(5*self.dim_cell,32)## 32 cancer types
        if self.use_regulizer_pathway == 'True':
            self.drug_path_way_class = nn.Sequential(nn.Linear(2*self.dim_drug , 1024), 
                                            nn.ReLU(),
                                            nn.Linear(1024,24))
        self.regression_ge = nn.Sequential(
            nn.Linear(2*self.dim_drug + 2*self.dim_cell, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        self.regression_cna = nn.Sequential(
            nn.Linear(2*self.dim_drug + self.dim_cell, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        self.regression_mut = nn.Sequential(
            nn.Linear(2*self.dim_drug + self.dim_cell, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        self.regression_chr = nn.Sequential(
            nn.Linear(2*self.dim_drug + self.dim_cell, 512),
            nn.ELU(),
            nn.Dropout(p=self.dropout_ratio),
            nn.Linear(512, self.view_dim),
            nn.ELU()
        )
        self.pred_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4*self.view_dim + 5*self.dim_cell + 2*self.dim_drug + 128, 512),
                nn.ELU(),
                nn.Dropout(p=self.dropout_ratio),
                nn.Linear(512, 1)
            ) for _ in range(7)
        ])
    def forward(self, batch_sample):
        drug_atom = batch_sample['drug_atom_repr']
        drug_bond = batch_sample['drug_bond_repr']
        ge = batch_sample['GE']
        mut = batch_sample['mut'].unsqueeze(1)
        cna = batch_sample['cna'].unsqueeze(1)
        chr = batch_sample['chr'].unsqueeze(1)
        drug_class = None
        cell_class = None
        drug_pathway = None
        # forward drug
        x_drug = self.GNN_drug(drug_atom, drug_bond) ###[fp,repr] 2*embed_size
        # x_drug = self.drug_emb(x_drug)
        # forward cell
        x_mut = self.mut_encoder (mut)
        x_mut = self.mut_emb_layer(x_mut)
        x_cna = self.cna_encoder(cna)
        x_cna = self.cna_emb_layer(x_cna)
        x_chr = self.chr_encoder(chr)
        x_chr = self.chr_emb_layer(x_chr)
        z, recon_x, mean, log_var = self.ge_encoder(ge)
        x_ge = self.ge_vae_emb(mean)
        cell_embed = torch.cat([x_ge, x_mut, x_cna, x_chr], dim = -1)
        # combine drug feature and cell line feature
        x_dg = torch.cat([x_drug, x_ge], -1)
        x_dm = torch.cat([x_drug, x_mut], -1)
        x_dc = torch.cat([x_drug, x_cna], -1)
        x_dchr = torch.cat([x_drug, x_chr], -1)
        x_dg, x_dm, x_dc, x_dchr = self.regression_ge(x_dg) ,self.regression_mut(x_dm), self.regression_cna(x_dc), self.regression_chr(x_dchr)
        x = torch.cat([x_dg, x_dm, x_dc, x_dchr, x_drug, cell_embed], -1)  ##Residual connection.        
        batch_size = x.size(0)
        ## Repeat the 0,1,2,3,4,5,6,7 for batch_size times.
        index_tensor = torch.tensor([i for i in range(7)]).repeat(batch_size, 1).to(x.device)
        x = [torch.cat((x, self.dose_embedding(index_tensor[:,i])), dim=-1) for i in range(7)]
        x = [self.pred_layers[i](x[i]) for i in range(7)]
        x = torch.stack(x, dim = 1).squeeze(-1)
        if self.use_regulizer == 'True':
            cell_class = self.cell_regulizer(cell_embed)
        if self.use_regulizer_pathway =='True':
            drug_pathway = self.drug_path_way_class(x_drug)
        return {'pred': x, 'cell_regulizer':cell_class,  'drug_pathway':drug_pathway}