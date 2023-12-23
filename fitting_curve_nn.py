import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
df = pd.read_csv('./GDSC2_curve_data/Dec16_4_2_fitted.csv')
df_raw = pd.read_csv('./GDSC2_curve_data/GDSC2_7_conc_Dec16.csv')
df['count_same_CL_drug_conc'] = df.groupby('CL_drug_conc')['CL_drug_conc'].transform('count')
df['CL'] = df.apply(lambda row: row.CL_drug_conc.split('_')[0], axis=1)
df['drug'] = df.apply(lambda row: row.CL_drug_conc.split('_')[1], axis=1)
df['conc'] = df.apply(lambda row: row.CL_drug_conc.split('_')[2], axis=1)
df['IC_50_NN_group'] = np.nan
df['rmse_nn_group'] = np.nan
cell_meta = pd.read_csv('./GDSC2_dataset/meta_data/cell_meta.csv')
drug_meta = pd.read_csv('./GDSC2_dataset/meta_data/drug_meta.csv')
df_raw['CL_drug_conc']=  df_raw['CL'].astype(str) + '_' + df_raw['drug']
def GDSC_x_50_from_lnIC50(lnIC50, maxC):
    # Calculate x from lnIC50 using the provided equation
    x = (lnIC50 - np.log(maxC)) / np.log(np.sqrt(10)) + 7
    return x/7
df_raw['IC50_x'] = df_raw.apply(lambda row: GDSC_x_50_from_lnIC50(row['LN_IC50'], row['maxc']), axis=1)
df_raw = df_raw[df_raw.dilution_pattern == 'half-log']
x_cols = [f'x_{i}' for i in range(7)]
y_cols = [f'y_{i}' for i in range(7)]
for i, col in enumerate(x_cols):
    df_raw[col] = (i + 1) / 7
net = Net()
net_cpu = Net()
net.cuda()
x_cols = [f'x_{i}' for i in range(7)]
y_cols = [f'y_{i}' for i in range(7)]
for i, col in enumerate(x_cols):
    df_raw[col] = (i + 1) / 7
x_values = df_raw[x_cols].values[0]
x_train = torch.tensor(df_raw[x_cols].values[0], dtype=torch.float32).unsqueeze(1)
y_df = torch.tensor(df[y_cols].values, dtype=torch.float32)
x_dense = np.linspace(-0.1, 1.1, 100000)
x_dense_tensor = torch.tensor(x_dense, dtype=torch.float32).unsqueeze(1)
x_train_gpu = x_train.cuda()
y_df_gpu = y_df.cuda()
x_dense_tensor_gpu = x_dense_tensor.cuda()
df_torch = df[['CL_drug_conc', 'CL', 'drug', 'conc'] + y_cols]
# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

def train_model(pair):
    subdf = df[df['CL_drug_conc'] == pair]
    idx = subdf.index
    n_samples = subdf.__len__()
    x_train_ = x_train_gpu.repeat(n_samples,1,1)
    y_train = y_df_gpu[idx].unsqueeze(2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Training loop
    for epoch in range(1501):
        optimizer.zero_grad()
        output = net(x_train_)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    torch.save(net.state_dict(), f'./fitting_curve/weight_group/NN_curve_{pair}.pth')
    rmse_nn = loss.item()
    y_dense_tensor = net(x_dense_tensor_gpu)
    responses = y_dense_tensor.squeeze()

    # Find where responses are less than 0.5
    less_than_05 = responses < 0.5

    # Find the first index where the response crosses 0.5
    # We do this by looking for the first False (response >= 0.5) after a True (response < 0.5)
    crossing_indices = (less_than_05[:-1] & ~less_than_05[1:]).nonzero()

    if crossing_indices.nelement() != 0:
        # Get the first crossing index
        first_crossing_idx = crossing_indices[0, 0].item() + 1  # +1 because we want the first response >= 0.5

        # Find the corresponding x value
        IC50_NN = x_dense_tensor_gpu[first_crossing_idx].item()
    else:
        difference = torch.abs(y_dense_tensor - 0.5)
        min_diff_idx = torch.argmin(difference)
        IC50_NN = x_dense_tensor[min_diff_idx].item()
    df.loc[idx, 'IC_50_NN_group'] = IC50_NN
    df.loc[idx, 'rmse_nn_group'] = rmse_nn

print(f'We have totally {df.CL_drug_conc.unique().__len__()} pairs')
print('start training')    
for i in range(df.CL_drug_conc.unique().__len__()):
    pair = df.CL_drug_conc.unique()[i]
    train_model(pair)
    if i % 500 == 0:
        print(f'{i}th pair finished')
df.to_csv('./GDSC2_curve_data/Dec16_nn_fitted.csv', index=False)
    
