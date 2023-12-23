import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.optimize import OptimizeWarning, curve_fit
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim

def sigmoid_2para(x, p, s):
    return 1.0 / (1 + np.exp((x - p) / -s))

def sigmoid_4para(c, L, k, c_0, d):
    return 1.0 / (L + np.exp(-k * (c - c_0))) + d
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def calculate_ic50_4_sig(L, k, c_0, d):
    """
    Calculate the IC50 value for a 4-parameter sigmoid model.

    Parameters:
    L, k, c_0, d: Parameters of the sigmoid model

    Returns:
    IC50 value, or 1.5 if the result is NaN
    """
    try:
        ic50 = c_0 - (1 / k) * np.log((2 / (1 - 2 * d)) - L) 
        if np.isnan(ic50):
            return 1.5
        else:
            return ic50
    except:
        return 1.5
    
def calculate_ic50_2_sig(p, s):
    return p

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

if __name__ == "__main__":
    df_raw = pd.read_csv('./GDSC2_curve_data/GDSC2_7_conc_Dec16.csv')
    df_raw = df_raw[df_raw.dilution_pattern == 'half-log']
    df_raw['CL_drug_conc']=  df_raw['CL'].astype(str) + '_' + df_raw['drug']
    def GDSC_x_50_from_lnIC50(lnIC50, maxC):
        # Calculate x from lnIC50 using the provided equation
        x = (lnIC50 - np.log(maxC)) / np.log(np.sqrt(10)) + 7
        return x/7
    df_raw['IC50_x'] = df_raw.apply(lambda row: GDSC_x_50_from_lnIC50(row['LN_IC50'], row['maxc']), axis=1)
    df_raw = df_raw.dropna()
    x_cols = [f'x_{i}' for i in range(7)]
    y_cols = [f'y_{i}' for i in range(7)]
    for i, col in enumerate(x_cols):
        df_raw[col] = (i + 1) / 7
    result_df= df_raw[['CL_drug_conc'] + y_cols]
    result_df['min_y'] = result_df[y_cols].min(axis=1)
    result_df['max_y'] = result_df[y_cols].max(axis=1)
    sub_df = result_df.copy()
    # sub_df = result_df[(result_df.max_y > 0.5) & (result_df.min_y < 0.5)]
    sub_df.reset_index(drop=True, inplace=True)
    x_values = df_raw[x_cols].values[0]
    ## Start fitting for 2-parameter and 4-parameter sigmoid
    for col in ['2_para_0', '2_para_1', 'rmse_2_para', '4_para_0', '4_para_1', '4_para_2', '4_para_3', 'rmse_4_para']:
        sub_df[col] = np.nan
    print('Start fitting for 2-parameter and 4-parameter sigmoid')
    for index, row in sub_df.iterrows():
        sigmoid_2_failed = False
        sigmoid_4_failed = False
        y_values = row[y_cols].values
        CL,drug_name,maxc = row.CL_drug_conc.split('_')
        # Fit the 2-parameter sigmoid model
        try:
            popt_2para, _ = curve_fit(sigmoid_2para, x_values, y_values, maxfev=10000)
            estimated_sigmoid_2_predictions = sigmoid_2para(x_values, *popt_2para)
            rmse_sigmoid_2_predictions = rmse(estimated_sigmoid_2_predictions, y_values)
            sub_df.at[index, '2_para_0'] = popt_2para[0]
            sub_df.at[index, '2_para_1'] = popt_2para[1]
            sub_df.at[index, 'rmse_2_para'] = rmse_sigmoid_2_predictions
        except RuntimeError as e:
            print(f"Failed to fit 2-parameter sigmoid for row {index} CL {CL}, drug {drug_name}: {e}")
            sigmoid_2_failed = True
        try:
        # Fit the 4-parameter sigmoid model
            popt_4para, _ = curve_fit(sigmoid_4para, x_values, y_values, maxfev=10000)
            estimated_sigmoid_4_predictions = sigmoid_4para(x_values, *popt_4para)
            rmse_sigmoid_4_predictions = rmse(estimated_sigmoid_4_predictions, y_values)
            sub_df.at[index, '4_para_0'] = popt_4para[0]
            sub_df.at[index, '4_para_1'] = popt_4para[1]
            sub_df.at[index, '4_para_2'] = popt_4para[2]
            sub_df.at[index, '4_para_3'] = popt_4para[3]
            sub_df.at[index, 'rmse_4_para'] = rmse_sigmoid_4_predictions
        except RuntimeError as e:
            print(f"Failed to fit 4-parameter sigmoid for row {index} CL {CL}, drug {drug_name}: {e}")
            sigmoid_4_failed = True
        if sigmoid_2_failed and sigmoid_4_failed:
            print(f"Skipping for row {index} CL {CL}, drug {drug_name} as both fittings failed")
            continue   
    # sub_df.to_csv('./GDSC2_curve_data/Dec16_4_2_fitted.csv', index=False)
    fitted_sub_df = sub_df.dropna()
    fitted_sub_df.reset_index(drop=True, inplace=True)
    temp_df = df_raw[['CL_drug_conc', 'IC50_x', 'LN_IC50']]
    temp_df.drop_duplicates(inplace=True)
    temp_df.reset_index(drop=True, inplace=True)
    fitted_sub_df = pd.merge(fitted_sub_df, temp_df, on='CL_drug_conc', how='inner')
    fitted_sub_df['IC50_4_para'] = fitted_sub_df.apply(lambda row: calculate_ic50_4_sig(
        row['4_para_0'], row['4_para_1'], row['4_para_2'], row['4_para_3']), axis=1)

    fitted_sub_df['IC50_2_para'] = fitted_sub_df['2_para_0'] 
    # fitted_sub_df.to_csv('./GDSC2_curve_data/Dec16_4_2_fitted.csv', index=False)
    fitted_sub_df.to_csv('./GDSC2_curve_data/Dec20_4_2_fitted.csv', index=False)