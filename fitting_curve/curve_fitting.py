# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import torch
from tqdm import tqdm
import warnings
from scipy.optimize import OptimizeWarning



# %%
df = pd.read_csv('../GDSC2_curve_data/final_GDSC2_7_conc.csv')
df = df[df['dilution_pattern'].isin(['half-log'])].reset_index(drop=True)
filtered_drugs = df.DRUG_ID_lib.value_counts().index[df.DRUG_ID_lib.value_counts().values > 1]
df = df[df['DRUG_ID_lib'].isin(filtered_drugs)].reset_index(drop=True)
def GDSC1_conc_from_x(x, maxC):
    CONC = maxC * np.exp((x - 9) * np.log(2))
    return CONC

# Define the columns
x_cols = ['x_' + str(i) for i in range(7)]
conc_cols = ['conc_' + str(i) for i in range(7)]

# Apply the function to each column and assign to the corresponding new column
for x_col, conc_col in zip(x_cols, conc_cols):
    df[conc_col] = df.apply(lambda row: GDSC1_conc_from_x(row[x_col], row['maxc']), axis=1)
def GDSC_x_50_from_lnIC50(lnIC50, maxC):
    # Calculate x from lnIC50 using the provided equation
    x = (lnIC50 - np.log(maxC)) / np.log(np.sqrt(10)) + 7
    return x/7
# for conc_col, x_col in zip(conc_cols, x_cols):
#     df[x_col] = df.apply(lambda row: GDSC2_x_from_conc(row[conc_col], row['maxc']), axis=1)
for i, col in enumerate(x_cols):
    df[col] = (i + 1) / 7
df['IC50_x'] = df.apply(lambda row: GDSC_x_50_from_lnIC50(row['LN_IC50'], row['maxc']), axis=1)

# %%
y_cols = ['y_' + str(i) for i in range(7)]
x_cols = ['x_' + str(i) for i in range(7)]
# df[y_cols] = 1 - df[y_cols]

# %%
## Find the rows with any y values that are NaN
y_nan_rows = df[df[y_cols].isnull().any(axis=1)]
## drop these rows
df = df.drop(y_nan_rows.index).reset_index(drop=True)

# %%
## for all the rows, if any in y_cols gets value > 1, filter out
print(f'Max response of y is {df[y_cols].max().max()}') 
print(f'Min response of y is {df[y_cols].min().min()}')

# %%
CL_drug_pairs = df[['CL', 'drug']].drop_duplicates().reset_index(drop=True)

# %%
class sigmoid_fit:
    def __init__(self, df):
        self.df = df
        self.x_cols = [f'x_{i}' for i in range(7)]
        self.y_cols = [f'y_{i}' for i in range(7)]
    def select_rows(self, CL, drug):
        return self.df[(self.df['CL'] == CL) & (self.df['drug'] == drug)]
    def sigmoid_2para(self, x, p, s):
        return 1.0 / (1 + np.exp((x - p) / -s))

    def sigmoid_4para(self, c, L, k, c_0, d):
        return 1.0 / (L + np.exp(-k * (c - c_0))) + d

    def rmse(self, predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    def train_curve_2para(self, CL, drug):
        sub_df = self.select_rows(CL, drug)
        x_values = sub_df[self.x_cols].values[0]
        y_values = sub_df[self.y_cols].values
        num_observations = y_values.shape[0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            popt_2para, _ = curve_fit(self.sigmoid_2para, np.tile(x_values, num_observations), y_values.flatten())
        estimated_sigmoid_2_predictions = self.sigmoid_2para(np.tile(x_values, num_observations), *popt_2para)
        rmse_sigmoid_2_predictions = self.rmse(estimated_sigmoid_2_predictions, y_values.flatten())
        return {'rmse': rmse_sigmoid_2_predictions, 'model': popt_2para}

    def train_curve_4para(self, CL, drug):
        sub_df = self.select_rows(CL, drug)
        x_values = sub_df[self.x_cols].values[0]
        y_values = sub_df[self.y_cols].values
        num_observations = y_values.shape[0]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            popt_4para, _ = curve_fit(self.sigmoid_4para, np.tile(x_values, num_observations), y_values.flatten())
        estimated_sigmoid_4_predictions = self.sigmoid_4para(np.tile(x_values, num_observations), *popt_4para)
        rmse_sigmoid_4_predictions = self.rmse(estimated_sigmoid_4_predictions, y_values.flatten())
        return {'rmse': rmse_sigmoid_4_predictions, 'model': popt_4para}
    def train_curves(self, CL, drug):
        sub_df = self.select_rows(CL, drug)
        x_values = sub_df[x_cols].values[0]
        y_values = sub_df[y_cols].values
        num_observations = y_values.shape[0]
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", OptimizeWarning)
        #     warnings.simplefilter("ignore", category=RuntimeWarning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Turn off filter
            # Your curve fitting code here
            # Reset the warning registry for the whole Python process
            warnings.resetwarnings()
            popt_2para, _ = curve_fit(self.sigmoid_2para, np.tile(x_values, num_observations), y_values.flatten())
            popt_4para, _ = curve_fit(self.sigmoid_4para, np.tile(x_values, num_observations), y_values.flatten())
        estimated_sigmoid_2_predictions = self.sigmoid_2para(np.tile(x_values, num_observations), *popt_2para)
        estimated_sigmoid_4_predictions = self.sigmoid_4para(np.tile(x_values, num_observations), *popt_4para)
        rmse_sigmoid_2_predictions = self.rmse(estimated_sigmoid_2_predictions, y_values.flatten())
        rmse_sigmoid_4_predictions = self.rmse(estimated_sigmoid_4_predictions, y_values.flatten())
        sigmoid_4_dict = {'rmse': rmse_sigmoid_4_predictions, 'model': popt_4para}
        sigmoid_2_dict = {'rmse': rmse_sigmoid_2_predictions, 'model': popt_2para}
        return sigmoid_2_dict, sigmoid_4_dict
    def plot_curves(self, CL, drug, sigmoid_2_dict, sigmoid_4_dict):
        sub_df = self.select_rows(CL, drug)
        a, b = sigmoid_2_dict['model']
        x_high = max(4*b + a, 1.1)
        x_low = min(-4*b + a, 0)
        ## use np linspace to generate 1000 points between x_low and x_high      
        x_dense = np.linspace(x_low, x_high, 1000)
        rmse_sigmoid_2_predictions = sigmoid_2_dict['rmse']
        predicted_sig_2para = self.sigmoid_2para(x_dense, a, b)

        L_fit, k_fit, x0_fit, d_fit = sigmoid_4_dict['model']
        rmse_sigmoid_4_predictions = sigmoid_4_dict['rmse']
        predicted_sig_4para = self.sigmoid_4para(x_dense, L_fit, k_fit, x0_fit, d_fit)
        plt.plot(x_dense, predicted_sig_2para, label=f'Fitted Sigmoid 2-Parameter\nRMSE: {rmse_sigmoid_2_predictions:.3f}')
        plt.plot(x_dense, predicted_sig_4para, label=f'Fitted Sigmoid 4-Parameter\nRMSE: {rmse_sigmoid_4_predictions:.3f}')
        # plt.scatter(x_repeated.numpy(), y_obs.numpy(), color='black', marker='o', facecolors='none', edgecolors='black', alpha=0.6, label='Observed Points (y_obs)')
        for x_col, y_final_cols in zip(self.x_cols, self.y_cols):
            # If only one unique maxc value, use a single color (e.g., 'blue')
            scatter = plt.scatter(sub_df[x_col], sub_df[y_final_cols], color='black', marker='o', facecolors='none', edgecolors='black', alpha=0.6)        
        plt.xlabel('Normalized concentration')
        plt.ylabel('Normalized response')
        drug_name = drug.split('_')[0]
        maxc = drug.split('_')[1]
        plt.title(f'Dose-response curves for drug {drug_name} on cell line {CL} with max concentration {maxc} µM')
        plt.legend()
        plt.grid(True)
        plt.show()

# %%
model_sigmoid = sigmoid_fit(df)
# %%
sigmoid_2_cols = [f'sigmoid_2_{i}' for i in ['p','s','rmse']]
sigmoid_4_cols = [f'sigmoid_4_{i}' for i in ['L','k','x0','d','rmse']]

# %%
for CL, drug in tqdm(zip(CL_drug_pairs['CL'], CL_drug_pairs['drug'])):
    sigmoid_2_failed = False
    sigmoid_4_failed = False
    try:
        sigmoid_2_dict = model_sigmoid.train_curve_2para(CL, drug)
        p, s = sigmoid_2_dict['model']
        rmse_sigmoid_2_predictions = sigmoid_2_dict['rmse']
        CL_drug_pairs.loc[(CL_drug_pairs['CL'] == CL) & (CL_drug_pairs['drug'] == drug), sigmoid_2_cols] = [p, s, rmse_sigmoid_2_predictions]
    except RuntimeError as e:
        print(f"Failed to fit 2-parameter sigmoid for CL {CL}, drug {drug}: {e}")
        sigmoid_2_failed = True
    try:
        sigmoid_4_dict = model_sigmoid.train_curve_4para(CL, drug)
        L, k, c_0, d = sigmoid_4_dict['model']
        rmse_sigmoid_4_predictions = sigmoid_4_dict['rmse']
        CL_drug_pairs.loc[(CL_drug_pairs['CL'] == CL) & (CL_drug_pairs['drug'] == drug), sigmoid_4_cols] = [L, k, c_0, d, rmse_sigmoid_4_predictions]
    except RuntimeError as e:
        print(f"Failed to fit 4-parameter sigmoid for CL {CL}, drug {drug}: {e}")
        sigmoid_4_failed = True
    if sigmoid_2_failed and sigmoid_4_failed:
        print(f"Skipping CL {CL}, drug {drug} as both fittings failed")
        continue   
CL_drug_pairs.to_csv('../GDSC2_curve_data/sigmoid_fitting_results.csv', index=False)


