# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# %%
# load nbe data from 2015 to 2025

df = pd.read_xml(f'./data/OR-nbe/2015OR_ElementData.xml')
df['year'] = 2015

for yr in range(2016, 2026):
    dfi = pd.read_xml(f'./data/OR-nbe/{yr}OR_ElementData.xml')
    dfi['year'] = yr
    df = pd.concat([df, dfi], axis=0)
    # df = pd.read_xml(f'./data/OR-nbe/{yr}OR_ElementData.xml')
    # print(f"{yr}: {df.columns}")

df.to_csv('./data/OR_nbe12_history.csv', index=False)

# %%

def transition_matrix_fc(df):
    """Calculates transition matrix using simple frequency counts."""
    counts = np.zeros((4, 4))
    counts[-1, -1] = 1.0
    for name, group in df.groupby('STRUCNUM'):
        for i in range(len(group) - 1):
            q_t = group.iloc[i][['CS1', 'CS2', 'CS3', 'CS4']].values
            q_t1 = group.iloc[i+1][['CS1', 'CS2', 'CS3', 'CS4']].values

            # frequency counts
            moved_12 = q_t[0] - q_t1[0]
            if moved_12 >= 0:
                counts[0, 0] += (q_t[0] - moved_12)
                counts[0, 1] += moved_12
            else:
                moved_12 = 0
            
            moved_23 = q_t[1] + moved_12 - q_t1[1]
            if moved_23 >= 0:
                counts[1, 1] += max(0, q_t[1] - moved_23)
                counts[1, 2] += moved_23
            else:
                moved_23 = 0
            
            moved_34 = q_t[2] + moved_23 - q_t1[2]
            if moved_34 >= 0:
                counts[2, 2] += max(0, q_t[2] - moved_34)
                counts[2, 3] += moved_34
            else:
                moved_34 = 0
            
            # # frequency counts
            # moved_12 = max(0, q_t[0] - q_t1[0])
            # counts[0, 0] += (q_t[0] - moved_12)
            # counts[0, 1] += moved_12
            
            # moved_23 = max(0, q_t[1] + moved_12 - q_t1[1])
            # counts[1, 1] += max(0, q_t[1] - moved_23)
            # counts[1, 2] += moved_23
            
            # moved_34 = max(0, q_t[2] + moved_23 - q_t1[2])
            # counts[2, 2] += max(0, q_t[2] - moved_34)
            # counts[2, 3] += moved_34
    
    # Normalize rows
    return counts / counts.sum(axis=1)[:, None]

df = pd.read_csv('./data/OR_nbe12_history.csv', header=0)
df['ele_hi'] = df['CS1']/df['TOTALQTY'] + 2/3 * df['CS2']/df['TOTALQTY'] +\
    1/3 * df['CS3']/df['TOTALQTY']

# keep only 'EN' = 12
df = df[df['EN'] == 38]
df_full = df.copy()

# drop structures with increasing ele_hi (potential repairs)
df = df.sort_values(by=['STRUCNUM', 'year'])
all_structures = df['STRUCNUM'].unique()
bridges_drop = []
for name, group in df.groupby('STRUCNUM'):
    # check if ele_hi is monotonically decreasing
    if not group['ele_hi'].is_monotonic_decreasing:
        bridges_drop.append(name)
df = df[~df['STRUCNUM'].isin(bridges_drop)]

# get transition matrix
Tmtx = transition_matrix_fc(df)
print(Tmtx)

# %%
# plot transtion prediction

# list of bridges with full data from 2015 to 2025
full_record_list = []
for name, group in df.groupby('STRUCNUM'):
    if group['year'].to_list() == list(range(2015, 2026)):
        full_record_list.append(name)

df_plot = df[df['STRUCNUM'].isin(full_record_list)]
df_plot = df_plot[~df_plot['STRUCNUM'].isin(bridges_drop)]

# struc_num = '02237A01E 00262'
# df_plot = df[df['STRUCNUM'] == struc_num]
# struc_num = '01618A01E 01138'
# df_plot = df_full[df_full['STRUCNUM'] == struc_num]

# collect actual cs distribution

cs_hist = []
record_yrs = np.arange(2015, 2026)
for yr in record_yrs:
    cs_yr = df_plot[df_plot['year'] == yr][['CS1', 'CS2', 'CS3', 'CS4']].values.sum(axis=0)
    cs_hist.append(cs_yr)
cs_hist = np.array(cs_hist)
cs_dist = cs_hist/cs_hist.sum(axis=1)[:, None]

csi = cs_dist[0]
cs_pred = [csi]
predicted_yrs = np.arange(2015, 2035)
for yr in predicted_yrs[1:]:
    csi = Tmtx.T @ csi
    cs_pred.append(csi)
cs_pred = np.array(cs_pred)

with sns.plotting_context('notebook', font_scale=1.5):
    sns.set_style('whitegrid')
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(record_yrs, np.cumsum(cs_dist, axis=1), 'o')
    ax.plot(predicted_yrs, np.cumsum(cs_pred, axis=1), '-')
    
with sns.plotting_context('notebook', font_scale=1.5):
    sns.set_style('whitegrid')
    
    fig, ax = plt.subplots(
        2, 2, figsize=(12, 8),
        sharex=True, sharey='row',
        tight_layout=True
    )
    ax[0,0].plot(record_yrs, cs_dist[:, 0], 'o')
    ax[0,0].plot(predicted_yrs, cs_pred[:, 0], '-')

    ax[0,1].plot(record_yrs, cs_dist[:, 1], 'o')
    ax[0,1].plot(predicted_yrs, cs_pred[:, 1], '-')

    ax[1,0].plot(record_yrs, cs_dist[:, 2], 'o')
    ax[1,0].plot(predicted_yrs, cs_pred[:, 2], '-')

    ax[1,1].plot(record_yrs, cs_dist[:, 3], 'o')
    ax[1,1].plot(predicted_yrs, cs_pred[:, 3], '-')
# %%
