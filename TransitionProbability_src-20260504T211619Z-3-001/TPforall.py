# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import xml.etree.ElementTree as ET
from collections import defaultdict

# %%
# Find the unique EN values in the XML file(in the first file, 2015OR_ElementData.xml))
file = '2015OR_ElementData.xml'

tree = ET.parse(f'./data/OR-nbe/{file}')
root = tree.getroot()

unique_EN = sorted(set(int(e.find('EN').text) for e in root.findall('.//FHWAED')))
print(f"Unique EN values in {file}:")
print(unique_EN)

#%%
# Dictionary: {STRUCNUM: set of unique EN}
bridge_elements = defaultdict(set)

# Loop through all records
for items in root.findall('.//FHWAED'):
    id_struc = items.find('STRUCNUM').text.strip()
    id_ele = items.find('EN').text.strip()
    
    bridge_elements[id_struc].add(id_ele)


# Find bridge with max unique EN
max_bridge = None
max_count = 0

for bridge_id, ele_id in bridge_elements.items():
    if len(ele_id) > max_count:
        max_count = len(ele_id)
        max_bridge = bridge_id

print("Bridge with max unique elements:")
print("STRUCNUM:", max_bridge)
print("Number of unique EN:", max_count)
print("Unique EN list:", sorted(bridge_elements[max_bridge]))


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

# Check for NaN values in the transition matrix for each EN and skip if found
nan_elements = []

# calculate transition matrix for all unique EN values
transition_results = {}

for en in unique_EN:
    df_en = df[df['EN'] == en].copy()

    if df_en.empty:
        print(f"EN = {en} skipped -> no data")
        continue

    df_en['ele_hi'] = (
        df_en['CS1']/df_en['TOTALQTY'] +
        2/3 * df_en['CS2']/df_en['TOTALQTY'] +
        1/3 * df_en['CS3']/df_en['TOTALQTY']
    )

    # drop structures with increasing ele_hi (potential repairs)
    df_en = df_en.sort_values(by=['STRUCNUM', 'year'])

    bridges_drop = []
    for name, group in df_en.groupby('STRUCNUM'):
        # check if ele_hi is monotonically decreasing
        if not group['ele_hi'].is_monotonic_decreasing:
            bridges_drop.append(name)

    df_en = df_en[~df_en['STRUCNUM'].isin(bridges_drop)]

    if len(df_en) < 2:
        print(f"EN = {en} skipped -> insufficient data (<2 rows)")
        continue



    Tmtx = transition_matrix_fc(df_en)

    # check for NaN values
    if np.isnan(Tmtx).any():
        print(f"EN = {en} skipped -> NaN in transition matrix (no transitions observed for at least one condition state)")
        nan_elements.append(en)
        continue

    transition_results[en] = Tmtx

    print(f"\nEN = {en}")
    print(Tmtx)



    # plot transition prediction for this EN
    # list of bridges with full data from 2015 to 2025 for this EN
    full_record_list = []
    for name, group in df_en.groupby('STRUCNUM'):
        if group['year'].to_list() == list(range(2015, 2026)):
            full_record_list.append(name)

    df_plot = df_en[df_en['STRUCNUM'].isin(full_record_list)]

    if df_plot.empty:
        print(f"EN = {en} skipped plot -> no full 2015-2025 records")
        continue

    # collect actual CS distribution
    cs_hist = []
    record_yrs = np.arange(2015, 2026)

    for yr in record_yrs:
        cs_yr = df_plot[df_plot['year'] == yr][['CS1', 'CS2', 'CS3', 'CS4']].values.sum(axis=0)
        cs_hist.append(cs_yr)

    cs_hist = np.array(cs_hist)
    cs_dist = cs_hist / cs_hist.sum(axis=1)[:, None]

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
        for i in range(4):
            line = ax.plot(record_yrs, np.cumsum(cs_dist, axis=1)[:, i], 'o')
            color = line[0].get_color()

            ax.plot(predicted_yrs, np.cumsum(cs_pred, axis=1)[:, i], '-', color=color)
        ax.set_title(f"Transition prediction for EN = {en}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative CS distribution")
        plt.show()


output_file = "transition_matrices.py"


with open(output_file, "w") as f:
    f.write("import numpy as np\n\n")

    # write skipped ENs as comments
    f.write("# Auto-generated file. Do not edit.\n")
    f.write("#\n")
    f.write("# Elements skipped due to NaN transition probabilities:\n")

    if len(nan_elements) == 0:
        f.write("# None\n")
    else:
        for en in nan_elements:
            f.write(f"# EN {en}: skipped due to insufficient transition data (NaN values)\n")


    f.write("TRANSITION_MATRICES = {\n")
    for en, mat in transition_results.items():
        mat_rounded = np.round(mat, 8)
        f.write(f"    {en}: np.array({mat_rounded.tolist()}, dtype=float),\n")

    f.write("}\n")

print(f"Saved {len(transition_results)} matrices to {output_file}")