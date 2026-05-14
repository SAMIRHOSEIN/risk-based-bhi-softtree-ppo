# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 
# %%
# ============================================================
# Create history CSV file and summary files if not already created
# ============================================================

os.makedirs('./results', exist_ok=True)

# load nbe data from 2015 to 2025
df = pd.read_xml(f'./data/OR-nbe/2015OR_ElementData.xml')
df['year'] = 2015

for yr in range(2016, 2026):
    dfi = pd.read_xml(f'./data/OR-nbe/{yr}OR_ElementData.xml')
    dfi['year'] = yr
    df = pd.concat([df, dfi], axis=0)
    # df = pd.read_xml(f'./data/OR-nbe/{yr}OR_ElementData.xml')
    # print(f"{yr}: {df.columns}")

df.to_csv('./results/OR_nbe12_history.csv', index=False)

#%%
# Convert EN to int and STRUCNUM to string for easier processing
df['EN'] = df['EN'].astype(int)
df['STRUCNUM'] = df['STRUCNUM'].astype(str)

#%%
# ============================================================
# Bridge with maximum number of common unique elements
# ============================================================

def common_elements_across_existing_years(group):
    yearly_element_sets = []

    for yr, yr_group in group.groupby('year'):
        elements_this_year = set(yr_group['EN'].dropna().astype(int).unique())
        yearly_element_sets.append(elements_this_year)

    if len(yearly_element_sets) == 0:
        return tuple()

    common_elements = set.intersection(*yearly_element_sets)

    return tuple(sorted(common_elements))



bridge_element_counts = (
    df.groupby('STRUCNUM')
    .apply(common_elements_across_existing_years)
    .reset_index(name='Common_EN_List')
)

bridge_element_counts['Number_of_common_elements'] = (
    bridge_element_counts['Common_EN_List'].apply(len)
)

# Find bridge with maximum number of common elements
max_bridge_row = bridge_element_counts.loc[
    bridge_element_counts['Number_of_common_elements'].idxmax()
]

print("\n============================================================")
print("BRIDGE WITH MAXIMUM NUMBER OF COMMON ELEMENTS ACROSS EXISTING YEARS")
print("============================================================")

print(f"Bridge ID (STRUCNUM): {max_bridge_row['STRUCNUM']}")
print(f"Number of common elements: {max_bridge_row['Number_of_common_elements']}")
print(f"Common EN list: {max_bridge_row['Common_EN_List']}")


bridge_element_counts.to_csv(
    './results/common_elements_by_bridge_all_years.csv',
    index=False
)

print("\nSaved summary CSV files:")
print("./results/common_elements_by_bridge.csv")
#%%
# ============================================================
# Summary files for all years: 2015–2025
# ============================================================
# 1. Unique EN values across all files, not just 2015
unique_EN = sorted(df['EN'].dropna().unique())

print("\nUnique EN values across all files 2015–2025:")
print(unique_EN)
print(f"Number of unique EN values: {len(unique_EN)}")


# 2. Number of bridges in each year + total unique bridges
bridges_by_year = (
    df.groupby('year')['STRUCNUM']
    .nunique()
    .reset_index(name='Number_of_bridges')
)

total_unique_bridges = df['STRUCNUM'].nunique()

bridges_by_year.loc[len(bridges_by_year)] = ['All years unique', total_unique_bridges]

bridges_by_year.to_csv('./results/bridges_count_by_year.csv', index=False)


# 3. For each element, count how many unique bridges contain that element
element_bridge_counts = (
    df.groupby('EN')['STRUCNUM']
    .nunique()
    .reset_index(name='Number_of_bridges_with_this_element')
    .sort_values(
        by='Number_of_bridges_with_this_element',
        ascending=False
    )
)

element_bridge_counts.to_csv('./results/element_bridge_counts_all_years.csv', index=False)

# 4. Repeated combinations of elements across bridges
# IMPORTANT: For each bridge, use only the elements that appear in ALL years where that bridge exists. This is the intersection of EN values across years for each STRUCNUM.
# Bridge A:
# 2015 → [12, 107, 310, 331]
# 2016 → [12, 107, 310]
# so the result will be [12, 107, 310]

def common_elements_across_existing_years(group):
    yearly_element_sets = []

    for yr, yr_group in group.groupby('year'):
        elements_this_year = set(yr_group['EN'].dropna().astype(int).unique())
        yearly_element_sets.append(elements_this_year)

    if len(yearly_element_sets) == 0:
        return tuple()

    common_elements = set.intersection(*yearly_element_sets)

    return tuple(sorted(common_elements))


bridge_element_combinations = (
    df.groupby('STRUCNUM')
    .apply(common_elements_across_existing_years)
    .reset_index(name='Element_Combination')
)

combination_summary = (
    bridge_element_combinations
    .groupby('Element_Combination')
    .agg(
        Number_of_bridges=('STRUCNUM', 'count'),
        Bridge_numbers=('STRUCNUM', lambda x: ', '.join(map(str, x)))
    )
    .reset_index()
)

combination_summary['Number_of_elements'] = (
    combination_summary['Element_Combination'].apply(len)
)

combination_summary['Element_Combination'] = (
    combination_summary['Element_Combination']
    .apply(lambda x: ', '.join(map(str, x)))
)

combination_summary = combination_summary[
    ['Number_of_elements', 'Element_Combination', 'Number_of_bridges', 'Bridge_numbers']
].sort_values(by='Number_of_bridges', ascending=False)

combination_summary.to_csv('./results/repeated_element_combinations_all_years.csv', index=False)



#%%
# ============================================================
# Summary files for year 2025 only
# ============================================================

df_2025 = df[df['year'] == 2025].copy()

# 1. Common elements by bridge for 2025 only
# Since this is only one year, common elements = unique elements in 2025
common_elements_by_bridge_2025 = (
    df_2025.groupby('STRUCNUM')['EN']
    .apply(lambda x: tuple(sorted(x.dropna().astype(int).unique())))
    .reset_index(name='Common_EN_List')
)

common_elements_by_bridge_2025['Number_of_common_elements'] = (
    common_elements_by_bridge_2025['Common_EN_List'].apply(len)
)

common_elements_by_bridge_2025.to_csv(
    './results/common_elements_by_bridge_2025.csv',
    index=False
)


# 2. For each element, count how many unique bridges contain that element in 2025
element_bridge_counts_2025 = (
    df_2025.groupby('EN')['STRUCNUM']
    .nunique()
    .reset_index(name='Number_of_bridges_with_this_element')
    .sort_values(
        by='Number_of_bridges_with_this_element',
        ascending=False
    )
)

element_bridge_counts_2025.to_csv(
    './results/element_bridge_counts_2025.csv',
    index=False
)


# 3. Repeated element combinations across bridges for 2025 only
bridge_element_combinations_2025 = (
    df_2025.groupby('STRUCNUM')['EN']
    .apply(lambda x: tuple(sorted(x.dropna().astype(int).unique())))
    .reset_index(name='Element_Combination')
)

combination_summary_2025 = (
    bridge_element_combinations_2025
    .groupby('Element_Combination')
    .agg(
        Number_of_bridges=('STRUCNUM', 'count'),
        Bridge_numbers=('STRUCNUM', lambda x: ', '.join(map(str, x)))
    )
    .reset_index()
)

combination_summary_2025['Number_of_elements'] = (
    combination_summary_2025['Element_Combination'].apply(len)
)

combination_summary_2025['Element_Combination'] = (
    combination_summary_2025['Element_Combination']
    .apply(lambda x: ', '.join(map(str, x)))
)

combination_summary_2025 = combination_summary_2025[
    ['Number_of_elements', 'Element_Combination', 'Number_of_bridges', 'Bridge_numbers']
].sort_values(by='Number_of_bridges', ascending=False)

combination_summary_2025.to_csv(
    './results/repeated_element_combinations_2025.csv',
    index=False
)

#%%
print("\nSaved all-year and 2025-only summary CSV files:")
print("./results/common_elements_by_bridge_all_years.csv")
print("./results/element_bridge_counts_all_years.csv")
print("./results/repeated_element_combinations_all_years.csv")
print("./results/common_elements_by_bridge_2025.csv")
print("./results/element_bridge_counts_2025.csv")
print("./results/repeated_element_combinations_2025.csv")
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

df = pd.read_csv('./results/OR_nbe12_history.csv', header=0)


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



output_file = "./results/transition_matrices.py"

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
    for el_num, matrics in transition_results.items():
        mat_rounded = np.round(matrics, 8)
        f.write(f"    {el_num}: np.array({mat_rounded.tolist()}, dtype=float),\n")

    f.write("}\n")

print(f"Saved {len(transition_results)} matrices to {output_file}")
#%%
import re
from pathlib import Path
# ============================================================
# Export element quantities for one bridge (only year 2025)
# ============================================================
bridge_id = "01577A016 04612"

# filter bridge
df_bridge = df[df['STRUCNUM'].astype(str).str.strip() == bridge_id].copy()

# keep only 2025
df_bridge = df_bridge[df_bridge['year'] == 2025]

# sort rows
df_bridge = df_bridge.sort_values(by=['year', 'EN'])

# select important columns
df_bridge_export = df_bridge[
    ['year', 'STRUCNUM', 'EN', 'TOTALQTY', 'CS1', 'CS2', 'CS3', 'CS4']
]

# create safe file name
safe_bridge_id = re.sub(r'[^A-Za-z0-9_-]+', '_', bridge_id.strip())

# output path
output_xlsx = Path.cwd() / "onebridgeVerification" / f"bridge_{safe_bridge_id}_2025_elements.xlsx"

df_bridge_export.to_excel(output_xlsx, index=False)

print("\nExcel file saved successfully:")
print(output_xlsx)
# %%
