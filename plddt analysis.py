
import pandas as pd
import numpy as np

print('=' * 55)
print('RELOADING RESULTS AND ANALYSING pLDDT')
print('=' * 55)


    df = pd.read_csv('final_generated_m1.csv')
    print("[OK] Successfully loaded 'final_generated_m1.csv'")



df['seq_length'] = df['sequence'].astype(str).str.len()

plddt_arr = df['plddt_esmfold'].values
total_seqs = len(df)


plddt_mean = np.mean(plddt_arr)
plddt_min = np.min(plddt_arr)
plddt_max = np.max(plddt_arr)

print(f"Total Sequences Analyzed: {total_seqs}")
print(f"pLDDT Mean: {plddt_mean:.2f}")
print(f"pLDDT Min:  {plddt_min:.2f}")
print(f"pLDDT Max:  {plddt_max:.2f}")
print("-" * 55)


print("Average Sequence Length by pLDDT Bin (5-point intervals):")


bins = range(0, 105, 5)


df['plddt_bin'] = pd.cut(df['plddt_esmfold'], bins=bins, right=False)


length_by_bin = df.groupby('plddt_bin', observed=False)['seq_length'].mean()


for bin_interval, avg_len in length_by_bin.items():
    if pd.notna(avg_len): 
        count_in_bin = df['plddt_bin'].value_counts()[bin_interval]
        print(f"  pLDDT [{bin_interval.left:02d} - {bin_interval.right:02d}) : {avg_len:5.1f} residues avg length | (Count: {count_in_bin})")

print('=' * 55)
