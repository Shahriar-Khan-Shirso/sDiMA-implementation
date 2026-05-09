
import numpy as np
from tqdm import tqdm

def cluster_diversity_fixed(seqs, sim_threshold, cov_threshold=0.80):
    n        = len(seqs)
    assigned = np.zeros(n, dtype=bool)
    n_cl     = 0
    for i in tqdm(range(n), desc=f'CD@{sim_threshold}', leave=False):
        if assigned[i]:
            continue
        n_cl        += 1
        assigned[i]  = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            s1, s2  = seqs[i], seqs[j]
            max_len = max(len(s1), len(s2))
            min_len = min(len(s1), len(s2))
            if min_len < cov_threshold * max_len:
                continue
            matches = sum(a == b for a, b in zip(s1[:min_len], s2[:min_len]))
            if matches / min_len >= sim_threshold:
                assigned[j] = True
    return n_cl / n, n_cl




import pandas as pd
best_df       = pd.read_csv('best_m1.csv')
best_seqs     = best_df['sequence'].tolist()

print()
print('=' * 55)
print(f'CLUSTERING DIVERSITY — best_m1 (pLDDT>30)  (n={len(best_seqs)})')
print('=' * 55)
if len(best_seqs) > 1:
    bcd_05,  bn_cl_05  = cluster_diversity_fixed(best_seqs, sim_threshold=0.50)
    bcd_095, bn_cl_095 = cluster_diversity_fixed(best_seqs, sim_threshold=0.95)
    print(f'  CD@0.5  = {bcd_05:.4f}   ({bn_cl_05} clusters / {len(best_seqs)} sequences)')
    print(f'  CD@0.95 = {bcd_095:.4f}   ({bn_cl_095} clusters / {len(best_seqs)} sequences)')
    best_df['cd_05']  = bcd_05
    best_df['cd_095'] = bcd_095
else:
    print('  Too few sequences for clustering (n<2). Skipping.')
    bcd_05, bcd_095 = None, None

best_df.to_csv('best_m1.csv', index=False)
print('[OK] best_m1.csv updated with CD metrics.')
