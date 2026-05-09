
from rapidfuzz import process, distance
import numpy as np
import pandas as pd
from tqdm import tqdm


best_df = pd.read_csv('best_m1.csv')
gen_seqs = best_df['sequence'].tolist()
print(f'[OK] Loaded best_m1.csv: {len(gen_seqs)} sequences (pLDDT > 30)')


try:
    train_seqs_full = train['sequence'].dropna().tolist()
    test_seqs_full  = test['sequence'].dropna().tolist()
    print(f'Training sequences (2010-2015): {len(train_seqs_full):,}')
    print(f'Test sequences (2016)         : {len(test_seqs_full):,}')
except NameError:
    print('[WARN] train/test not in memory. Reloading from CSV...')
    # Reload original_16.csv (the test set) and the training set from merged data
    # Note: You may need to adjust path or regenerate train list.
    test_df = pd.read_csv('original_16.csv')
    test_seqs_full = test_df['sequence'].dropna().tolist()
    

    merged = pd.read_csv('merged_protein_data.csv')
    merged['publicationYear'] = pd.to_numeric(merged['publicationYear'], errors='coerce')
    merged = merged.drop_duplicates(subset=['sequence'])
    train_raw = merged[merged['publicationYear'].between(2010, 2015)].copy()
    

    def filter_proteins(df):
        mask = df['macromoleculeType_x'].str.upper().str.contains('PROTEIN', na=False)
        return df[mask].copy()
    train = filter_proteins(train_raw)
    
    VALID_AA = set('ACDEFGHIKLMNPQRSTVWYX')
    def clean_sequence(seq):
        if not isinstance(seq, str): return None
        seq = seq.upper().strip()
        if len(seq) < 40 or len(seq) > 512: return None
        if set(seq) - VALID_AA: return None
        return seq
    train['sequence'] = train['sequence'].apply(clean_sequence)
    train = train.dropna(subset=['sequence'])
    train_seqs_full = train['sequence'].tolist()
    
    print(f'Training reloaded: {len(train_seqs_full):,}')
    print(f'Test reloaded:     {len(test_seqs_full):,}')

print('\n' + '=' * 55)
print('NOVELTY METRICS (L1 = vs training | L2 = vs test 2016)')
print('=' * 55)
print(f'  Generated (best_m1, pLDDT>30) : {len(gen_seqs)}')
print(f'  Training (2010-2015)          : {len(train_seqs_full):,}')
print(f'  Test (2016)                   : {len(test_seqs_full):,}')


print('\n[1] Novelty L1 (vs full training set)...')
l1_similarities, l1_best = [], []
for g_seq in tqdm(gen_seqs, desc='Novelty L1'):
    res = process.extractOne(
        g_seq, train_seqs_full,
        scorer=distance.Levenshtein.normalized_similarity)
    l1_similarities.append(res[1])
    l1_best.append(res[0])


print('\n[2] Novelty L2 (vs test 2016)...')
l2_similarities, l2_best = [], []
for g_seq in tqdm(gen_seqs, desc='Novelty L2'):
    res = process.extractOne(
        g_seq, test_seqs_full,
        scorer=distance.Levenshtein.normalized_similarity)
    l2_similarities.append(res[1])
    l2_best.append(res[0])


best_df['novelty_l1_sim']   = np.array(l1_similarities)
best_df['novelty_l2_sim']   = np.array(l2_similarities)
best_df['novelty_l1']       = 1.0 - best_df['novelty_l1_sim']
best_df['novelty_l2']       = 1.0 - best_df['novelty_l2_sim']
best_df['best_match_train'] = l1_best
best_df['best_match_test']  = l2_best
best_df.to_csv('best_m1.csv', index=False)





avg_l1_sim = np.mean(l1_similarities)
avg_l2_sim = np.mean(l2_similarities)
novelty_l1 = 1.0 - avg_l1_sim
novelty_l2 = 1.0 - avg_l2_sim

print('\n' + '=' * 55)
print(f'NOVELTY SUMMARY  (best_m1.csv, n={len(gen_seqs)} sequences, pLDDT > 30)')
print('=' * 55)
print(f'  Novelty L1 (vs Training): {novelty_l1:.4f}  ({novelty_l1*100:.1f}% novel)')
print(f'  Novelty L2 (vs Test)    : {novelty_l2:.4f}  ({novelty_l2*100:.1f}% novel)')
print('=' * 55)
print('[OK] Novelty scores added to best_m1.csv')
