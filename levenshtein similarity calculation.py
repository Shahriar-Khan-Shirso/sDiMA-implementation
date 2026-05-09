
import numpy as np
import pandas as pd
import Levenshtein
from tqdm import tqdm

print('=' * 55)

print('=' * 55)


best_df = pd.read_csv('best_m1.csv')
gen_seqs_list = best_df['sequence'].tolist()

test_df = pd.read_csv('original_16.csv')
test_seqs_list = test_df['sequence'].dropna().tolist()

print(f'[OK] Loaded best_m1.csv: {len(gen_seqs_list)} sequences')
print(f'[OK] Loaded test set: {len(test_seqs_list)} sequences')


lev_pcts = []

print('\n[1] Calculating best Levenshtein match for each sequence...')
for gen_seq in tqdm(gen_seqs_list, desc='Processing best_m1'):
    best_match_score = 0.0
    
    for test_seq in test_seqs_list:
        max_len = max(len(gen_seq), len(test_seq))
        if max_len == 0:
            current_pct = 0.0
        else:
            lev_d = Levenshtein.distance(gen_seq, test_seq)
            current_pct = (max_len - lev_d) * 100.0 / max_len
        
        # Track the highest match found in the test set
        if current_pct > best_match_score:
            best_match_score = current_pct
            
    lev_pcts.append(best_match_score)


best_df['lev_match_pct'] = lev_pcts
best_df.to_csv('best_m1.csv', index=False)
print(f'\n[OK] best_m1.csv updated.')

try:
    final_df = pd.read_csv('final_generated_m1.csv')
    # Create lookup from best_df
    align_lookup = best_df.set_index('sequence')['lev_match_pct'].to_dict()
    
    # Map values
    final_df['lev_match_pct'] = final_df['sequence'].map(lambda s: align_lookup.get(s, np.nan))
    
    final_df.to_csv('final_generated_m1.csv', index=False)
    print('[OK] Propagated metrics to final_generated_m1.csv')
except FileNotFoundError:
    print('[INFO] final_generated_m1.csv not found, skipping.')


print('\n' + '=' * 30)
print('       RESULTS SUMMARY')
print('=' * 30)
print(f"Mean Levenshtein Match: {best_df['lev_match_pct'].mean():.2f}%")
print(f"Max Levenshtein Match:  {best_df['lev_match_pct'].max():.2f}%")
print(f"Min Levenshtein Match:  {best_df['lev_match_pct'].min():.2f}%")
print('=' * 30)
