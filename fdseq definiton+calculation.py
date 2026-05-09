
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from tqdm import tqdm

print('=' * 55)
print('FIXED: Rebuild best_m1.csv (pLDDT > 30) and compute FDseq (FULL reference)')
print('=' * 55)


try:
    final_df = pd.read_csv('final_generated_m1.csv')
    print(f'[OK] Loaded final_generated_m1.csv : {len(final_df)} sequences')
except FileNotFoundError:
    print('[ERROR] final_generated_m1.csv not found. Run Cell 9 first.')
    raise

# Check pLDDT column
if 'plddt_esmfold' not in final_df.columns:
    print('[ERROR] Column "plddt_esmfold" missing. Did Cell 9 run correctly?')
    raise


PLDDT_THRESH = 30.0
best_df = final_df[final_df['plddt_esmfold'] > PLDDT_THRESH].copy()
print(f'[OK] Sequences with pLDDT > {PLDDT_THRESH}: {len(best_df)}')

# Show breakdown (optional)
for t in [40, 50, 60, 70]:
    cnt = (best_df['plddt_esmfold'] > t).sum()
    if cnt > 0:
        print(f'     > {t}: {cnt}')

# Overwrite best_m1.csv
best_df.to_csv('best_m1.csv', index=False)
print(f'[OK] Saved best_m1.csv with {len(best_df)} sequences (pLDDT > {PLDDT_THRESH})')


gen_seqs = best_df['sequence'].tolist()
print(f'\nGenerated sequences: {len(gen_seqs)}')


try:
    # Try to use existing DataFrames from Cell 4
    train_seqs_all = train['sequence'].dropna().tolist()
    test_seqs_all  = test['sequence'].dropna().tolist()
    print(f'Training sequences (2010-2015) from memory: {len(train_seqs_all):,}')
    print(f'Test sequences (2016) from memory         : {len(test_seqs_all):,}')
except NameError:
    print('[WARN] train/test not in memory. Reloading from merged_protein_data.csv...')
    merged = pd.read_csv('merged_protein_data.csv')
    merged['publicationYear'] = pd.to_numeric(merged['publicationYear'], errors='coerce')
    merged = merged.drop_duplicates(subset=['sequence'])
    
    train_raw = merged[merged['publicationYear'].between(2010, 2015)].copy()
    test_raw  = merged[merged['publicationYear'] == 2016].copy()
    
    # Protein filter
    def filter_proteins(df):
        mask = df['macromoleculeType_x'].str.upper().str.contains('PROTEIN', na=False)
        return df[mask].copy()
    train = filter_proteins(train_raw)
    test  = filter_proteins(test_raw)
    
    # Sequence cleaning (same as Cell 4)
    VALID_AA = set('ACDEFGHIKLMNPQRSTVWYX')
    def clean_sequence(seq):
        if not isinstance(seq, str):
            return None
        seq = seq.upper().strip()
        if len(seq) < 40 or len(seq) > 512:
            return None
        if set(seq) - VALID_AA:
            return None
        return seq
    
    train['sequence'] = train['sequence'].apply(clean_sequence)
    test['sequence']  = test['sequence'].apply(clean_sequence)
    train = train.dropna(subset=['sequence'])
    test  = test.dropna(subset=['sequence'])
    
    train_seqs_all = train['sequence'].tolist()
    test_seqs_all  = test['sequence'].tolist()
    print(f'Training sequences (reloaded): {len(train_seqs_all):,}')
    print(f'Test sequences (reloaded)    : {len(test_seqs_all):,}')


print(f'\nFDseq will use ALL {len(train_seqs_all):,} training and {len(test_seqs_all):,} test sequences.')


def embed_fd(seqs, encoder, device, batch=32):
    encoder.eval()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            out = encoder.encode(seqs[i:i+batch])
            emb = out[0] if isinstance(out, tuple) else out
            # If 3D (batch, seq_len, dim) -> mean pool -> 2D
            if emb.ndim == 3:
                emb = emb.mean(dim=1)
            emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu())
    full = torch.cat(all_embs, dim=0).numpy()
    if full.ndim > 2:
        full = full.reshape(full.shape[0], -1)
    return full


print('\nEmbedding sequences (this may take a few minutes for 24k+5k+all seq>30)...')
gen_embs   = embed_fd(gen_seqs,       encoder, DEVICE)
train_embs = embed_fd(train_seqs_all, encoder, DEVICE)
test_embs  = embed_fd(test_seqs_all,  encoder, DEVICE)

def get_stats(x):
    return np.mean(x, axis=0), np.cov(x, rowvar=False)

mu_gen, cov_gen   = get_stats(gen_embs)
mu_train, cov_train = get_stats(train_embs)
mu_test, cov_test   = get_stats(test_embs)

def fd(mu1, cov1, mu2, cov2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff @ diff + np.trace(cov1 + cov2 - 2 * covmean)

fd_train = fd(mu_gen, cov_gen, mu_train, cov_train)
fd_test  = fd(mu_gen, cov_gen, mu_test, cov_test)

print('\n' + '=' * 55)
print('FDseq RESULTS (using FULL reference sets)')
print('=' * 55)
print(f'FDseq vs Training (2010-2015, n={len(train_seqs_all):,}) : {fd_train:.6f}')
print(f'FDseq vs Test (2016, n={len(test_seqs_all):,})          : {fd_test:.6f}')
print('=' * 55)


best_df['fd_train'] = fd_train
best_df['fd_test']  = fd_test
best_df.to_csv('best_m1.csv', index=False)


final_df.loc[final_df['sequence'].isin(best_df['sequence']), 'fd_train'] = fd_train
final_df.loc[final_df['sequence'].isin(best_df['sequence']), 'fd_test']  = fd_test
final_df.to_csv('final_generated_m1.csv', index=False)

print('\n[OK] FDseq scores added to best_m1.csv and final_generated_m1.csv')
print(f'[OK] best_m1.csv now contains {len(best_df)} sequences (all pLDDT > {PLDDT_THRESH})')
print('=' * 55)
