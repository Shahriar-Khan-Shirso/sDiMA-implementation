import esm

print("Loading ESMFold...")
esmfold = esm.pretrained.esmfold_v1()
esmfold = esmfold.eval().to(DEVICE)  
print("[OK] ESMFold loaded and moved to GPU")
import os
import numpy as np
import pandas as pd
import torch

torch.cuda.empty_cache()
from esm.pretrained import esmfold_v1

PDB_SAVE_DIR = 'plddt_pdbs'
os.makedirs(PDB_SAVE_DIR, exist_ok=True)


fold_df    = pd.read_csv('fold_sequence.csv')   
gen_seqs   = fold_df['sequence'].tolist()
index_list = list(range(len(gen_seqs)))

print(f'[1] Folding {len(gen_seqs)} sequences with ESMFold...')
print(f'    PDB files → {PDB_SAVE_DIR}/')
print(f'    Estimated time: ~{len(gen_seqs)*20//60} min on GPU.')

plddt_dict   = calculate_plddt(
    predictions=gen_seqs,
    index_list=index_list,
    device=str(DEVICE),
    pdb_path=PDB_SAVE_DIR
)

plddt_scores = [plddt_dict.get(seq, 0.0) for seq in gen_seqs]
plddt_arr    = np.array(plddt_scores)


fold_df['plddt_esmfold'] = plddt_arr
fold_df.to_csv('final_generated_m1.csv', index=False)

print(f'\n  pLDDT mean  = {plddt_arr.mean():.2f}')
print(f'  pLDDT range = {plddt_arr.min():.2f} – {plddt_arr.max():.2f}')

PLDDT_THRESHOLD = 30.0
best_mask = plddt_arr > PLDDT_THRESHOLD
best_df   = fold_df[best_mask].copy()

print(f'\n  Sequences with pLDDT > {PLDDT_THRESHOLD}: {len(best_df)} / {len(fold_df)}')

best_df.to_csv('best_m1.csv', index=False)
print(f'[OK] Saved best_m1.csv: {len(best_df)} sequences')


torch.cuda.empty_cache()
print('[OK] VRAM cleared.')
