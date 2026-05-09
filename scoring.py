
import math
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from collections import Counter
from tqdm import tqdm

seqs = pd.read_csv('generated_raw.csv')['sequence'].tolist()
print(f'[OK] Loaded {len(seqs)} sequences from generated_raw.csv')



def build_kmer_freq(train_sequences, k=3, sample_size=10000):
    sample = random.sample(train_sequences, min(sample_size, len(train_sequences)))
    counts = Counter()
    for s in sample:
        for i in range(len(s) - k + 1):
            counts[s[i:i+k]] += 1
    total = sum(counts.values()) + 1e-8
    return {km: c / total for km, c in counts.items()}

def kmer_score(seq, train_freq, k=3):
    seq_kmers = Counter(seq[i:i+k] for i in range(len(seq) - k + 1))
    seq_total = sum(seq_kmers.values()) + 1e-8
    score = 0.0
    for km, cnt in seq_kmers.items():
        p_seq   = cnt / seq_total
        p_train = train_freq.get(km, 1e-6)
        score  += p_seq * math.log(p_seq / p_train + 1e-8)
    return max(0.0, 1.0 - abs(score) / 5.0)

def polar_frac_score(seq):
    polar = set('DERKHNSQT')
    frac  = sum(1 for aa in seq if aa in polar) / len(seq)
    if 0.35 <= frac <= 0.65:
        return 1.0
    return max(0.0, 1.0 - abs(frac - 0.5) * 4)

def aa_entropy_score(seq):
    counts = Counter(seq)
    total  = len(seq)
    probs  = [c / total for c in counts.values()]
    entropy     = -sum(p * math.log(p + 1e-8) for p in probs)
    max_entropy = math.log(20)
    return min(1.0, entropy / max_entropy)

def bio_score(seq, train_freq):
    return (0.34 * kmer_score(seq, train_freq)
          + 0.33 * polar_frac_score(seq)
          + 0.33 * aa_entropy_score(seq))

train_seqs_ref = train['sequence'].dropna().tolist()
print('[1] Building k-mer frequency table...')
train_freq = build_kmer_freq(train_seqs_ref, k=3)

print('[2] Computing bio scores...')
bio_scores = [bio_score(s, train_freq) for s in tqdm(seqs)]



PLL_STRIDE  = 5      
PLL_BATCH   = 8      

@torch.no_grad()
def esm2_pll(sequences, model, alphabet, stride=PLL_STRIDE, batch_size=PLL_BATCH):

    batch_converter = alphabet.get_batch_converter()
    model.eval()
    plls = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        pll_batch  = []

        for seq in batch_seqs:
            seq_t = seq[:L_MAX]
            data  = [('p', seq_t)]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(DEVICE)               # (1, L+2)
            L_seq  = tokens.shape[1] - 2             # strip BOS/EOS

            log_pll = 0.0
            n_masked = 0

            for pos in range(0, L_seq, stride):
                masked = tokens.clone()
                masked[0, pos + 1] = alphabet.mask_idx   # +1 for BOS offset

                out    = model(masked, repr_layers=[], return_contacts=False)
                logits = out['logits'][0, pos + 1, :]    # (vocab,)
                log_p  = F.log_softmax(logits, dim=-1)
                true_tok = tokens[0, pos + 1].item()
                log_pll += log_p[true_tok].item()
                n_masked += 1

            pll_batch.append(log_pll / max(1, n_masked))

        plls.extend(pll_batch)

    return plls

print(f'[3] Computing pLDDT proxy (ESM-2 PLL, stride={PLL_STRIDE})...')

raw_plls = esm2_pll(seqs, encoder.model, encoder.alphabet)

# Normalise PLL to [0, 1]  (min-max across this batch)
pll_arr    = np.array(raw_plls)
pll_min    = pll_arr.min()
pll_max    = pll_arr.max()
plddt_proxy = (pll_arr - pll_min) / (pll_max - pll_min + 1e-8)
print(f'[OK] PLL range: {pll_min:.3f} → {pll_max:.3f} | '
      f'proxy range: {plddt_proxy.min():.3f} → {plddt_proxy.max():.3f}')

# ── COMBINED SCORE ────────────────────────────────────────────
bio_arr      = np.array(bio_scores)
combined     = 0.3 * bio_arr + 0.7 * plddt_proxy
combined     = 1*bio_arr #to do,for onlyb biological score filtering,pLDDT functions also can be ommited on that case

scored_df = pd.DataFrame({
    'sequence'    : seqs,
    'bio_score'   : bio_arr,
    'plddt_proxy' : plddt_proxy,
    'combined_score': combined,
})
scored_df.to_csv('scored_sequences.csv', index=False)

print(f'\n[OK] Saved → scored_sequences.csv  ({len(scored_df)} rows)')
print(f'     combined_score | mean: {combined.mean():.3f} | '
      f'min: {combined.min():.3f} | max: {combined.max():.3f}')
print(f'     bio_score      | mean: {bio_arr.mean():.3f}')
print(f'     plddt_proxy    | mean: {plddt_proxy.mean():.3f}')
