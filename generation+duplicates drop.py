
import torch
import numpy as np
import pandas as pd
import random

# --- Load EMA checkpoint ---
ckpt        = torch.load('m1_fixed_checkpoint_10000.pt', map_location=DEVICE)
denoiser.load_state_dict(ckpt['denoiser_ema'])
decoder.load_state_dict(ckpt['decoder'])
latent_mean = ckpt['latent_mean'].to(DEVICE)
latent_std  = ckpt['latent_std'].to(DEVICE)
print('[OK] EMA checkpoint loaded')


train_lengths = [len(s) for s in train['sequence'].dropna().tolist()]
_len_p10 = max(int(np.percentile(train_lengths, 10)), 40)
_len_p90 = min(int(np.percentile(train_lengths, 90)), L_MAX)

def sample_lengths(n):
    lens = np.array(random.choices(train_lengths, k=n))
    return np.clip(lens, _len_p10, _len_p90).astype(int)

print(f'[OK] Length range (p10–p90): {_len_p10}–{_len_p90} aa')


AA_STR = 'ACDEFGHIKLMNPQRSTVWY'

@torch.no_grad()
def decode_latents(latents_norm, temperature=1.0):
    decoder.eval()
    N       = latents_norm.shape[0]
    lengths = sample_lengths(N)

    lat_dev = (latents_norm.to(DEVICE)
               * latent_std.unsqueeze(0).unsqueeze(0)
               + latent_mean.unsqueeze(0).unsqueeze(0))

    sequences = []
    DECODE_BATCH = 32
    for i in range(0, N, DECODE_BATCH):
        z_b    = lat_dev[i:i+DECODE_BATCH]
        lens   = lengths[i:i+DECODE_BATCH]
        logits = decoder(z_b)                     

        for b in range(logits.shape[0]):
            L_i   = int(lens[b])
            lgt_i = logits[b, :L_i, :]

            if temperature != 1.0:
                probs = torch.softmax(lgt_i / temperature, dim=-1)
                idx   = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                idx = lgt_i.argmax(dim=-1)


            seq_chars = []
            for tok in idx:
                aa = encoder.alphabet.get_tok(tok.item())
                if aa in AA_STR:
                    seq_chars.append(aa)
            seq = ''.join(seq_chars)

            if len(seq) >= 40:
                sequences.append(seq)
    return sequences


N_GENERATE     = 10000
GEN_BATCH_SIZE = 50

print(f'\nGenerating {N_GENERATE} sequences '
      f'(EMA, 200 DDIM steps, batch={GEN_BATCH_SIZE})...')
denoiser.eval()

all_gen_latents = []
for start in range(0, N_GENERATE, GEN_BATCH_SIZE):
    b        = min(GEN_BATCH_SIZE, N_GENERATE - start)
    gen_mask = torch.ones(b, L_MAX, dtype=torch.bool, device=DEVICE)

    gen_lat = diffusion.p_sample_loop(
        denoiser,
        shape=(b, L_MAX, LATENT_DIM),
        mask=gen_mask,
        n_sample_steps=200,
        temperature=0.85
    )
    all_gen_latents.append(gen_lat.cpu())
    if (start // GEN_BATCH_SIZE) % 20 == 0:
        print(f'  {start + b} / {N_GENERATE}')

generated_latents = torch.cat(all_gen_latents, dim=0)
print(f'[OK] Latents shape: {generated_latents.shape}')

raw_seqs = decode_latents(generated_latents, temperature=1.0)
print(f'[OK] Decoded: {len(raw_seqs)} valid sequences')


before = len(raw_seqs)
raw_seqs = list(dict.fromkeys(raw_seqs)) 
after = len(raw_seqs)
print(f'[OK] Duplicates removed: {before - after} dropped | {after} remaining')


gen_df_71 = pd.DataFrame({'sequence': raw_seqs})
gen_df_71.to_csv('generated_raw.csv', index=False)
print(f'[OK] Saved → generated_raw.csv  ({len(gen_df_71)} rows)')
