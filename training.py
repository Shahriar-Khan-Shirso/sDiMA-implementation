
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR  
import math
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 8    

train_seqs = train['sequence'].dropna().tolist()
print(f'Training on {len(train_seqs):,} sequences')
print(f'Pre-computing per-position latents (B, {L_MAX}, {LATENT_DIM})...')
print(f'This will use ~{len(train_seqs) * L_MAX * LATENT_DIM * 4 / 1e9:.1f} GB of RAM.')
print('(Stored on CPU — moved to GPU batch-by-batch during training)')

encoder.eval()
all_latents_list = []
all_masks_list   = []

ENCODE_BATCH = 16
with torch.no_grad():
    for i in tqdm(range(0, len(train_seqs), ENCODE_BATCH), desc='Encoding'):
        batch_seqs = train_seqs[i:i+ENCODE_BATCH]
        reps, mask = encoder.encode(batch_seqs)  
        all_latents_list.append(reps.cpu())
        all_masks_list.append(mask.cpu())

all_latents = torch.cat(all_latents_list, dim=0)  
all_masks   = torch.cat(all_masks_list,   dim=0)   
print(f'[OK] Latents shape : {all_latents.shape}')
print(f'[OK] Masks   shape : {all_masks.shape}')



real_flat   = all_latents[all_masks]          
latent_mean = real_flat.mean(0).to(DEVICE)    
latent_std  = real_flat.std(0).clamp(min=0.01).to(DEVICE)  


all_latents_norm = all_latents.clone()
all_latents_norm[all_masks] = (
    (all_latents[all_masks] - latent_mean.cpu())
    / latent_std.cpu()
)
print(f'[OK] Latents normalised | std range: '
      f'{latent_std.min().item():.4f} – {latent_std.max().item():.4f}')


AA_STR   = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = {aa: i for i, aa in enumerate(AA_STR)}
PAD_IDX  = -100  
aa_token_targets = torch.full((len(train_seqs), L_MAX), PAD_IDX, dtype=torch.long)
for i, seq in enumerate(train_seqs):
    for j, aa in enumerate(seq[:L_MAX]):
        esm_idx = encoder.alphabet.get_idx(aa)  
        if esm_idx is not None:
            aa_token_targets[i, j] = esm_idx

print(f'[OK] AA token targets shape: {aa_token_targets.shape}')



class SeqLatentDataset(Dataset):
    def __init__(self, latents, masks, aa_targets):
        self.latents    = latents
        self.masks      = masks
        self.aa_targets = aa_targets

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        return self.latents[idx], self.masks[idx], self.aa_targets[idx]


dataset    = SeqLatentDataset(all_latents_norm, all_masks, aa_token_targets)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0, drop_last=True
)

MIN_SNR_GAMMA = 5.0

def build_snr_weights(diffusion, gamma=5.0, device='cpu'):

    T = diffusion.n_steps
    try:
        if hasattr(diffusion, 'alphas_cumprod'):
            acp = diffusion.alphas_cumprod.float()
        elif hasattr(diffusion, 'sqrt_alphas_cumprod'):
            acp = diffusion.sqrt_alphas_cumprod.float() ** 2
        elif hasattr(diffusion, 'alpha_bar'):
            acp = torch.tensor(diffusion.alpha_bar, dtype=torch.float32)
        else:
            print('[SNR] alphas_cumprod not found → uniform weights (no SNR weighting)')
            return torch.ones(T, device=device)

        acp     = acp.to(device)
        snr     = acp / (1.0 - acp).clamp(min=1e-8)         
        weights = torch.minimum(snr, torch.full_like(snr, gamma)) / snr
        weights = weights.clamp(min=0.1)                       
        return weights

    except Exception as e:
        print(f'[SNR] Weight build failed ({e}) → uniform weights')
        return torch.ones(T, device=device)

snr_weights = build_snr_weights(diffusion, gamma=MIN_SNR_GAMMA, device=DEVICE)
print(f'[OK] SNR weights | min: {snr_weights.min():.4f} | '
      f'max: {snr_weights.max():.4f} | mean: {snr_weights.mean():.4f}')



N_EPOCHS       = 60
LR             = 1e-4
WARMUP_STEPS   = 500
SELF_COND_PROB = 0.9


optimizer = AdamW(
    list(denoiser.parameters()) + list(decoder.parameters()),
    lr=LR, weight_decay=1e-5
)

steps_per_epoch = len(dataloader)

0
warmup_sched = LinearLR(
    optimizer,
    start_factor = 1e-6,
    end_factor   = 1.0,
    total_iters  = WARMUP_STEPS
)
cawr_sched = CosineAnnealingWarmRestarts(
    optimizer,
    T_0     = 15 * steps_per_epoch,   # restart period = 15 epochs
    T_mult  = 1,                       # equal-length restarts
    eta_min = LR * 0.01               # LR floor = 1e-6
)
scheduler = SequentialLR(
    optimizer,
    schedulers = [warmup_sched, cawr_sched],
    milestones = [WARMUP_STEPS]        # hand off after warmup completes
)

denoiser.train()
decoder.train()

print(f'\nStarting training: {N_EPOCHS} epochs | '
      f'batch={BATCH_SIZE} | lr={LR} | device={DEVICE}')
print(f' Scheduler : LinearWarmup({WARMUP_STEPS} steps) → CAWR(T_0=15 epochs, 4 restarts)')
print(f' SNR weight: min-SNR-γ={MIN_SNR_GAMMA}')
print(f' CE weight : progressive 0.10 → 0.30 over {N_EPOCHS} epochs')
print(f'        Diffusion : epsilon-prediction + Tan-10 schedule')
print(f'        Latent    : (B, {L_MAX}, {LATENT_DIM})  — per-position sequence latent')
print('=' * 60)

global_step = 0

for epoch in range(N_EPOCHS):
    epoch_diff_loss = 0.0
    epoch_ce_loss   = 0.0
    n_batches = 0


    ce_weight = 0.10 + 0.20 * (epoch / max(1, N_EPOCHS - 1))

    for x0_batch, mask_batch, aa_tgt_batch in tqdm(
            dataloader, desc=f'Epoch {epoch+1}/{N_EPOCHS}', leave=False):

        x0     = x0_batch.to(DEVICE)    
        mask   = mask_batch.to(DEVICE)   
        aa_tgt = aa_tgt_batch.to(DEVICE) 
        B      = x0.shape[0]

        t_idx  = torch.randint(0, diffusion.n_steps, (B,), device=DEVICE)
        t_cont = t_idx.float() / diffusion.n_steps

        x_t, noise = diffusion.q_sample(x0, t_idx)  


        x_self_cond = None
        if torch.rand(1).item() < SELF_COND_PROB:
            with torch.no_grad():
                pred_eps_sc = denoiser(x_t, t_cont, mask, None).detach()
                x_self_cond = diffusion.x0_from_eps(
                    x_t, pred_eps_sc, t_idx).clamp(-3, 3).detach()
                x_self_cond = x_self_cond * mask.unsqueeze(-1).float()


        pred_eps = denoiser(x_t, t_cont, mask, x_self_cond) 

        mask_f = mask.unsqueeze(-1).float()         
        n_real = mask_f.sum().clamp(min=1)


        w_t       = snr_weights[t_idx].view(B, 1, 1)                         
        diff_loss = ((pred_eps - noise) ** 2 * mask_f * w_t).sum() / n_real  

        x0_recovered = diffusion.x0_from_eps(
            x_t, pred_eps.detach(), t_idx).clamp(-3, 3)
        x0_recovered = x0_recovered * mask_f


        logits  = decoder(x0_recovered, mask=mask)
        ce_loss = F.cross_entropy(
                    logits.reshape(-1, 20),  
                    aa_tgt.reshape(-1),     
                    ignore_index=PAD_IDX
                    )

        loss = diff_loss + ce_weight * ce_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(denoiser.parameters()) + list(decoder.parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()
        ema.update()

        epoch_diff_loss += diff_loss.item()
        epoch_ce_loss   += ce_loss.item()
        n_batches  += 1
        global_step += 1

    avg_diff   = epoch_diff_loss / n_batches
    avg_ce     = epoch_ce_loss   / n_batches
    current_lr = scheduler.get_last_lr()[0]

    print(f'Epoch {epoch}/{N_EPOCHS} | '
          f'eps-Loss: {avg_diff:.4f} | '
          f'CE-Loss: {avg_ce:.4f} | '
          f'CE-w: {ce_weight:.2f} | '
          f'LR: {current_lr:.2e}')

torch.save({
    'denoiser'    : denoiser.state_dict(),
    'denoiser_ema': ema.get_model().state_dict(),
    'decoder'     : decoder.state_dict(),
    'latent_mean' : latent_mean,
    'latent_std'  : latent_std,
}, 'm1_fixed_checkpoint_10000.pt')
print('\n[OK] Training done. Checkpoint saved: m1_fixed_checkpoint_10000.pt')
