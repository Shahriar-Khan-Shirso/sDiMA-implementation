import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 480
AA_STR     = 'ACDEFGHIKLMNPQRSTVWY'


class TransformerDecoder(nn.Module):
    def __init__(self, latent_dim=480, hidden_dim=480, num_layers=3, vocab_size=20):
        super().__init__()
        self.position_embeddings = nn.Embedding(512, hidden_dim)
        self.register_buffer('position_ids', torch.arange(512).unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, z, mask=None):
        if z.dim() == 2:
            z = z.unsqueeze(1)
            squeeze_out = True
        else:
            squeeze_out = False

        seq_len = z.size(1)
        pos_emb = self.position_embeddings(self.position_ids[:, :seq_len])
        x = z + pos_emb

        if mask is not None:
            src_key_padding_mask = ~mask
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.transformer(x)

        out = self.fc(x)
        if squeeze_out:
            out = out.squeeze(1)
        return out


def finetune_decoder(decoder, all_latents_norm, all_masks, aa_token_targets,
                     ckpt, ckpt_path='m1_fixed_checkpoint_10000.pt'):

    real_mask_flat  = all_masks.reshape(-1)
    latents_flat    = all_latents_norm.reshape(-1, LATENT_DIM)
    aa_targets_flat = aa_token_targets.reshape(-1)

    ft_latents = latents_flat[real_mask_flat].clone()
    ft_aa_tgts = aa_targets_flat[real_mask_flat].clone()

    ft_ds     = TensorDataset(ft_latents, ft_aa_tgts)
    ft_loader = DataLoader(ft_ds, batch_size=512, shuffle=True, drop_last=False)

    FT_EPOCHS   = 10
    FT_LR       = 1e-4
    NOISE_START = 2.0
    NOISE_END   = 0.8

    optimizer_ft = AdamW(decoder.parameters(), lr=FT_LR, weight_decay=1e-4)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_ft,
        T_max=FT_EPOCHS * len(ft_loader),
        eta_min=1e-6
    )

    decoder.train()

    for epoch in range(FT_EPOCHS):
        noise_std = NOISE_START + (NOISE_END - NOISE_START) * epoch / max(1, FT_EPOCHS - 1)

        epoch_loss    = 0.0
        epoch_correct = 0
        epoch_total   = 0
        n_b = 0

        for z_b, aa_b in ft_loader:
            z_b  = z_b.to(DEVICE)
            aa_b = aa_b.to(DEVICE)

            noise   = torch.randn_like(z_b) * noise_std
            z_noisy = z_b + noise

            logits = decoder(z_noisy)
            loss   = F.cross_entropy(logits, aa_b)

            optimizer_ft.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer_ft.step()
            scheduler_ft.step()

            epoch_loss    += loss.item()
            epoch_correct += (logits.argmax(-1) == aa_b).sum().item()
            epoch_total   += aa_b.shape[0]
            n_b += 1

        avg_ce  = epoch_loss / n_b
        avg_acc = epoch_correct / epoch_total * 100

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:02d}/{FT_EPOCHS} | '
                  f'CE: {avg_ce:.4f} | '
                  f'Acc: {avg_acc:.1f}% | '
                  f'noise_std: {noise_std:.3f} | '
                  f'LR: {scheduler_ft.get_last_lr()[0]:.2e}')

    ckpt['decoder'] = decoder.state_dict()
    torch.save(ckpt, ckpt_path)

    decoder.eval()
    with torch.no_grad():
        z_s   = ft_latents[:64].to(DEVICE)
        noise = torch.randn_like(z_s) * 0.1
        preds = decoder(z_s + noise).argmax(-1).cpu()
        true  = ft_aa_tgts[:64]
        acc   = (preds == true).float().mean().item()
        pred_aas = ''.join(AA_STR[i] for i in preds[:20].tolist())
        true_aas = ''.join(AA_STR[i] for i in true[:20].tolist())

    print(f'Sanity check acc: {acc*100:.1f}%')
    print(f'Pred (first 20): {pred_aas}')
    print(f'True (first 20): {true_aas}')

    return decoder
