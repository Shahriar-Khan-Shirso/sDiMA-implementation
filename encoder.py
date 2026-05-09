import torch
import torch.nn as nn
import esm

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
L_MAX      = 320
LATENT_DIM = 480
VOCAB_SIZE = 20
AA_STR     = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX   = {aa: i for i, aa in enumerate(AA_STR)}


class ProteinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.batch_converter = self.alphabet.get_batch_converter()
        self.embed_dim = LATENT_DIM
        self.pad_idx   = self.alphabet.padding_idx

    @torch.no_grad()
    def encode(self, sequences):
        seqs_trunc = [s[:L_MAX] for s in sequences]
        data = [(f'p{i}', s) for i, s in enumerate(seqs_trunc)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(DEVICE)

        out = self.model(tokens, repr_layers=[12], return_contacts=False)
        reps_full = out['representations'][12][:, 1:-1, :]

        B = reps_full.shape[0]
        out_reps = torch.zeros(B, L_MAX, LATENT_DIM, device=DEVICE)
        out_mask = torch.zeros(B, L_MAX, dtype=torch.bool, device=DEVICE)

        for i, s in enumerate(seqs_trunc):
            L_i = min(len(s), L_MAX)
            out_reps[i, :L_i, :] = reps_full[i, :L_i, :]
            out_mask[i, :L_i]    = True

        return out_reps, out_mask
