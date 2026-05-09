import torch
import torch.nn as nn

LATENT_DIM = 480
AA_STR     = 'ACDEFGHIKLMNPQRSTVWY'


class SequenceDecoder(nn.Module):
    def __init__(self, esm_model, alphabet):
        super().__init__()
        self.lm_head  = esm_model.lm_head
        self.alphabet = alphabet

        vocab_size_esm = len(alphabet)
        tok_to_our = torch.full((vocab_size_esm,), -100, dtype=torch.long)
        for i, aa in enumerate(AA_STR):
            esm_idx = alphabet.get_idx(aa)
            tok_to_our[esm_idx] = i
        self.register_buffer('tok_to_our', tok_to_our)

    def forward(self, z):
        return self.lm_head(z)

    def logits_to_our20(self, logits):
        B, L, _ = logits.shape
        out = torch.full((B, L, 20), float('-inf'), device=logits.device)
        for our_i, aa in enumerate(AA_STR):
            esm_i = self.alphabet.get_idx(aa)
            out[:, :, our_i] = logits[:, :, esm_i]
        return out
