import torch
import torch.nn as nn
from torch.nn import functional as F


class BiLangModel(nn.Module):
    def __init__(self,t):
        super().__init__()
        self.token_embd_table = nn.Embedding(t,t )

    def forward(self, idx, targets=None):
        logits = self.token_embd_table(idx)
        if targets is None:
            loss = None
        else:

            B, T, C = logits.shape
            logits = logits.view(B * T, C)

            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


