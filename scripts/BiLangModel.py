import torch
import torch.nn as nn
from torch.nn import functional as F
from scripts import tocknizer


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

    def generate(self,idx, max_tocken_len = 100):
        for i in range(max_tocken_len):
            logits, loss = self.forward(idx)
            logits = logits[:,-1]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def trainer(model,dataset, epoch, context_size=8):
    optimizer= torch.optim.AdamW(model.parameters(),lr=0.001)
    n = len(dataset) - context_size - 1
    for e in range(epoch):
        total_loss = 0.0
        steps = 0
        for start in range(0, n, context_size):
            x, y = tocknizer.genIO(dataset, start, context_size)
            x = x.view(1, -1)
            y = y.view(1, -1)
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        print(f"epoch {e}: avg loss {total_loss / steps:.4f}")
