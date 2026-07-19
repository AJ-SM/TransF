import torch
import torch.nn as nn
from torch.nn import functional as F
from scripts import tocknizer

class HEAD(nn.Module):
    def __init__(self,headSize,n_embbed):
        super().__init__()
        self.key = nn.Linear(n_embbed,headSize,bias=False)
        self.value = nn.Linear(n_embbed,headSize,bias=False)
        self.query = nn.Linear(n_embbed,headSize,bias=False)






class BiLangModel(nn.Module):
    def __init__(self,t,n_embbed,block_size=32):
        super().__init__()
        # print("Test Forwarrd 2 ", t )
        self.block_size = block_size
        self.token_embd_table = nn.Embedding(t,n_embbed)
        # print("Test Size ",self.token_embd_table.__sizeof__())
        self.postion_embd_table = nn.Embedding(block_size,n_embbed)
        self.llm_head = nn.Linear(n_embbed,t)

    def forward(self, idx, targets=None):
        _,T = idx.shape
        # print("Test size" , T)
        token_embeddings = self.token_embd_table(idx)
        x = token_embeddings + self.postion_embd_table(torch.arange(T))
        logits = self.llm_head(x)


        if targets is None:
            loss = None
        else:

            B, T, C = logits.shape
            # print("B T C ",B,T,C)
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self,idx, max_tocken_len = 100):
        for i in range(max_tocken_len):

            idx_cond = idx[:, -self.block_size:]
            logits, loss = self.forward(idx_cond)
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

    # torch.save(model.state_dict(),".model/BiLangModel.pth")
