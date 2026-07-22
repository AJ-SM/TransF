import torch
import torch.nn as nn
from torch.nn import functional as F

# from main import n_embbed
from scripts import tocknizer





class MultiHEAD(nn.Module):
    def __init__(self,num_heads,head_size,n_embbed,block_size):
        super().__init__()
        self.heads= nn.ModuleList(HEAD(head_size,n_embbed,block_size) for _ in range(num_heads))
        self.proj = nn.Linear(head_size * num_heads, n_embbed)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):


    def __init__(self,n_embbed):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embbed, 4 * n_embbed), nn.ReLU(), nn.Linear(4 * n_embbed, n_embbed),nn.Dropout(0.1))

    def forward(self,x):

        return self.net(x)






class HEAD(nn.Module):
    def __init__(self,headSize,n_embbed,block_size):
        super().__init__()
        self.key = nn.Linear(n_embbed,headSize,bias=False)
        self.value = nn.Linear(n_embbed,headSize,bias=False)
        self.query = nn.Linear(n_embbed,headSize,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(0.1)
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x)
        q= self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out




class Block(nn.Module):


    def __init__(self, n_embd, n_head, block_size):

        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHEAD(n_head, head_size, n_embd, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class BiLangModel(nn.Module):
    def __init__(self,t,n_embbed,block_size=32,n_head=4):
        super().__init__()
        # print("Test Forwarrd 2 ", t )
        self.block_size = block_size
        self.token_embd_table = nn.Embedding(t,n_embbed)
        # print("Test Size ",self.token_embd_table.__sizeof__())
        self.postion_embd_table = nn.Embedding(block_size,n_embbed)
        self.blocks = nn.Sequential(*[Block(n_embbed, n_head, block_size) for _ in range(6)])
        self.ln_f = nn.LayerNorm(n_embbed)
        self.llm_head = nn.Linear(n_embbed,t)

    def forward(self, idx, targets=None):
        _,T = idx.shape
        # print("Test size" , T)
        token_embeddings = self.token_embd_table(idx)
        x = token_embeddings + self.postion_embd_table(torch.arange(T))
        x = self.blocks(x)
        x = self.ln_f(x)
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
