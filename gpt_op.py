# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3900
eval_interval = 300
learning_rate = 1e-2
device = 'cpu'
eval_iters = 200
n_embd = 32

with open('./octavio.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# encoding individual characters to a numerical (int) value
def encode(s): return [stoi[c] for c in s]
# going backwards in the convertion
def decode(l): return ''.join([itos[s] for s in l])


data = torch.tensor(encode(text), dtype=torch.long)
# dividing the data into 90% for training and 10% for validating
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    # random set of 4 indexes -> starting point for 4 sets or blocks of characters to train/predict
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # starting from each index, 'x' becomes index and the following 8 characters
    x = torch.stack([data[i:i+block_size] for i in ix])
    # 'y' is the offset of 1, for later use comparing to the prediction
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token looks for the next token directly from a lookup (vocab_size x vocab_size) table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # creates the logits(raw, unnormalized predictions) and the loss (prediction - actual value(target))
        # idx and targets are both (B(atch), T(block_size)) tensor of integers
        # (B, T, C(channel = vocabl_size))
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # based on the logits previously created generates the most appropiate next character for the current (indexed) context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last T step
            logits = logits[:, -1, :]  # Becomes (B, C)
            # normalizes with softmax to get probabilites adding up to 1
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the runnig sequence  = adds the latest prediction to the predicted 'sentence'
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
# create pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # get new sample to train
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
