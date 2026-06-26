import pickle, os
import numpy as np, torch
from model import GPTConfig, GPT          # model.py is in this folder -- the SAME transformer, unchanged

# hyperparameters
batch_size = 64
max_iters = 800
eval_interval = 200
eval_iters = 100
learning_rate = 0.1

HERE = os.path.dirname(__file__)
data_dir = os.path.join(HERE, 'data/mnist')
meta_path = os.path.join(data_dir, 'meta.pkl')
meta = pickle.load(open(meta_path, 'rb'))

seq_len = meta['seq_len']
vocab_size = meta['vocab_size']

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# meta.pkl gives vocab_size and seq_len
block_size = seq_len - 1                  # predict pixel t+1 from pixels 0..t
model = GPT(GPTConfig(block_size=block_size, vocab_size=vocab_size,
                      n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=False))
# ... optimizer + training loop: same shape as your char-GPT ...

def get_batch(split):  
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
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

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=np.uint16)
print(model.generate(context, max_new_tokens=block_size)[0].tolist())
