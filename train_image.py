import pickle, os
import time
import numpy as np, torch
from model import GPTConfig, GPT          # model.py is in this folder -- the SAME transformer, unchanged
from sample_image import generate_samples

# hyperparameters
batch_size = 64
max_iters = 1500
eval_interval = 150
eval_iters = 60
learning_rate = 3e-4

torch.manual_seed(1337)

HERE = os.path.dirname(__file__)
data_dir = os.path.join(HERE, 'data/mnist')
meta_path = os.path.join(data_dir, 'meta.pkl')
meta = pickle.load(open(meta_path, 'rb'))

seq_len = meta['seq_len']
vocab_size = meta['vocab_size']

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# fold each flat stream into (num_images, seq_len) so a row is one whole image
train_images = train_data.reshape(-1, seq_len)
val_images = val_data.reshape(-1, seq_len)

# meta.pkl gives vocab_size and seq_len
block_size = seq_len - 1                  # predict pixel t+1 from pixels 0..t
model_args = dict(block_size=block_size, vocab_size=vocab_size,
                  n_layer=4, n_head=4, n_embd=128, dropout=0.0, bias=False)
model = GPT(GPTConfig(**model_args))
# ... optimizer + training loop: same shape as your char-GPT ...

def get_batch(split):
    images = train_images if split == 'train' else val_images
    ix = torch.randint(len(images), (batch_size,))
    batch = torch.from_numpy(images[ix].astype(np.int64))   # (batch_size, seq_len)
    x = batch[:, :-1].contiguous()
    y = batch[:, 1:].contiguous()
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_start = time.perf_counter()
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

train_secs = time.perf_counter() - train_start
print(f"training: {int(train_secs) // 60:02d}:{int(train_secs) % 60:02d} total over {max_iters} iters")

# save a checkpoint so sample_image.py can draw without retraining
model_checkpoint = {'model': model.state_dict(), 'model_args': model_args, 'meta': meta}
torch.save(model_checkpoint, os.path.join(HERE, 'ckpt.pt'))

# generate from the model
model.eval()
generate_samples(model=model, img_h=meta['img_h'], img_w=meta['img_w'])
