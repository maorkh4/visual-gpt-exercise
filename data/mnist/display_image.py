"""
Prepare MNIST for an image-GPT.

The idea: an image is just a sequence of tokens. We (optionally) shrink each
digit, flatten it row-major, binarize every pixel to {0, 1}, and write one long stream
of tokens -- exactly like nanoGPT's char datasets, except the "characters" are
black/white pixels and the vocabulary has size 2.

Why downscale? Self-attention is O(seq_len^2) and autoregressive sampling is ~O(seq_len^3),
so image resolution is the main lever on CPU cost. 28x28 = 784 tokens is slow to
sample on CPU; 14x14 = 196 tokens (the default, --downscale=2) is much cheaper, and MNIST
digits stay clearly legible. Use --downscale=1 for full resolution.

Outputs (next to this file): train.bin, val.bin (uint16 token streams, nanoGPT memmap
format) and meta.pkl (vocab_size + image geometry, for model init and decoding).
"""
import os, gzip, pickle, argparse
import numpy as np
import requests

HERE = os.path.dirname(__file__)

meta_path = os.path.join(HERE, 'meta.pkl')
meta = pickle.load(open(meta_path, 'rb'))

seq_len = meta['seq_len']
H = meta['img_h']
W = meta['img_w']

# round-trip proof: decode the first training image straight from train.bin
art = np.fromfile(os.path.join(HERE, "train.bin"), dtype=np.uint16, count=seq_len, offset=seq_len * 6).reshape(H, W)
print("\nfirst training digit, decoded from train.bin:")
for i, row in enumerate(art):
    if i % H == 0 and i > 0:
        print(" " * (W * 2))
    print("".join("##" if v else ".." for v in row))
