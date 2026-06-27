import os
import struct
import time
import zlib
import torch
from model import GPTConfig, GPT

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), 'samples')


def generate(model, img_h, img_w, temperature=0.8):
    context = torch.zeros((1, 1), dtype=torch.long)
    max_new_tokens = (img_h * img_w) - 1
    art = model.generate(context, max_new_tokens=max_new_tokens,
                         temperature=temperature)[0].reshape(img_h, img_w)
    for row in art:
        print("".join("##" if v else ".." for v in row))
    return art


def save_png(art, path, scale=8):
    """Write an (img_h, img_w) tensor of 0/1 pixels as an 8-bit grayscale PNG."""
    pixels = [[255 if int(v) == 0 else 0 for v in row] for row in art]

    def _chunk(tag, data):
        return (struct.pack('>I', len(data)) + tag + data
                + struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff))

    height, width = len(pixels) * scale, len(pixels[0]) * scale
    raw = bytearray()
    for row in pixels:
        scaled = bytes(b for v in row for b in (v,) * scale)
        for _ in range(scale):
            raw += b'\x00' + scaled  # filter-type 0 (None) per scanline

    png = (b'\x89PNG\r\n\x1a\n'
           + _chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0))
           + _chunk(b'IDAT', zlib.compress(bytes(raw)))
           + _chunk(b'IEND', b''))
    with open(path, 'wb') as f:
        f.write(png)


def generate_samples(model, img_h, img_w, temperature=0.8, n_samples=10):
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    sample_secs = []
    for i in range(n_samples):
        print(" " * (img_w * 2))
        t0 = time.perf_counter()
        art = generate(model=model, img_h=img_h, img_w=img_w, temperature=temperature)
        sample_secs.append(time.perf_counter() - t0)
        save_png(art, os.path.join(SAMPLE_DIR, f"sample_{i:02d}.png"))

    print(f"sampling: {sum(sample_secs):.1f}s total over {n_samples} samples "
      f"({sum(sample_secs) / n_samples:.2f}s avg/sample)")


if __name__ == '__main__':
    HERE = os.path.dirname(__file__)
    model_checkpoint = torch.load(os.path.join(HERE, 'ckpt.pt'), map_location='cpu')
    model = GPT(GPTConfig(**model_checkpoint['model_args']))
    model.load_state_dict(model_checkpoint['model'])
    model.eval()

    meta = model_checkpoint['meta']
    generate_samples(model=model, img_h=meta['img_h'], img_w=meta['img_w'])
