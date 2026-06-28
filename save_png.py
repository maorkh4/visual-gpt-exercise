import os
import struct
import zlib
import torch
from model import GPTConfig, GPT
from sample_image import generate_samples

HERE = os.path.dirname(__file__)
SAMPLE_DIR = os.path.join(HERE, 'samples')


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


if __name__ == '__main__':
    model_checkpoint = torch.load(os.path.join(HERE, 'ckpt.pt'), map_location='cpu')
    model = GPT(GPTConfig(**model_checkpoint['model_args']))
    model.load_state_dict(model_checkpoint['model'])
    model.eval()

    meta = model_checkpoint['meta']
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    arts = generate_samples(model=model, img_h=meta['img_h'], img_w=meta['img_w'])
    for i, art in enumerate(arts):
        save_png(art, os.path.join(SAMPLE_DIR, f"sample_{i:02d}.png"))
