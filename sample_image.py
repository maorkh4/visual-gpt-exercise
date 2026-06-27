import os
import time
import torch
from model import GPTConfig, GPT


def generate(model, img_h, img_w, temperature=0.8):
    context = torch.zeros((1, 1), dtype=torch.long)
    max_new_tokens = (img_h * img_w) - 1
    art = model.generate(context, max_new_tokens=max_new_tokens,
                         temperature=temperature)[0].reshape(img_h, img_w)
    for row in art:
        print("".join("##" if v else ".." for v in row))


def generate_samples(model, img_h, img_w, temperature=0.8, n_samples=10):
    sample_secs = []
    for i in range(n_samples):
        print(" " * (meta['img_w'] * 2))
        t0 = time.perf_counter()
        generate(model=model, img_h=img_h, img_w=img_w, temperature=temperature)
        sample_secs.append(time.perf_counter() - t0)

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
