import torch

def generate(model, img_h, img_w):
    context = torch.zeros((1, 2), dtype=torch.long)
    max_new_tokens = (img_h * img_w) - 2
    art = model.generate(context, max_new_tokens=max_new_tokens)[0].reshape(img_h, img_w)
    for row in art:
        print("".join("##" if v else ".." for v in row))