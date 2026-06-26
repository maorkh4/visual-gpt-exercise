## Part 1 — Tokenize (understand the representation)

`prepare.py` already builds the tokens. Open it and answer:

**Q1.** What is `vocab_size`, and why that number?

`vocab_size` is 2. `prepare.py` binarizes every pixel to 0 or 1, so the whole dataset uses only
two distinct token values, and `vocab_size` is just the count of distinct tokens.

**Q2.** We binarize at threshold 128. What information is thrown away? When would that hurt
(think ahead to color)?

All grayscale intensity is thrown away. Each pixel collapses from 256 possible levels down to
a single on/off bit. Pixels above the threshold become ink and below it become background, so
the soft edges disappear, and exactly which edge pixels survive depends on where the threshold
sits: a high threshold keeps less ink (thin or broken strokes), a low threshold keeps more
(stain-like). One bit per pixel is enough for a black-on-white digit, but not for anything with
real shading. Grayscale images need many brightness levels per pixel, and color needs many
levels across several channels (e.g. R/G/B). In those cases binarizing throws away the very
information that defines the image, and a far larger vocabulary is needed to capture it.

**Q3.** Images are flattened row-major and concatenated into one long stream; decoding does
`.reshape(H, W)`. Why must the model's notion of *position* line up with this exact order?

The positional embedding is indexed by absolute position in the stream, and the model learns
spatial relationships from those indices, so the stream order must be exactly the row-major
order that decoding reverses with `reshape(H, W)`. A pixel correlates with its 2-D neighbors,
but after row-major flattening the pixel directly above sits `W` positions earlier, and the
first pixel of a row has no meaningful immediate predecessor (its left "neighbor" in the stream
is the end of the previous row). If you encoded in one order but decoded in another, every
position-to-position spatial relationship the model learned would point at the wrong pixel.

---

## Part 3 — Generate

**Q.** An image is `seq_len` pixels. How many tokens do you ask `generate()` to produce, and
what do you **seed** it with? (Hint: what's almost always in MNIST's top-left corner?)

I seed with a single token of value 0 and ask `generate()` for `(img_h * img_w) - 1` new
tokens. The seed is one pixel and the generated tokens are the other `seq_len - 1`, so together
they make a full `seq_len`-pixel image. I seed with 0 because that's the background value, and
the top-left corner of an MNIST digit is almost always empty background, so it's a safe, neutral
pixel to start the model off from.

- **Then** run `prepare.py --downscale=1` to rebuild at full 28x28, retrain, and **time the
  generation.** Why is it so much slower?

Because the cost is driven by sequence length, and going from 14x14 to 28x28 takes the image
from 196 tokens to 784, which is 4 times larger. Attention is O(T^2) and sampling is sequential
at roughly O(T^3), so a 4 times longer sequence makes both training and generation blow up, with
generation hit hardest because each new pixel is produced one at a time over a longer context.
My own runs (1500 iters each) show it:

| resolution | tokens | train loss | val loss | training time | sampling (per image) |
|------------|--------|-----------|----------|---------------|----------------------|
| 14x14      | 196    | 0.1550    | 0.1528   | 12:12         | 0.57s                |
| 28x28      | 784    | 0.0883    | 0.0874   | 59:08         | 5.46s                |

4 times the tokens cost about 4.8 times the training time and about 9.6 times the per-image
sampling time. The loss is lower at 28x28 (more pixels to fit a sharper digit), but the time
grows much faster than the token count, which is exactly why 14x14 is the default.

---

## You should be able to answer

**1.** Why does the *same* `model.py` work for both text and images?

`model.py` is a generic sequence model over integer tokens, with nothing in it that is
text-specific. It embeds token ids, runs attention over positions, and predicts the next token
id. Whether those ids mean characters or pixels is decided entirely by how the data is prepared
and decoded, not by the model.

**2.** Why can't you batch images the way you batch text? *(the deep one)*

Because the model's positional embedding sets what each position means, and only whole images
line up with it. For text you pick a random offset into one long token stream and take a
contiguous chunk, and any chunk is a valid example because text has no fixed alignment. For
images that breaks: position 0 means the top-left pixel, but a random offset would land position
0 on some arbitrary pixel in the image, and a different one every batch, so the positions the
model learned would point at the wrong pixels. The solution is to keep the data as whole images
(shape `(num_images, seq_len)`), sample whole rows by image index, and slice `x = batch[:, :-1]`,
`y = batch[:, 1:]` so every example starts at the real top-left and position `i` always means
the same pixel.

**3.** How are "image in" and "image out" the same operation?

Both are next-token prediction over the same sequence. Reading an image and generating one differ
only in whether the next token comes from the data or from the model's own sample. Attention
only ever consumes the preceding context to produce a distribution over the next token, so
feeding in a real image (training) and sampling a new one (generation) run the identical forward
pass.

**4.** What makes sequence length the dominant cost?

Self-attention cost grows quadratically with sequence length `T`. Attention builds a `T x T`
score matrix where every position attends to every other, so both compute and memory scale as
`T^2`. For images `T = H x W`, so doubling the resolution quadruples the token count and raises
attention cost by roughly 16x. That `T^2` term dominates everything else in the model.
