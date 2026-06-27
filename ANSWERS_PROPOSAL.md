# Proposal: making `ANSWERS.md` review-ready

This is a proposed **rewrite** of your answers — plain prose, tightened and technically
corrected. It does **not** touch `ANSWERS.md`; review it here and copy across what you like.

The framing: keep each answer to a short paragraph that leads with the direct answer, then
gives the mechanism behind it. No headings or labels, just sentences — but each answer starts
with the conclusion so a reviewer can agree or disagree in the first line, and ends with the
"why" so they can check the reasoning.

I corrected the technical content where it was off or thin (noted in "What I changed" at the
end). The one exception is the batching question (Core Q2): that's the Part 2 trap the
exercise wants you to crack yourself, so I left your wording and added a nudge instead of the
answer.

---

## Proposed answers

### Part 1

**Q1 — vocab_size.**
`vocab_size` is 2. `prepare.py` binarizes every pixel to 0 or 1, so the whole dataset uses only
two distinct token values, and `vocab_size` is just the count of distinct tokens.

**Q2 — what binarizing at 128 throws away.**
All grayscale intensity is thrown away — each pixel collapses from 256 possible levels (one
byte) down to a single on/off bit. Pixels above the threshold become ink and below it become
background, so the soft anti-aliased edges and shading gradients disappear, and exactly which
edge pixels survive depends on where the cutoff sits: a high threshold keeps less ink (thin or
broken strokes), a low threshold keeps more (blotchy, stain-like). This hurts anywhere
intensity itself carries meaning — grayscale shading, and especially color, where each channel
needs many levels. Binarizing destroys exactly the information a color model depends on, so
you'd need a much larger vocabulary (or a continuous representation) instead of 1 bit per pixel.

**Q3 — why position must line up with row-major order.**
The positional embedding is indexed by absolute position in the stream, and the model learns
spatial relationships from those indices, so the stream order must be exactly the row-major
order that decoding reverses with `reshape(H, W)`. A pixel correlates with its 2-D neighbors,
but after row-major flattening the pixel directly above sits `W` positions earlier, and the
first pixel of a row has no meaningful immediate predecessor (its left "neighbor" in the stream
is the end of the previous row). If you encoded in one order but decoded in another, every
position-to-position spatial relationship the model learned would point at the wrong pixel.

### You should be able to answer

**1 — why the same `model.py` works for text and images.**
`model.py` is a generic sequence model over integer tokens — nothing in it is text-specific. It
embeds token ids, runs attention over positions, and predicts the next token id. Whether those
ids mean characters or pixels is decided entirely by the data pipeline, not by the model.

**2 — why you can't batch images like text *(the deep one)*.**
With text, any random offset into the stream is a valid example; with images, an example has to
be a whole image generated from its top-left corner, so you can't just grab an arbitrary slice.

> *Left for you to sharpen — this is the Part 2 trap, so I'm not going to fill it in. Your draft
> is on the right track. Push on it: after you pick a random offset, what does the model think
> position 0 of that slice is, and is it the same pixel from one batch to the next? See
> `EXERCISE.md` Part 2 (~lines 99–118).*

**3 — how "image in" and "image out" are the same operation.**
Both are next-token prediction over the same sequence; "reading" an image and "generating" one
differ only in whether the next token comes from the data or from the model's own sample.
Attention only ever consumes the preceding context to produce a distribution over the next
token, so teacher-forced training (image in) and autoregressive sampling (image out) run the
identical forward pass.

**4 — what makes sequence length the dominant cost.**
Self-attention cost grows quadratically with sequence length `T`. Attention builds a `T × T`
score matrix — every position attends to every other — so both compute and memory scale as
`T²`. For images `T = H × W`, so doubling the resolution quadruples the token count and raises
attention cost by roughly 16×. That `T²` term dominates everything else in the model.

---

## What I changed (and why)

- **Q2 direction fix.** Your draft had both branches starting with "low threshold." Corrected
  to: high threshold → less ink, low threshold → more ink. Also reframed the core loss as
  "256 levels → 1 bit" rather than just "boundary colors change," which is the sharper way to
  say what's discarded.
- **Q3 made concrete.** Your point about neighbors was right; I made the mechanism explicit —
  the pixel above is `W` positions back, and row starts have no real predecessor — and tied it
  to the learned positional embedding.
- **Core Q1 tightened.** Replaced "generates vectors that can be interpreted as tokens" with
  the cleaner "generic sequence model over integer tokens; the data pipeline decides meaning."
- **Core Q2 — untouched.** Left as your wording plus a nudge (see above).
- **Core Q3 sharpened.** Added the training-vs-sampling framing (teacher forcing vs.
  autoregressive) as the concrete reason they're the same forward pass.
- **Core Q4 — committed to the shape.** Your draft said cost "rises" with `T`; the key word is
  *quadratically* (`T²`), with the resolution-doubling → ~16× example to make it land.
