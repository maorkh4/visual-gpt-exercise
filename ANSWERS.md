## Part 1 — Tokenize (understand the representation)

`prepare.py` already builds the tokens. Open it and answer:

- **Q1.** What is `vocab_size`, and why that number?

    `vocab_size` is 2, images had binary pixels

- **Q2.** We binarize at threshold 128. What information is thrown away? When would that
  hurt (think ahead to color)?

    The gray colors (boundaries) change by the threshold, so depending on it - the digit looks different. Low threshold - more ink (looks like a stain), low threshold - less ink, digit might not be noticeable

- **Q3.** Images are flattened row-major and concatenated into one long stream; decoding does
  `.reshape(H, W)`. Why must the model's notion of *position* line up with this exact order?

    Each pixel should look more or less like its neighbors, so the "left" and "upper" pixel might affect more than other pixel. The position is important because the model has to get the data from the neighbors which doesn't have to be the previous tokens (like in the case of the first token in the row)
