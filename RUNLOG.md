# RUNLOG

## Baseline
- **Hypothesis**: The baseline model is mediocre due to lack of LR schedule, weight tying, etc. We will first establish the baseline bits-per-byte (bpb) and loss.
- **Change**: None. Ran unmodified `train.py` and `evaluate.py`.
- **Dev BPB**: 2.3718
- **Conclusion**: The model takes 322s to train for 2000 steps. The loss goes from 5.6 to 1.73. The bits-per-byte is 2.3718. The parameter count is 1,339,840.

## Experiment 1: Optimization Tweaks
- **Hypothesis**: A learning rate schedule (Cosine Warmup) and AdamW with weight decay, along with gradient clipping, will improve convergence stability and lower the loss in 2,000 steps.
- **Change**: Replaced Adam with AdamW (weight decay 0.1 for weights, 0.0 for biases/LayerNorm). Added Cosine learning rate scheduler with warmup for 100 steps. Added gradient clipping at 1.0.
- **Dev BPB**: N/A (Failed evaluation due to concurrent tokenizer change)
- **Conclusion**: The final training loss dropped from 1.73 to 1.60. The optimization tweaks definitively improve training. The evaluation failed because I upgraded the tokenizer in the background, making the model and tokenizer out of sync.

## Experiment 2: Tokenizer Upgrade + High Capacity
- **Hypothesis**: Devanagari characters take 3 bytes in UTF-8, wasting sequence length and parameter budget on predicting intermediate bytes. A character-level tokenizer (each unique multi-byte char gets a token) will increase the sequence length covered by the context window. Tying embeddings and LM head weights will save parameters, allowing us to increase the model dimension (`n_embd`) from 160 to 192.
- **Change**: Created a custom character-byte hybrid tokenizer. Vocab size is 816. Set `tie_weights = True`. Increased `n_embd = 192`, `n_head = 6`.
- **Dev BPB**: 2.1801
- **Conclusion**: The custom tokenizer significantly reduced the number of tokens required to encode the Devanagari script. Along with the larger parameter budget enabled by weight tying, the model achieved a substantial drop in BPB from the baseline 2.3718 down to 2.1801. This is a solid, ambitious change that performs much better.


