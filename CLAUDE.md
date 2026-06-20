# CLAUDE.md

## What is Pyxis

A from-scratch LLM inference engine in Rust. No external inference
library dependencies (no Candle, no llama.cpp). The transformer
inference pipeline is implemented entirely from first principles.

## Architecture

The inference pipeline follows the standard transformer flow:

```
Input text
  → Tokenizer (HF `tokenizers` crate)
  → Embedding table lookup
  → Transformer blocks × N layers:
      RMSNorm → Attention (Q/K/V projection, scaled dot-product, output projection)
      → Residual connection
      → RMSNorm → FFN (gate + up → SiLU → down)
      → Residual connection
  → Output head (embedding matrix, transposed)
  → Sampling (temperature, top-p, repetition penalty)
  → Output token
  → Repeat (with KV cache)
```

## Key Design Decisions

- **CPU-first**: No GPU/Metal code initially. Raw slice operations
  or `ndarray` for tensor math.
- **Safetensors format**: Model weights are loaded from HuggingFace
  safetensors files (JSON header + raw byte arrays).
- **Tokenizer**: Uses the `tokenizers` crate (HuggingFace) for BPE.
  Not reimplemented — the transformer itself is where the learning
  value is.
- **KV cache**: Required for usable autoregressive generation.
  Without it, generation is O(n²) per token.

## Build & Test

```bash
cargo check
cargo nextest run    # or cargo test
cargo clippy -- -D warnings
```

## Target Model

Qwen3-1.7B (24 transformer layers, 2048 hidden dimensions, ~151K
vocabulary). The model files are:

```
config.json            # architecture (num_layers, hidden_size, etc.)
tokenizer.json         # BPE vocabulary and merge rules
model.safetensors      # weights (~1GB quantized Q4, ~6.8GB fp32)
```

## Safetensors Key Names (Qwen3-1.7B)

```
model.embed_tokens.weight                      # embedding table [151936, 2048]
model.layers.{0..23}.self_attn.q_proj.weight   # Wq per layer
model.layers.{0..23}.self_attn.k_proj.weight   # Wk per layer
model.layers.{0..23}.self_attn.v_proj.weight   # Wv per layer
model.layers.{0..23}.self_attn.o_proj.weight   # output projection per layer
model.layers.{0..23}.mlp.gate_proj.weight      # FFN gate
model.layers.{0..23}.mlp.up_proj.weight        # FFN up
model.layers.{0..23}.mlp.down_proj.weight      # FFN down
model.layers.{0..23}.input_layernorm.weight    # RMSNorm before attention
model.layers.{0..23}.post_attention_layernorm.weight  # RMSNorm before FFN
model.norm.weight                              # final RMSNorm
lm_head.weight                                 # output head (may be tied to embed_tokens)
```

## Git & PR Workflow

- PRs are created as regular (non-draft) PRs unless explicitly told otherwise.

## Code Style

- All written output in English: commit messages, code comments, PR titles, PR descriptions, and GitHub issue content
- No unnecessary abstractions — straightforward, readable math code
