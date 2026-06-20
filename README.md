# Pyxis

A from-scratch LLM inference engine in Rust.

Pyxis loads a pre-trained transformer model (safetensors format) and runs
inference on CPU — no dependency on Candle, llama.cpp, or any other
inference library. The entire transformer pipeline (attention, KV cache,
RoPE, sampling) is implemented from scratch.

Named after [Pyxis](https://en.wikipedia.org/wiki/Pyxis) (the Compass
constellation), part of the ancient Argo Navis ship family.

## Status

Early development. Not yet functional.

## Goal

Load a small open model (Qwen3-1.7B class) and generate text from a
prompt on CPU. Performance target: ~1 token/sec without quantization,
5-10 tokens/sec with Q4/Q8 quantization.

## Components

| Component | Description |
|-----------|-------------|
| Safetensors parser | Read model weights from HuggingFace safetensors format |
| Tensor operations | Matrix multiply, softmax, RMSNorm, RoPE, SiLU |
| Transformer block | Self-attention + FFN, repeated per layer |
| KV cache | Cache past key/value tensors for efficient autoregressive generation |
| Sampling | Temperature, top-p, repetition penalty |
| Quantization | Q4/Q8 weight loading and integer math (planned) |

## Usage (planned)

```rust
use pyxis::Model;

let model = Model::load("/path/to/qwen3-1.7b-q4/")?;
let mut session = model.session();

// Generate text
let output = session.generate("Explain this:", &Default::default())?;

// Streaming
session.generate_streaming("Explain this:", &Default::default(), |token| {
    print!("{}", token);
})?;

// Multi-turn: KV cache preserved across calls
let answer = session.generate("Follow-up question?", &Default::default())?;
```

## License

MIT
