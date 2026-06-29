# Pyxis

A from-scratch LLM inference engine in Rust.

Pyxis loads a pre-trained transformer model and runs inference — no
dependency on Candle, llama.cpp, or any other inference library. The
entire transformer pipeline (attention, KV cache, RoPE, sampling) is
implemented from scratch.

Named after [Pyxis](https://en.wikipedia.org/wiki/Pyxis) (the Compass
constellation), part of the ancient Argo Navis ship family.

## Target Model

Qwen3-1.7B (24 transformer layers, 2048 hidden dimensions, ~151K
vocabulary).

## Components

| Component | Description |
|-----------|-------------|
| Safetensors parser | Read model weights from HuggingFace safetensors format |
| GGUF parser | Read model weights from GGUF format |
| Tensor operations | Matrix multiply, softmax, RMSNorm, RoPE, SiLU |
| Transformer block | Self-attention + FFN, repeated per layer |
| KV cache | Cache past key/value tensors for efficient autoregressive generation |
| Sampling | Temperature, top-p, repetition penalty |
| Quantization | BF16, int8, int4 (self-quantized at load time) |
| Matmul | Rayon multi-threaded with NEON SIMD, or Metal GPU compute shader |

## Usage

```sh
# Basic generation
pyxis /path/to/qwen3-1.7b/ "Hello"

# With int8 quantization
pyxis /path/to/qwen3-1.7b/ "Hello" --quantize

# With int4 quantization
pyxis /path/to/qwen3-1.7b/ "Hello" --quantize-int4

# With Metal GPU acceleration
pyxis /path/to/qwen3-1.7b/ "Hello" --metal

# Benchmark mode
pyxis /path/to/qwen3-1.7b/ "Hello" --bench
```

## License

MIT
