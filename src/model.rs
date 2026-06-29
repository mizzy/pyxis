use crate::attention::Attention;
use crate::embedding::Embedding;
use crate::ffn::Ffn;
use crate::kv_cache::KvCache;
use crate::output_head::OutputHead;
use crate::rmsnorm::RmsNorm;
use crate::safetensors::TensorStore;
use crate::sampler::Sampler;
use crate::tokenizer::PyxisTokenizer;
use crate::transformer::{Transformer, TransformerBlock};
use serde::Deserialize;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

const QWEN3_EOS_TOKEN_ID: u32 = 151_645;

pub struct BenchmarkResult {
    pub load_time_ms: f64,
    pub prefill_tokens: usize,
    pub prefill_time_ms: f64,
    pub prefill_tokens_per_sec: f64,
    pub decode_tokens: usize,
    pub decode_time_ms: f64,
    pub decode_tokens_per_sec: f64,
}

impl BenchmarkResult {
    pub fn display(&self) {
        eprintln!("=== Pyxis Benchmark ===");
        eprintln!("Model load:  {:.0} ms", self.load_time_ms);
        eprintln!(
            "Prefill:     {} tokens in {:.0} ms ({:.2} tokens/s)",
            self.prefill_tokens, self.prefill_time_ms, self.prefill_tokens_per_sec
        );
        eprintln!(
            "Decode:      {} tokens in {:.0} ms ({:.2} tokens/s)",
            self.decode_tokens, self.decode_time_ms, self.decode_tokens_per_sec
        );
        eprintln!(
            "Total:       {:.0} ms",
            self.load_time_ms + self.prefill_time_ms + self.decode_time_ms
        );
    }
}

pub struct Model {
    tokenizer: PyxisTokenizer,
    embedding: Embedding,
    transformer: Transformer,
    output_head: OutputHead,
    sampler: Sampler,
    hidden_dim: usize,
    num_layers: usize,
    kv_dim: usize,
    eos_token_id: u32,
}

#[derive(Deserialize)]
struct ModelConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    vocab_size: usize,
    tie_word_embeddings: bool,
}

impl Model {
    pub fn load(model_dir: &Path) -> io::Result<Self> {
        let tokenizer = PyxisTokenizer::load(&model_dir.join("tokenizer.json"))?;
        let config = read_config(&model_dir.join("config.json"))?;
        let tensor_store = TensorStore::load(model_dir)?;

        let embed_tokens_weight = tensor_store.tensor_f32("model.embed_tokens.weight")?;
        let embedding = Embedding::new(
            embed_tokens_weight.clone(),
            config.vocab_size,
            config.hidden_size,
        );

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let input_norm = RmsNorm::new(
                tensor_store
                    .tensor_f32(&format!("model.layers.{layer_idx}.input_layernorm.weight"))?,
                config.rms_norm_eps,
            );
            let q_norm_name = format!("model.layers.{layer_idx}.self_attn.q_norm.weight");
            let k_norm_name = format!("model.layers.{layer_idx}.self_attn.k_norm.weight");
            let q_norm = if tensor_store.has_tensor(&q_norm_name) {
                Some(RmsNorm::new(
                    tensor_store.tensor_f32(&q_norm_name)?,
                    config.rms_norm_eps,
                ))
            } else {
                None
            };
            let k_norm = if tensor_store.has_tensor(&k_norm_name) {
                Some(RmsNorm::new(
                    tensor_store.tensor_f32(&k_norm_name)?,
                    config.rms_norm_eps,
                ))
            } else {
                None
            };
            let attention = Attention::new(
                tensor_store
                    .tensor_f32(&format!("model.layers.{layer_idx}.self_attn.q_proj.weight"))?,
                tensor_store
                    .tensor_f32(&format!("model.layers.{layer_idx}.self_attn.k_proj.weight"))?,
                tensor_store
                    .tensor_f32(&format!("model.layers.{layer_idx}.self_attn.v_proj.weight"))?,
                tensor_store
                    .tensor_f32(&format!("model.layers.{layer_idx}.self_attn.o_proj.weight"))?,
                q_norm,
                k_norm,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rope_theta,
            );
            let post_attn_norm = RmsNorm::new(
                tensor_store.tensor_f32(&format!(
                    "model.layers.{layer_idx}.post_attention_layernorm.weight"
                ))?,
                config.rms_norm_eps,
            );
            let ffn = Ffn::new(
                tensor_store
                    .tensor_f32(&format!("model.layers.{layer_idx}.mlp.gate_proj.weight"))?,
                tensor_store.tensor_f32(&format!("model.layers.{layer_idx}.mlp.up_proj.weight"))?,
                tensor_store
                    .tensor_f32(&format!("model.layers.{layer_idx}.mlp.down_proj.weight"))?,
                config.hidden_size,
                config.intermediate_size,
            );

            blocks.push(TransformerBlock::new(
                input_norm,
                attention,
                post_attn_norm,
                ffn,
            ));
        }

        let final_norm = RmsNorm::new(
            tensor_store.tensor_f32("model.norm.weight")?,
            config.rms_norm_eps,
        );
        let transformer = Transformer::new(blocks, final_norm, config.hidden_size);
        let output_weight = if config.tie_word_embeddings {
            embed_tokens_weight
        } else {
            tensor_store.tensor_f32("lm_head.weight")?
        };
        let output_head = OutputHead::new(output_weight, config.vocab_size, config.hidden_size);
        let sampler = Sampler::new(0.7, 0.9, 1.2);

        Ok(Self {
            tokenizer,
            embedding,
            transformer,
            output_head,
            sampler,
            hidden_dim: config.hidden_size,
            num_layers: config.num_hidden_layers,
            kv_dim: config.num_key_value_heads * config.head_dim,
            eos_token_id: QWEN3_EOS_TOKEN_ID,
        })
    }

    pub fn new_kv_cache(&self) -> KvCache {
        KvCache::new(self.num_layers, self.kv_dim)
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> String {
        let mut token_ids = self.tokenizer.encode(prompt);
        let mut kv_cache = self.new_kv_cache();
        let mut generated_token_ids = Vec::new();

        if token_ids.is_empty() || max_tokens == 0 {
            return String::new();
        }

        let seq_len = token_ids.len();
        let mut x = Vec::with_capacity(seq_len * self.hidden_dim);
        for token_id in &token_ids {
            x.extend_from_slice(self.embedding.lookup(*token_id as usize));
        }
        self.transformer.forward(&mut x, seq_len, 0, &mut kv_cache);
        let last_start = (seq_len - 1) * self.hidden_dim;
        let last_hidden = &x[last_start..last_start + self.hidden_dim];
        let mut logits = self.output_head.logits(last_hidden);
        let mut next_token_id = self.sampler.sample(&mut logits, &token_ids) as u32;

        for _ in 0..max_tokens {
            if next_token_id == self.eos_token_id {
                break;
            }

            print!("{}", self.tokenizer.decode(&[next_token_id]));
            io::stdout().flush().expect("flush stdout");

            generated_token_ids.push(next_token_id);
            token_ids.push(next_token_id);

            let mut x = self.embedding.lookup(next_token_id as usize).to_vec();
            let start_pos = token_ids.len() - 1;
            self.transformer
                .forward(&mut x, 1, start_pos, &mut kv_cache);
            let mut logits = self.output_head.logits(&x);
            next_token_id = self.sampler.sample(&mut logits, &token_ids) as u32;
        }

        self.tokenizer.decode(&generated_token_ids)
    }

    pub fn benchmark(&self, prompt: &str, max_tokens: usize) -> BenchmarkResult {
        let mut token_ids = self.tokenizer.encode(prompt);
        if token_ids.is_empty() {
            return BenchmarkResult {
                load_time_ms: 0.0,
                prefill_tokens: 0,
                prefill_time_ms: 0.0,
                prefill_tokens_per_sec: 0.0,
                decode_tokens: 0,
                decode_time_ms: 0.0,
                decode_tokens_per_sec: 0.0,
            };
        }

        let mut kv_cache = self.new_kv_cache();
        let seq_len = token_ids.len();

        let prefill_start = Instant::now();
        let mut x = Vec::with_capacity(seq_len * self.hidden_dim);
        for token_id in &token_ids {
            x.extend_from_slice(self.embedding.lookup(*token_id as usize));
        }
        self.transformer.forward(&mut x, seq_len, 0, &mut kv_cache);
        let prefill_time_ms = elapsed_ms(prefill_start);

        let mut decode_tokens = 0;
        let mut decode_time_ms = 0.0;

        if max_tokens > 0 {
            let decode_start = Instant::now();
            let last_start = (seq_len - 1) * self.hidden_dim;
            let last_hidden = &x[last_start..last_start + self.hidden_dim];
            let mut logits = self.output_head.logits(last_hidden);
            let mut next_token_id = self.sampler.sample(&mut logits, &token_ids) as u32;

            for _ in 0..max_tokens {
                if next_token_id == self.eos_token_id {
                    break;
                }

                decode_tokens += 1;
                token_ids.push(next_token_id);

                let mut x = self.embedding.lookup(next_token_id as usize).to_vec();
                let start_pos = token_ids.len() - 1;
                self.transformer
                    .forward(&mut x, 1, start_pos, &mut kv_cache);
                let mut logits = self.output_head.logits(&x);
                next_token_id = self.sampler.sample(&mut logits, &token_ids) as u32;
            }

            decode_time_ms = elapsed_ms(decode_start);
        }

        BenchmarkResult {
            load_time_ms: 0.0,
            prefill_tokens: seq_len,
            prefill_time_ms,
            prefill_tokens_per_sec: tokens_per_second(seq_len, prefill_time_ms),
            decode_tokens,
            decode_time_ms,
            decode_tokens_per_sec: tokens_per_second(decode_tokens, decode_time_ms),
        }
    }
}

fn read_config(path: &Path) -> io::Result<ModelConfig> {
    let json = fs::read_to_string(path)?;
    serde_json::from_str(&json).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn tokens_per_second(tokens: usize, elapsed_ms: f64) -> f64 {
    if elapsed_ms == 0.0 {
        0.0
    } else {
        tokens as f64 / (elapsed_ms / 1000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_config_deserializes() {
        let config = serde_json::from_str::<ModelConfig>(
            r#"{
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 6144,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000.0,
                "vocab_size": 151936,
                "tie_word_embeddings": true
            }"#,
        )
        .unwrap();

        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.intermediate_size, 6144);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.rope_theta, 1_000_000.0);
        assert_eq!(config.vocab_size, 151_936);
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn benchmark_result_has_positive_values() {
        let result = BenchmarkResult {
            load_time_ms: 10.0,
            prefill_tokens: 5,
            prefill_time_ms: 20.0,
            prefill_tokens_per_sec: 250.0,
            decode_tokens: 7,
            decode_time_ms: 30.0,
            decode_tokens_per_sec: 233.33,
        };

        assert_eq!(result.load_time_ms, 10.0);
        assert_eq!(result.prefill_tokens, 5);
        assert_eq!(result.prefill_time_ms, 20.0);
        assert_eq!(result.prefill_tokens_per_sec, 250.0);
        assert_eq!(result.decode_tokens, 7);
        assert_eq!(result.decode_time_ms, 30.0);
        assert_eq!(result.decode_tokens_per_sec, 233.33);
    }
}
