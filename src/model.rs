use crate::attention::Attention;
use crate::embedding::Embedding;
use crate::ffn::Ffn;
use crate::kv_cache::KvCache;
use crate::output_head::OutputHead;
use crate::rmsnorm::RmsNorm;
use crate::safetensors::SafeTensors;
use crate::tokenizer::PyxisTokenizer;
use crate::transformer::{Transformer, TransformerBlock};
use serde::Deserialize;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const QWEN3_EOS_TOKEN_ID: u32 = 151_645;

pub struct Model {
    tokenizer: PyxisTokenizer,
    embedding: Embedding,
    transformer: Transformer,
    output_head: OutputHead,
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
        let tensors = SafeTensors::load(&find_safetensors_file(model_dir)?)?;

        let embed_tokens_weight = tensors.tensor_f32("model.embed_tokens.weight")?;
        let embedding = Embedding::new(
            embed_tokens_weight.clone(),
            config.vocab_size,
            config.hidden_size,
        );

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let input_norm = RmsNorm::new(
                tensors.tensor_f32(&format!("model.layers.{layer_idx}.input_layernorm.weight"))?,
                config.rms_norm_eps,
            );
            let attention = Attention::new(
                tensors.tensor_f32(&format!("model.layers.{layer_idx}.self_attn.q_proj.weight"))?,
                tensors.tensor_f32(&format!("model.layers.{layer_idx}.self_attn.k_proj.weight"))?,
                tensors.tensor_f32(&format!("model.layers.{layer_idx}.self_attn.v_proj.weight"))?,
                tensors.tensor_f32(&format!("model.layers.{layer_idx}.self_attn.o_proj.weight"))?,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rope_theta,
            );
            let post_attn_norm = RmsNorm::new(
                tensors.tensor_f32(&format!(
                    "model.layers.{layer_idx}.post_attention_layernorm.weight"
                ))?,
                config.rms_norm_eps,
            );
            let ffn = Ffn::new(
                tensors.tensor_f32(&format!("model.layers.{layer_idx}.mlp.gate_proj.weight"))?,
                tensors.tensor_f32(&format!("model.layers.{layer_idx}.mlp.up_proj.weight"))?,
                tensors.tensor_f32(&format!("model.layers.{layer_idx}.mlp.down_proj.weight"))?,
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
            tensors.tensor_f32("model.norm.weight")?,
            config.rms_norm_eps,
        );
        let transformer = Transformer::new(blocks, final_norm, config.hidden_size);
        let output_weight = if config.tie_word_embeddings {
            embed_tokens_weight
        } else {
            tensors.tensor_f32("lm_head.weight")?
        };
        let output_head = OutputHead::new(output_weight, config.vocab_size, config.hidden_size);

        Ok(Self {
            tokenizer,
            embedding,
            transformer,
            output_head,
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
        let mut next_token_id = self.output_head.greedy(last_hidden) as u32;

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
            next_token_id = self.output_head.greedy(&x) as u32;
        }

        self.tokenizer.decode(&generated_token_ids)
    }
}

fn read_config(path: &Path) -> io::Result<ModelConfig> {
    let json = fs::read_to_string(path)?;
    serde_json::from_str(&json).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn find_safetensors_file(model_dir: &Path) -> io::Result<PathBuf> {
    let model_file = model_dir.join("model.safetensors");
    if model_file.exists() {
        return Ok(model_file);
    }

    if model_dir.join("model.safetensors.index.json").exists() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "sharded models not supported",
        ));
    }

    let mut candidates = Vec::new();
    for entry in fs::read_dir(model_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("safetensors") {
            candidates.push(path);
        }
    }
    candidates.sort();

    candidates
        .into_iter()
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no safetensors file found"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;

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
    fn find_safetensors_file_returns_single_file() {
        let dir = tempfile::tempdir().unwrap();
        let model_file = dir.path().join("model.safetensors");
        File::create(&model_file).unwrap();

        let path = find_safetensors_file(dir.path()).unwrap();

        assert_eq!(path, model_file);
    }

    #[test]
    fn find_safetensors_file_errors_on_sharded() {
        let dir = tempfile::tempdir().unwrap();
        File::create(dir.path().join("model.safetensors.index.json")).unwrap();

        let error = find_safetensors_file(dir.path()).unwrap_err();

        assert!(error.to_string().contains("sharded models not supported"));
    }

    #[test]
    fn find_safetensors_file_errors_on_missing() {
        let dir = tempfile::tempdir().unwrap();

        let error = find_safetensors_file(dir.path()).unwrap_err();

        assert!(error.to_string().contains("no safetensors file found"));
    }
}
