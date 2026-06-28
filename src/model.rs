use crate::attention::Attention;
use crate::embedding::Embedding;
use crate::ffn::Ffn;
use crate::rmsnorm::RmsNorm;
use crate::safetensors::SafeTensors;
use crate::transformer::{Transformer, TransformerBlock};
use serde::Deserialize;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const QWEN3_EOS_TOKEN_ID: u32 = 151_645;

pub struct Model {
    tokenizer: Tokenizer,
    embedding: Embedding,
    transformer: Transformer,
    lm_head_weight: Vec<f32>,
    vocab_size: usize,
    hidden_dim: usize,
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
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
        let config = load_config(&model_dir.join("config.json"))?;
        let safetensors_path = find_safetensors_file(model_dir)?;
        let safetensors = SafeTensors::load(&safetensors_path)?;

        let embedding_weight = safetensors.tensor_f32("model.embed_tokens.weight")?;
        let lm_head_weight = if config.tie_word_embeddings {
            embedding_weight.clone()
        } else {
            safetensors.tensor_f32("lm_head.weight")?
        };
        let embedding = Embedding::new(embedding_weight, config.vocab_size, config.hidden_size);

        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let input_norm = RmsNorm::new(
                safetensors.tensor_f32(&format!("model.layers.{i}.input_layernorm.weight"))?,
                config.rms_norm_eps,
            );
            let attention = Attention::new(
                safetensors.tensor_f32(&format!("model.layers.{i}.self_attn.q_proj.weight"))?,
                safetensors.tensor_f32(&format!("model.layers.{i}.self_attn.k_proj.weight"))?,
                safetensors.tensor_f32(&format!("model.layers.{i}.self_attn.v_proj.weight"))?,
                safetensors.tensor_f32(&format!("model.layers.{i}.self_attn.o_proj.weight"))?,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.rope_theta,
            );
            let post_attn_norm = RmsNorm::new(
                safetensors
                    .tensor_f32(&format!("model.layers.{i}.post_attention_layernorm.weight"))?,
                config.rms_norm_eps,
            );
            let ffn = Ffn::new(
                safetensors.tensor_f32(&format!("model.layers.{i}.mlp.gate_proj.weight"))?,
                safetensors.tensor_f32(&format!("model.layers.{i}.mlp.up_proj.weight"))?,
                safetensors.tensor_f32(&format!("model.layers.{i}.mlp.down_proj.weight"))?,
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
            safetensors.tensor_f32("model.norm.weight")?,
            config.rms_norm_eps,
        );
        let transformer = Transformer::new(blocks, final_norm, config.hidden_size);

        Ok(Self {
            tokenizer,
            embedding,
            transformer,
            lm_head_weight,
            vocab_size: config.vocab_size,
            hidden_dim: config.hidden_size,
        })
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> String {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .expect("failed to tokenize prompt");
        let mut token_ids = encoding.get_ids().to_vec();
        let mut generated_ids = Vec::with_capacity(max_tokens);

        if token_ids.is_empty() {
            return String::new();
        }

        for _ in 0..max_tokens {
            let seq_len = token_ids.len();
            let mut x = Vec::with_capacity(seq_len * self.hidden_dim);
            for token_id in &token_ids {
                x.extend_from_slice(self.embedding.lookup(*token_id as usize));
            }

            self.transformer.forward(&mut x, seq_len);

            let last_position = (seq_len - 1) * self.hidden_dim;
            let hidden_state = &x[last_position..last_position + self.hidden_dim];
            let logits = self.compute_logits(hidden_state);
            let next_token_id = argmax(&logits) as u32;

            if next_token_id == QWEN3_EOS_TOKEN_ID {
                break;
            }

            token_ids.push(next_token_id);
            generated_ids.push(next_token_id);

            let token = self
                .tokenizer
                .decode(&[next_token_id], true)
                .expect("failed to decode generated token");
            print!("{token}");
            io::stdout()
                .flush()
                .expect("failed to flush generated token");
        }

        self.tokenizer
            .decode(&generated_ids, true)
            .expect("failed to decode generated output")
    }

    fn compute_logits(&self, hidden_state: &[f32]) -> Vec<f32> {
        compute_logits(
            hidden_state,
            &self.lm_head_weight,
            self.vocab_size,
            self.hidden_dim,
        )
    }
}

fn load_config(path: &Path) -> io::Result<ModelConfig> {
    let contents = fs::read_to_string(path)?;
    serde_json::from_str(&contents).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))
}

fn find_safetensors_file(model_dir: &Path) -> io::Result<PathBuf> {
    let single_file = model_dir.join("model.safetensors");
    if single_file.exists() {
        return Ok(single_file);
    }

    if model_dir.join("model.safetensors.index.json").exists() {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "sharded safetensors models are not supported yet; expected a single model.safetensors file",
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

    candidates.into_iter().next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "no .safetensors file found in model directory",
        )
    })
}

fn compute_logits(
    hidden_state: &[f32],
    lm_head_weight: &[f32],
    vocab_size: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    assert_eq!(hidden_state.len(), hidden_dim);
    assert_eq!(lm_head_weight.len(), vocab_size * hidden_dim);

    let mut logits = vec![0.0; vocab_size];
    for (i, logit) in logits.iter_mut().enumerate() {
        let row_start = i * hidden_dim;
        *logit = (0..hidden_dim)
            .map(|j| hidden_state[j] * lm_head_weight[row_start + j])
            .sum();
    }
    logits
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_returns_index_of_largest() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn argmax_handles_negative_values() {
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
    }

    #[test]
    fn compute_logits_with_identity_weight() {
        let lm_head_weight = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        assert_eq!(
            compute_logits(&[1.0, 2.0, 3.0], &lm_head_weight, 3, 3),
            vec![1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn compute_logits_returns_correct_length() {
        let logits = compute_logits(&[1.0, 2.0, 3.0], &[0.0; 12], 4, 3);

        assert_eq!(logits.len(), 4);
    }
}
