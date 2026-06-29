use crate::attention::Attention;
use crate::ffn::Ffn;
use crate::kv_cache::{KvCache, LayerCache};
use crate::rmsnorm::RmsNorm;

pub struct TransformerBlock {
    input_norm: RmsNorm,
    attention: Attention,
    post_attn_norm: RmsNorm,
    ffn: Ffn,
}

impl TransformerBlock {
    pub fn new(
        input_norm: RmsNorm,
        attention: Attention,
        post_attn_norm: RmsNorm,
        ffn: Ffn,
    ) -> Self {
        Self {
            input_norm,
            attention,
            post_attn_norm,
            ffn,
        }
    }

    pub fn forward(&self, x: &mut [f32], seq_len: usize, start_pos: usize, cache: &mut LayerCache) {
        assert!(seq_len > 0);
        assert_eq!(x.len() % seq_len, 0);
        let hidden_dim = x.len() / seq_len;

        let residual = x.to_vec();
        for position in x.chunks_exact_mut(hidden_dim) {
            self.input_norm.forward(position);
        }

        let attention_output = self.attention.forward(x, seq_len, start_pos, cache);
        assert_eq!(attention_output.len(), x.len());
        for ((value, residual), attention) in x.iter_mut().zip(residual).zip(attention_output) {
            *value = residual + attention;
        }

        let residual = x.to_vec();
        for position in x.chunks_exact_mut(hidden_dim) {
            self.post_attn_norm.forward(position);
        }

        for (position_idx, position) in x.chunks_exact_mut(hidden_dim).enumerate() {
            let ffn_output = self.ffn.forward(position);
            assert_eq!(ffn_output.len(), hidden_dim);
            let start = position_idx * hidden_dim;

            for ((value, residual), ffn) in position
                .iter_mut()
                .zip(&residual[start..start + hidden_dim])
                .zip(ffn_output)
            {
                *value = residual + ffn;
            }
        }
    }
}

pub struct Transformer {
    blocks: Vec<TransformerBlock>,
    final_norm: RmsNorm,
    hidden_dim: usize,
}

impl Transformer {
    pub fn new(blocks: Vec<TransformerBlock>, final_norm: RmsNorm, hidden_dim: usize) -> Self {
        Self {
            blocks,
            final_norm,
            hidden_dim,
        }
    }

    pub fn forward(&self, x: &mut [f32], seq_len: usize, start_pos: usize, kv_cache: &mut KvCache) {
        assert_eq!(x.len(), seq_len * self.hidden_dim);

        for (block_idx, block) in self.blocks.iter().enumerate() {
            block.forward(x, seq_len, start_pos, kv_cache.layer_mut(block_idx));
        }

        for position in x.chunks_exact_mut(self.hidden_dim) {
            self.final_norm.forward(position);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{KvCache, LayerCache};

    const HIDDEN_DIM: usize = 4;
    const NUM_Q_HEADS: usize = 2;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 2;
    const INTERMEDIATE_SIZE: usize = 8;
    const EPS: f32 = 1e-6;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {actual} to be close to {expected}"
        );
    }

    fn assert_vec_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.iter().zip(expected) {
            assert_close(*actual, *expected);
        }
    }

    fn norm() -> RmsNorm {
        RmsNorm::new(vec![1.0; HIDDEN_DIM], EPS)
    }

    fn attention_norm() -> Option<RmsNorm> {
        Some(RmsNorm::new(vec![1.0; HEAD_DIM], EPS))
    }

    fn zero_attention() -> Attention {
        Attention::new(
            vec![0.0; HIDDEN_DIM * HIDDEN_DIM],
            vec![0.0; HIDDEN_DIM * HIDDEN_DIM],
            vec![0.0; HIDDEN_DIM * HIDDEN_DIM],
            vec![0.0; HIDDEN_DIM * HIDDEN_DIM],
            attention_norm(),
            attention_norm(),
            HIDDEN_DIM,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            10.0,
        )
    }

    fn identity_attention() -> Attention {
        Attention::new(
            identity_weight(HIDDEN_DIM),
            identity_weight(HIDDEN_DIM),
            identity_weight(HIDDEN_DIM),
            identity_weight(HIDDEN_DIM),
            attention_norm(),
            attention_norm(),
            HIDDEN_DIM,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            10.0,
        )
    }

    fn zero_ffn() -> Ffn {
        Ffn::new(
            vec![0.0; INTERMEDIATE_SIZE * HIDDEN_DIM],
            vec![0.0; INTERMEDIATE_SIZE * HIDDEN_DIM],
            vec![0.0; HIDDEN_DIM * INTERMEDIATE_SIZE],
            HIDDEN_DIM,
            INTERMEDIATE_SIZE,
        )
    }

    fn zero_block() -> TransformerBlock {
        TransformerBlock::new(norm(), zero_attention(), norm(), zero_ffn())
    }

    fn identity_weight(size: usize) -> Vec<f32> {
        let mut weight = vec![0.0; size * size];
        for i in 0..size {
            weight[i * size + i] = 1.0;
        }
        weight
    }

    fn rms_norm_values(x: &[f32]) -> Vec<f32> {
        let mean_square = x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
        let factor = 1.0 / (mean_square + EPS).sqrt();
        x.iter().map(|value| value * factor).collect()
    }

    #[test]
    fn transformer_block_forward_applies_residual_connections() {
        let block = zero_block();
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let expected = x.clone();
        let mut cache = LayerCache::new(HIDDEN_DIM);

        block.forward(&mut x, 1, 0, &mut cache);

        assert_vec_close(&x, &expected);
    }

    #[test]
    fn transformer_block_forward_seq_len_2() {
        let block = zero_block();
        let mut x = vec![1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0];
        let expected = x.clone();
        let mut cache = LayerCache::new(HIDDEN_DIM);

        block.forward(&mut x, 2, 0, &mut cache);

        assert_vec_close(&x, &expected);
    }

    #[test]
    fn transformer_forward_stacks_blocks() {
        let transformer = Transformer::new(vec![zero_block(), zero_block()], norm(), HIDDEN_DIM);
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let expected = rms_norm_values(&x);
        let mut cache = KvCache::new(2, HIDDEN_DIM);

        transformer.forward(&mut x, 1, 0, &mut cache);

        assert_vec_close(&x, &expected);
    }

    #[test]
    fn transformer_forward_with_identity_attention() {
        let block = TransformerBlock::new(norm(), identity_attention(), norm(), zero_ffn());
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let input = x.clone();
        let expected_attention = rms_norm_values(&input);
        let expected: Vec<_> = input
            .iter()
            .zip(expected_attention)
            .map(|(residual, attention)| residual + attention)
            .collect();
        let mut cache = LayerCache::new(HIDDEN_DIM);

        block.forward(&mut x, 1, 0, &mut cache);

        assert_ne!(x, input);
        assert_vec_close(&x, &expected);
    }
}
