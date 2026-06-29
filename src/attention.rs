use crate::kv_cache::LayerCache;
use crate::rmsnorm::RmsNorm;

pub struct Attention {
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    hidden_dim: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wq: Vec<f32>,
        wk: Vec<f32>,
        wv: Vec<f32>,
        wo: Vec<f32>,
        q_norm: Option<RmsNorm>,
        k_norm: Option<RmsNorm>,
        hidden_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope_theta: f32,
    ) -> Self {
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        assert_eq!(wq.len(), q_dim * hidden_dim);
        assert_eq!(wk.len(), kv_dim * hidden_dim);
        assert_eq!(wv.len(), kv_dim * hidden_dim);
        assert_eq!(wo.len(), hidden_dim * q_dim);

        Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            hidden_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
        }
    }

    pub fn forward(
        &self,
        x: &[f32],
        seq_len: usize,
        start_pos: usize,
        cache: &mut LayerCache,
    ) -> Vec<f32> {
        let hidden_dim = self.hidden_dim;
        let q_dim = self.num_q_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        assert_eq!(x.len(), seq_len * hidden_dim);
        assert_eq!(cache.len(), start_pos);

        let mut queries = vec![0.0; seq_len * q_dim];
        let mut new_keys = vec![0.0; seq_len * kv_dim];
        let mut new_values = vec![0.0; seq_len * kv_dim];

        for pos in 0..seq_len {
            let input = &x[pos * hidden_dim..(pos + 1) * hidden_dim];

            let mut query = matmul(input, &self.wq, q_dim, hidden_dim);
            let mut key = matmul(input, &self.wk, kv_dim, hidden_dim);
            let value = matmul(input, &self.wv, kv_dim, hidden_dim);

            if let Some(ref norm) = self.q_norm {
                for head in query.chunks_exact_mut(self.head_dim) {
                    norm.forward(head);
                }
            }
            if let Some(ref norm) = self.k_norm {
                for head in key.chunks_exact_mut(self.head_dim) {
                    norm.forward(head);
                }
            }

            apply_rope(&mut query, self.head_dim, start_pos + pos, self.rope_theta);
            apply_rope(&mut key, self.head_dim, start_pos + pos, self.rope_theta);

            queries[pos * q_dim..(pos + 1) * q_dim].copy_from_slice(&query);
            new_keys[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&key);
            new_values[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(&value);
        }

        cache.append(&new_keys, &new_values, seq_len);
        let cached_len = cache.len();
        let keys = cache.keys();
        let values = cache.values();
        let mut output = vec![0.0; seq_len * hidden_dim];
        let scale = (self.head_dim as f32).sqrt();

        for pos in 0..seq_len {
            let mut attention_output = vec![0.0; q_dim];

            for q_head in 0..self.num_q_heads {
                let kv_head = q_head * self.num_kv_heads / self.num_q_heads;
                let q_start = pos * q_dim + q_head * self.head_dim;
                let mut scores = vec![f32::NEG_INFINITY; cached_len];
                let max_key_pos = start_pos + pos;

                for (key_pos, score) in scores.iter_mut().enumerate().take(max_key_pos + 1) {
                    let k_start = key_pos * kv_dim + kv_head * self.head_dim;
                    let dot = (0..self.head_dim)
                        .map(|dim| queries[q_start + dim] * keys[k_start + dim])
                        .sum::<f32>();
                    *score = dot / scale;
                }

                softmax(&mut scores);

                let out_start = q_head * self.head_dim;
                for (key_pos, score) in scores.iter().enumerate() {
                    let v_start = key_pos * kv_dim + kv_head * self.head_dim;

                    for dim in 0..self.head_dim {
                        attention_output[out_start + dim] += score * values[v_start + dim];
                    }
                }
            }

            let projected = matmul(&attention_output, &self.wo, hidden_dim, q_dim);
            output[pos * hidden_dim..(pos + 1) * hidden_dim].copy_from_slice(&projected);
        }

        output
    }
}

fn matmul(input: &[f32], weight: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    assert_eq!(input.len(), in_features);
    assert_eq!(weight.len(), out_features * in_features);

    let mut output = vec![0.0; out_features];

    for (out_idx, output_value) in output.iter_mut().enumerate().take(out_features) {
        let row_start = out_idx * in_features;
        *output_value = (0..in_features)
            .map(|in_idx| input[in_idx] * weight[row_start + in_idx])
            .sum();
    }

    output
}

fn apply_rope(q_or_k: &mut [f32], head_dim: usize, pos: usize, theta: f32) {
    assert_eq!(q_or_k.len() % head_dim, 0);
    assert_eq!(head_dim % 2, 0);

    let half = head_dim / 2;
    for head in q_or_k.chunks_exact_mut(head_dim) {
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos = angle.cos();
            let sin = angle.sin();
            let x0 = head[i];
            let x1 = head[i + half];

            head[i] = x0 * cos - x1 * sin;
            head[i + half] = x1 * cos + x0 * sin;
        }
    }
}

fn softmax(scores: &mut [f32]) {
    let max = scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |acc, score| acc.max(score));
    let mut sum = 0.0;

    for score in scores.iter_mut() {
        *score = (*score - max).exp();
        sum += *score;
    }

    for score in scores {
        *score /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::LayerCache;
    use crate::rmsnorm::RmsNorm;

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

    fn identity_weight(size: usize) -> Vec<f32> {
        let mut weight = vec![0.0; size * size];
        for i in 0..size {
            weight[i * size + i] = 1.0;
        }
        weight
    }

    fn head_norm(weight: Vec<f32>) -> RmsNorm {
        RmsNorm::new(weight, 1e-6)
    }

    fn unit_head_norm(head_dim: usize) -> Option<RmsNorm> {
        Some(head_norm(vec![1.0; head_dim]))
    }

    #[test]
    fn matmul_computes_correctly() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let output = matmul(&input, &weight, 2, 3);

        assert_vec_close(&output, &[1.0, 2.0]);
    }

    #[test]
    fn apply_rope_at_position_zero_is_identity() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        apply_rope(&mut values, 4, 0, 10.0);

        assert_vec_close(&values, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn apply_rope_rotates_at_nonzero_position() {
        let mut values = vec![1.0, 0.0, 0.0, 1.0];
        let theta = 10.0_f32;
        let angle_0 = 1.0_f32;
        let angle_1 = 1.0 / theta.sqrt();

        apply_rope(&mut values, 4, 1, theta);

        assert_vec_close(
            &values,
            &[angle_0.cos(), -angle_1.sin(), angle_0.sin(), angle_1.cos()],
        );
    }

    #[test]
    fn softmax_produces_valid_distribution() {
        let mut scores = vec![1.0, 2.0, 3.0];

        softmax(&mut scores);

        assert_close(scores.iter().sum(), 1.0);
        assert!(scores[0] < scores[1]);
        assert!(scores[1] < scores[2]);
    }

    #[test]
    fn softmax_handles_large_values() {
        let mut scores = vec![1000.0, 1001.0, 1002.0];

        softmax(&mut scores);

        assert!(scores.iter().all(|score| score.is_finite()));
        assert_close(scores.iter().sum(), 1.0);
    }

    #[test]
    fn forward_single_token_identity_weights() {
        let hidden_dim = 8;
        let attention = Attention::new(
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            unit_head_norm(4),
            unit_head_norm(4),
            hidden_dim,
            2,
            2,
            4,
            10.0,
        );
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut cache = LayerCache::new(hidden_dim);

        let output = attention.forward(&x, 1, 0, &mut cache);

        assert_vec_close(&output, &x);
    }

    #[test]
    fn forward_causal_mask() {
        let hidden_dim = 8;
        let attention = Attention::new(
            vec![0.0; hidden_dim * hidden_dim],
            vec![0.0; hidden_dim * hidden_dim],
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            unit_head_norm(4),
            unit_head_norm(4),
            hidden_dim,
            2,
            2,
            4,
            10.0,
        );
        let x = vec![
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0,
        ];
        let mut cache = LayerCache::new(hidden_dim);

        let output = attention.forward(&x, 2, 0, &mut cache);

        assert_vec_close(&output[..hidden_dim], &x[..hidden_dim]);
        assert_vec_close(
            &output[hidden_dim..],
            &[3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0],
        );
    }

    #[test]
    fn forward_with_hidden_dim_smaller_than_q_dim() {
        let hidden_dim = 4;
        let num_q_heads = 2;
        let num_kv_heads = 2;
        let head_dim = 4;
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mut wv = vec![0.0; kv_dim * hidden_dim];
        let mut wo = vec![0.0; hidden_dim * q_dim];

        for dim in 0..hidden_dim {
            wv[dim * hidden_dim + dim] = 1.0;
            wo[dim * q_dim + dim] = 1.0;
        }

        let attention = Attention::new(
            vec![0.0; q_dim * hidden_dim],
            vec![0.0; kv_dim * hidden_dim],
            wv,
            wo,
            unit_head_norm(head_dim),
            unit_head_norm(head_dim),
            hidden_dim,
            num_q_heads,
            num_kv_heads,
            head_dim,
            10.0,
        );
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut cache = LayerCache::new(kv_dim);

        let output = attention.forward(&x, 2, 0, &mut cache);

        assert_eq!(output.len(), 2 * hidden_dim);
        assert_vec_close(&output, &[1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn forward_gqa_shares_kv_heads() {
        let hidden_dim = 16;
        let kv_dim = 8;
        let attention = Attention::new(
            vec![0.0; hidden_dim * hidden_dim],
            vec![0.0; kv_dim * hidden_dim],
            vec![0.0; kv_dim * hidden_dim],
            vec![0.0; hidden_dim * hidden_dim],
            unit_head_norm(4),
            unit_head_norm(4),
            hidden_dim,
            4,
            2,
            4,
            10.0,
        );
        let x = vec![1.0; hidden_dim * 3];
        let mut cache = LayerCache::new(kv_dim);

        let output = attention.forward(&x, 3, 0, &mut cache);

        assert_eq!(output.len(), hidden_dim * 3);
    }

    #[test]
    fn forward_without_qk_norm() {
        let hidden_dim = 8;
        let attention = Attention::new(
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            None,
            None,
            hidden_dim,
            2,
            2,
            4,
            10.0,
        );
        let x = vec![1.0, 0.5, -1.0, 2.0, 0.25, -0.5, 1.5, -2.0];
        let mut cache = LayerCache::new(hidden_dim);

        let output = attention.forward(&x, 1, 0, &mut cache);

        assert_eq!(output.len(), hidden_dim);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn forward_with_kv_cache_incremental() {
        let hidden_dim = 8;
        let attention = Attention::new(
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            unit_head_norm(4),
            unit_head_norm(4),
            hidden_dim,
            2,
            2,
            4,
            10.0,
        );
        let x = vec![
            1.0, 0.5, -1.0, 2.0, 0.25, -0.5, 1.5, -2.0, -0.75, 1.25, 0.0, 0.5, 2.0, -1.0, 0.75,
            -0.25, 0.3, -1.5, 1.2, 0.8, -0.4, 2.5, -0.9, 1.1,
        ];
        let mut full_cache = LayerCache::new(hidden_dim);
        let full_output = attention.forward(&x, 3, 0, &mut full_cache);

        let mut incremental_cache = LayerCache::new(hidden_dim);
        let _ = attention.forward(&x[..2 * hidden_dim], 2, 0, &mut incremental_cache);
        let incremental_output =
            attention.forward(&x[2 * hidden_dim..], 1, 2, &mut incremental_cache);

        assert_vec_close(&incremental_output, &full_output[2 * hidden_dim..]);
    }

    #[test]
    fn qk_norm_is_applied_per_head() {
        let hidden_dim = 8;
        let head_dim = 4;
        let input = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let k_norm_weight = vec![1.0, 2.0, 3.0, 4.0];
        let mut base_cache = LayerCache::new(hidden_dim);
        let base_attention = Attention::new(
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            Some(head_norm(vec![1.0; head_dim])),
            Some(head_norm(k_norm_weight.clone())),
            hidden_dim,
            2,
            2,
            head_dim,
            10.0,
        );
        let base_output = base_attention.forward(&input, 2, 0, &mut base_cache);

        let mut scaled_cache = LayerCache::new(hidden_dim);
        let scaled_attention = Attention::new(
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            identity_weight(hidden_dim),
            Some(head_norm(vec![2.0; head_dim])),
            Some(head_norm(k_norm_weight)),
            hidden_dim,
            2,
            2,
            head_dim,
            10.0,
        );
        let scaled_output = scaled_attention.forward(&input, 2, 0, &mut scaled_cache);

        assert!(scaled_output[hidden_dim] > base_output[hidden_dim] + 0.1);

        let second_key_rms = ((3.0 * 3.0 + 1.0) / head_dim as f32 + 1e-6).sqrt();
        let norm_0 = 3.0 / second_key_rms;
        let norm_2 = 0.0 / second_key_rms * 3.0;
        let expected_first = norm_0 * 1.0_f32.cos() - norm_2 * 1.0_f32.sin();
        assert_close(scaled_cache.keys()[hidden_dim], expected_first);
    }
}
