pub struct OutputHead {
    weight: Vec<f32>,
    vocab_size: usize,
    hidden_dim: usize,
}

impl OutputHead {
    pub fn new(weight: Vec<f32>, vocab_size: usize, hidden_dim: usize) -> Self {
        assert_eq!(weight.len(), vocab_size * hidden_dim);

        Self {
            weight,
            vocab_size,
            hidden_dim,
        }
    }

    pub fn logits(&self, hidden_state: &[f32]) -> Vec<f32> {
        assert_eq!(hidden_state.len(), self.hidden_dim);

        let mut logits = vec![0.0; self.vocab_size];

        for (token_idx, logit) in logits.iter_mut().enumerate() {
            let row_start = token_idx * self.hidden_dim;
            *logit = (0..self.hidden_dim)
                .map(|hidden_idx| hidden_state[hidden_idx] * self.weight[row_start + hidden_idx])
                .sum();
        }

        logits
    }

    pub fn greedy(&self, hidden_state: &[f32]) -> usize {
        argmax(&self.logits(hidden_state))
    }
}

pub fn argmax(values: &[f32]) -> usize {
    assert!(!values.is_empty());

    let mut max_idx = 0;
    let mut max_value = values[0];

    for (idx, value) in values.iter().copied().enumerate().skip(1) {
        if value > max_value {
            max_idx = idx;
            max_value = value;
        }
    }

    max_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logits_with_identity_weight() {
        let output_head = OutputHead::new(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            3,
            3,
        );

        let logits = output_head.logits(&[1.0, 2.0, 3.0]);

        assert_eq!(logits, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn logits_returns_correct_length() {
        let output_head = OutputHead::new(vec![0.0; 12], 4, 3);

        let logits = output_head.logits(&[1.0, 2.0, 3.0]);

        assert_eq!(logits.len(), 4);
    }

    #[test]
    fn greedy_returns_argmax_of_logits() {
        let output_head = OutputHead::new(vec![1.0, 0.0, 3.0, 0.0, 2.0, 0.0], 3, 2);

        let token_id = output_head.greedy(&[1.0, 1.0]);

        assert_eq!(token_id, 1);
    }

    #[test]
    fn argmax_returns_index_of_largest() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
    }

    #[test]
    fn argmax_handles_negative_values() {
        assert_eq!(argmax(&[-3.0, -1.0, -2.0]), 1);
    }

    #[test]
    #[should_panic]
    fn new_panics_on_wrong_weight_length() {
        OutputHead::new(vec![0.0; 10], 3, 4);
    }
}
