pub struct Embedding {
    weights: Vec<f32>,
    vocab_size: usize,
    hidden_dim: usize,
}

impl Embedding {
    pub fn new(weights: Vec<f32>, vocab_size: usize, hidden_dim: usize) -> Self {
        assert_eq!(weights.len(), vocab_size * hidden_dim);

        Self {
            weights,
            vocab_size,
            hidden_dim,
        }
    }

    pub fn lookup(&self, token_id: usize) -> &[f32] {
        assert!(token_id < self.vocab_size);

        let start = token_id * self.hidden_dim;
        let end = start + self.hidden_dim;
        &self.weights[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_returns_correct_row() {
        let weights = (0..12).map(|value| value as f32).collect();
        let embedding = Embedding::new(weights, 3, 4);

        assert_eq!(embedding.lookup(0), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(embedding.lookup(1), &[4.0, 5.0, 6.0, 7.0]);
        assert_eq!(embedding.lookup(2), &[8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    #[should_panic]
    fn lookup_panics_for_out_of_bounds_token_id() {
        let embedding = Embedding::new(vec![0.0; 12], 3, 4);

        embedding.lookup(3);
    }

    #[test]
    #[should_panic]
    fn new_panics_on_wrong_weight_length() {
        Embedding::new(vec![0.0; 10], 3, 4);
    }
}
