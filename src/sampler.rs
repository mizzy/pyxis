use crate::output_head::argmax;

pub struct Sampler {
    repetition_penalty: f32,
}

impl Sampler {
    pub fn new(repetition_penalty: f32) -> Self {
        Self { repetition_penalty }
    }

    pub fn sample(&self, logits: &mut [f32], token_ids: &[u32]) -> usize {
        self.apply_repetition_penalty(logits, token_ids);
        argmax(logits)
    }

    fn apply_repetition_penalty(&self, logits: &mut [f32], token_ids: &[u32]) {
        if self.repetition_penalty == 1.0 {
            return;
        }

        for token_id in token_ids {
            let idx = *token_id as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.repetition_penalty;
                } else {
                    logits[idx] *= self.repetition_penalty;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_penalty_leaves_logits_unchanged() {
        let sampler = Sampler::new(1.0);
        let mut logits = vec![1.0, 3.0, 2.0];

        sampler.apply_repetition_penalty(&mut logits, &[1]);

        assert_eq!(logits, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn penalty_reduces_positive_logits() {
        let sampler = Sampler::new(1.5);
        let mut logits = vec![1.0, 3.0, 2.0];

        sampler.apply_repetition_penalty(&mut logits, &[1]);

        assert_eq!(logits, vec![1.0, 2.0, 2.0]);
    }

    #[test]
    fn penalty_amplifies_negative_logits() {
        let sampler = Sampler::new(1.5);
        let mut logits = vec![-3.0, 1.0, 2.0];

        sampler.apply_repetition_penalty(&mut logits, &[0]);

        assert_eq!(logits, vec![-4.5, 1.0, 2.0]);
    }

    #[test]
    fn sample_returns_different_token_when_repeated() {
        let sampler = Sampler::new(2.0);
        let mut logits = vec![2.0, 3.0, 1.0];

        let token_id = sampler.sample(&mut logits, &[1]);

        assert_eq!(token_id, 0);
    }

    #[test]
    fn penalty_handles_duplicate_token_ids() {
        let sampler = Sampler::new(1.5);
        let mut logits = vec![1.0, 9.0, 2.0];

        sampler.apply_repetition_penalty(&mut logits, &[1, 1]);

        assert_eq!(logits, vec![1.0, 4.0, 2.0]);
    }
}
