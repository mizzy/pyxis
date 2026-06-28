use crate::output_head::argmax;
use std::cell::Cell;

const DEFAULT_RNG_SEED: u64 = 42;

pub struct Sampler {
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    rng_state: Cell<u64>,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, repetition_penalty: f32) -> Self {
        Self {
            temperature,
            top_p,
            repetition_penalty,
            rng_state: Cell::new(DEFAULT_RNG_SEED),
        }
    }

    pub fn sample(&self, logits: &mut [f32], token_ids: &[u32]) -> usize {
        self.apply_repetition_penalty(logits, token_ids);

        if self.temperature == 0.0 {
            return argmax(logits);
        }

        self.apply_temperature(logits);
        self.apply_top_p_and_sample(logits)
    }

    fn apply_temperature(&self, logits: &mut [f32]) {
        for logit in logits.iter_mut() {
            *logit /= self.temperature;
        }
    }

    fn apply_top_p_and_sample(&self, logits: &mut [f32]) -> usize {
        assert!(!logits.is_empty());

        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs = logits
            .iter()
            .enumerate()
            .map(|(idx, &logit)| (idx, (logit - max).exp()))
            .collect::<Vec<_>>();
        let sum = probs.iter().map(|(_, prob)| prob).sum::<f32>();

        for (_, prob) in &mut probs {
            *prob /= sum;
        }

        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumulative = 0.0;
        let mut cutoff_idx = probs.len();
        for (idx, (_, prob)) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative >= self.top_p {
                cutoff_idx = idx + 1;
                break;
            }
        }

        let filtered = &probs[..cutoff_idx];
        let filtered_sum = filtered.iter().map(|(_, prob)| prob).sum::<f32>();
        let threshold = self.next_random_f32() * filtered_sum;

        let mut acc = 0.0;
        for &(token_idx, prob) in filtered {
            acc += prob;
            if acc >= threshold {
                return token_idx;
            }
        }

        filtered.last().unwrap().0
    }

    fn next_random_f32(&self) -> f32 {
        let next = xorshift64(self.rng_state.get());
        self.rng_state.set(next);
        (next as f32) / (u64::MAX as f32)
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

fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_penalty_leaves_logits_unchanged() {
        let sampler = Sampler::new(0.0, 1.0, 1.0);
        let mut logits = vec![1.0, 3.0, 2.0];

        sampler.apply_repetition_penalty(&mut logits, &[1]);

        assert_eq!(logits, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn penalty_reduces_positive_logits() {
        let sampler = Sampler::new(0.0, 1.0, 1.5);
        let mut logits = vec![1.0, 3.0, 2.0];

        sampler.apply_repetition_penalty(&mut logits, &[1]);

        assert_eq!(logits, vec![1.0, 2.0, 2.0]);
    }

    #[test]
    fn penalty_amplifies_negative_logits() {
        let sampler = Sampler::new(0.0, 1.0, 1.5);
        let mut logits = vec![-3.0, 1.0, 2.0];

        sampler.apply_repetition_penalty(&mut logits, &[0]);

        assert_eq!(logits, vec![-4.5, 1.0, 2.0]);
    }

    #[test]
    fn sample_returns_different_token_when_repeated() {
        let sampler = Sampler::new(0.0, 1.0, 2.0);
        let mut logits = vec![2.0, 3.0, 1.0];

        let token_id = sampler.sample(&mut logits, &[1]);

        assert_eq!(token_id, 0);
    }

    #[test]
    fn penalty_handles_duplicate_token_ids() {
        let sampler = Sampler::new(0.0, 1.0, 1.5);
        let mut logits = vec![1.0, 9.0, 2.0];

        sampler.apply_repetition_penalty(&mut logits, &[1, 1]);

        assert_eq!(logits, vec![1.0, 4.0, 2.0]);
    }

    #[test]
    fn temperature_zero_returns_argmax() {
        let sampler = Sampler::new(0.0, 1.0, 1.0);
        let mut logits = vec![1.0, 4.0, 2.0];

        let token_id = sampler.sample(&mut logits, &[]);

        assert_eq!(token_id, 1);
    }

    #[test]
    fn apply_temperature_scales_logits() {
        let sampler = Sampler::new(2.0, 1.0, 1.0);
        let mut logits = vec![2.0, 4.0];

        sampler.apply_temperature(&mut logits);

        assert_eq!(logits, vec![1.0, 2.0]);
    }

    #[test]
    fn top_p_one_samples_from_all() {
        let sampler = Sampler::new(1.0, 1.0, 1.0);
        let mut seen = [false; 3];

        for _ in 0..20 {
            let mut logits = vec![0.0, 0.0, 0.0];
            let token_id = sampler.sample(&mut logits, &[]);
            seen[token_id] = true;
        }

        assert_eq!(seen, [true, true, true]);
    }

    #[test]
    fn top_p_filters_low_probability_tokens() {
        let sampler = Sampler::new(1.0, 0.9, 1.0);

        for _ in 0..20 {
            let mut logits = vec![10.0, 0.0, 0.0];
            let token_id = sampler.sample(&mut logits, &[]);

            assert_eq!(token_id, 0);
        }
    }

    #[test]
    fn sample_with_temperature_and_top_p() {
        let sampler = Sampler::new(0.7, 0.9, 1.2);
        let mut logits = vec![1.0, 3.0, 2.0];

        let token_id = sampler.sample(&mut logits, &[1]);

        assert!(token_id < logits.len());
    }
}
