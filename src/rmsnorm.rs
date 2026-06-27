pub struct RmsNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &mut [f32]) {
        assert_eq!(x.len(), self.weight.len());

        let mean_square = x.iter().map(|value| value * value).sum::<f32>() / x.len() as f32;
        let factor = 1.0 / (mean_square + self.eps).sqrt();

        for (value, weight) in x.iter_mut().zip(&self.weight) {
            *value *= factor * weight;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {actual} to be close to {expected}"
        );
    }

    #[test]
    fn forward_normalizes_uniform_vector() {
        let rmsnorm = RmsNorm::new(vec![1.0, 1.0, 1.0, 1.0], 1e-6);
        let mut x = vec![1.0, 1.0, 1.0, 1.0];

        rmsnorm.forward(&mut x);

        for value in x {
            assert_close(value, 1.0);
        }
    }

    #[test]
    fn forward_normalizes_with_weights() {
        let rmsnorm = RmsNorm::new(vec![0.5, 1.0], 1e-6);
        let mut x = vec![2.0, 4.0];

        rmsnorm.forward(&mut x);

        assert_close(x[0], 2.0 / 10.0_f32.sqrt() * 0.5);
        assert_close(x[1], 4.0 / 10.0_f32.sqrt());
    }

    #[test]
    fn forward_handles_zero_input() {
        let rmsnorm = RmsNorm::new(vec![1.0, 1.0], 1e-6);
        let mut x = vec![0.0, 0.0];

        rmsnorm.forward(&mut x);

        assert_eq!(x, vec![0.0, 0.0]);
    }

    #[test]
    #[should_panic]
    fn forward_panics_on_length_mismatch() {
        let rmsnorm = RmsNorm::new(vec![1.0, 1.0, 1.0], 1e-6);
        let mut x = vec![1.0, 1.0, 1.0, 1.0];

        rmsnorm.forward(&mut x);
    }
}
