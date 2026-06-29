use crate::matmul::matmul;

pub struct Ffn {
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
    hidden_dim: usize,
    intermediate_size: usize,
}

impl Ffn {
    pub fn new(
        gate_proj: Vec<f32>,
        up_proj: Vec<f32>,
        down_proj: Vec<f32>,
        hidden_dim: usize,
        intermediate_size: usize,
    ) -> Self {
        assert_eq!(gate_proj.len(), intermediate_size * hidden_dim);
        assert_eq!(up_proj.len(), intermediate_size * hidden_dim);
        assert_eq!(down_proj.len(), hidden_dim * intermediate_size);

        Self {
            gate_proj,
            up_proj,
            down_proj,
            hidden_dim,
            intermediate_size,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.hidden_dim);

        let mut gate = matmul(x, &self.gate_proj, self.intermediate_size, self.hidden_dim);
        let up = matmul(x, &self.up_proj, self.intermediate_size, self.hidden_dim);

        for (gate_value, up_value) in gate.iter_mut().zip(up) {
            *gate_value = silu(*gate_value) * up_value;
        }

        matmul(
            &gate,
            &self.down_proj,
            self.hidden_dim,
            self.intermediate_size,
        )
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
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

    fn assert_vec_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.iter().zip(expected) {
            assert_close(*actual, *expected);
        }
    }

    #[test]
    fn silu_at_zero_is_zero() {
        assert_close(silu(0.0), 0.0);
    }

    #[test]
    fn silu_at_positive_value() {
        assert_close(silu(1.0), 0.7310586);
    }

    #[test]
    fn silu_at_negative_value() {
        assert_close(silu(-1.0), -0.26894143);
    }

    #[test]
    fn forward_with_known_weights() {
        let ffn = Ffn::new(
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![2.0, 0.0, 0.0, -1.0, 0.5, 0.5],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            2,
            3,
        );

        let output = ffn.forward(&[1.0, 2.0]);

        assert_vec_close(&output, &[1.4621172, 0.7633953]);
    }

    #[test]
    #[should_panic]
    fn forward_panics_on_wrong_input_length() {
        let ffn = Ffn::new(vec![0.0; 6], vec![0.0; 6], vec![0.0; 6], 2, 3);

        ffn.forward(&[1.0, 2.0, 3.0]);
    }
}
