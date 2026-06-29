use rayon::prelude::*;

pub fn matmul(input: &[f32], weight: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    assert_eq!(input.len(), in_features);
    assert_eq!(weight.len(), out_features * in_features);

    (0..out_features)
        .into_par_iter()
        .map(|out_idx| {
            let row_start = out_idx * in_features;
            (0..in_features)
                .map(|in_idx| input[in_idx] * weight[row_start + in_idx])
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (*actual - *expected).abs() < 1e-5,
                "expected {actual} to be close to {expected}"
            );
        }
    }

    #[test]
    fn matmul_identity_matrix() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

        let output = matmul(&input, &weight, 3, 3);

        assert_vec_close(&output, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn matmul_known_values() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let output = matmul(&input, &weight, 2, 3);

        assert_vec_close(&output, &[1.0, 2.0]);
    }

    #[test]
    fn matmul_large_output() {
        let input = vec![1.0; 100];
        let weight = vec![1.0; 1000 * 100];

        let output = matmul(&input, &weight, 1000, 100);

        assert_eq!(output.len(), 1000);
        assert!(output.iter().all(|value| *value == 100.0));
    }
}
