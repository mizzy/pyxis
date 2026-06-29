#![cfg(target_os = "macos")]

use pyxis::matmul::matmul;
use pyxis::metal_matmul::MetalMatmul;
use pyxis::weights::Weights;

fn metal_matmul_or_skip() -> Option<MetalMatmul> {
    let metal = MetalMatmul::new();
    if metal.is_none() {
        eprintln!("Metal not available, skipping test");
    }
    metal
}

fn assert_vec_close(actual: &[f32], expected: &[f32]) {
    assert_vec_close_with_tolerance(actual, expected, 1e-4);
}

fn assert_vec_close_with_tolerance(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (*actual - *expected).abs() < tolerance,
            "expected {actual} to be close to {expected}"
        );
    }
}

fn cpu_matmul_f32(
    input: &[f32],
    weight: &[f32],
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), in_features);
    assert_eq!(weight.len(), out_features * in_features);

    (0..out_features)
        .map(|row| {
            let row_start = row * in_features;
            input
                .iter()
                .zip(&weight[row_start..row_start + in_features])
                .map(|(input, weight)| *input * *weight)
                .sum()
        })
        .collect()
}

#[test]
fn metal_matmul_identity() {
    let Some(metal) = metal_matmul_or_skip() else {
        return;
    };
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let output = metal.matmul(&input, &weight, 3, 3);

    assert_vec_close(&output, &input);
}

#[test]
fn metal_matmul_known_values() {
    let Some(metal) = metal_matmul_or_skip() else {
        return;
    };
    let input = vec![1.0, 2.0, 3.0];
    let weight = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    let output = metal.matmul(&input, &weight, 2, 3);

    assert_vec_close(&output, &[1.0, 2.0]);
}

#[test]
fn metal_matmul_large() {
    let Some(metal) = metal_matmul_or_skip() else {
        return;
    };
    let input = vec![1.0; 100];
    let weight = vec![1.0; 1000 * 100];

    let output = metal.matmul(&input, &weight, 1000, 100);

    assert_eq!(output.len(), 1000);
    assert!(output.iter().all(|value| *value == 100.0));
}

#[test]
fn metal_matmul_matches_cpu() {
    let Some(metal) = metal_matmul_or_skip() else {
        return;
    };
    let out_features = 16;
    let in_features = 17;
    let input: Vec<f32> = (0..in_features)
        .map(|index| (index as f32 % 7.0) - 3.0)
        .collect();
    let weight: Vec<f32> = (0..out_features * in_features)
        .map(|index| ((index * 17 + 11) % 23) as f32 / 7.0 - 1.5)
        .collect();

    let actual = metal.matmul(&input, &weight, out_features, in_features);
    let expected = matmul(&input, &Weights::F32(weight), out_features, in_features);

    assert_vec_close(&actual, &expected);
}

#[test]
fn metal_matmul_matches_cpu_transformer_dimensions() {
    let Some(metal) = metal_matmul_or_skip() else {
        return;
    };
    let out_features = 2048;
    let in_features = 2048;
    let input: Vec<f32> = (0..in_features)
        .map(|index| (index as f32 % 31.0) / 17.0 - 0.9)
        .collect();
    let weight: Vec<f32> = (0..out_features * in_features)
        .map(|index| ((index * 29 + 7) % 37) as f32 / 19.0 - 0.95)
        .collect();

    let actual = metal.matmul(&input, &weight, out_features, in_features);
    let expected = cpu_matmul_f32(&input, &weight, out_features, in_features);

    assert_vec_close_with_tolerance(&actual, &expected, 1e-2);
}

#[test]
fn metal_matmul_multi_tile() {
    let Some(metal) = metal_matmul_or_skip() else {
        return;
    };
    let out_features = 64;
    let in_features = 5504;
    let input: Vec<f32> = (0..in_features)
        .map(|index| (index as f32 % 43.0) / 23.0 - 0.85)
        .collect();
    let weight: Vec<f32> = (0..out_features * in_features)
        .map(|index| ((index * 31 + 13) % 47) as f32 / 29.0 - 0.8)
        .collect();

    let actual = metal.matmul(&input, &weight, out_features, in_features);
    let expected = cpu_matmul_f32(&input, &weight, out_features, in_features);

    assert_vec_close_with_tolerance(&actual, &expected, 1e-2);
}

#[test]
fn metal_buffer_matmul_matches_cpu() {
    let Some(metal) = metal_matmul_or_skip() else {
        return;
    };
    let out_features = 16;
    let in_features = 17;
    let input: Vec<f32> = (0..in_features)
        .map(|index| (index as f32 % 7.0) - 3.0)
        .collect();
    let weight: Vec<f32> = (0..out_features * in_features)
        .map(|index| ((index * 17 + 11) % 23) as f32 / 7.0 - 1.5)
        .collect();
    let weight_buffer = metal.create_buffer(&weight);

    let actual_with_buffer =
        metal.matmul_with_buffer(&input, &weight_buffer, out_features, in_features);
    let metal_weights = Weights::MetalF32 {
        buffer: weight_buffer,
        len: weight.len(),
    };
    let actual = matmul(&input, &metal_weights, out_features, in_features);
    let expected = matmul(&input, &Weights::F32(weight), out_features, in_features);

    assert_vec_close(&actual_with_buffer, &expected);
    assert_vec_close(&actual, &expected);
}

#[test]
fn weights_to_metal_f32() {
    let Some(metal) = metal_matmul_or_skip() else {
        return;
    };
    let weights = Weights::F32(vec![1.0, 2.0, 3.0, 4.0]);

    let metal_weights = weights.to_metal(&metal);

    assert_eq!(metal_weights.len(), 4);
    let Weights::MetalF32 { len, .. } = metal_weights else {
        panic!("expected MetalF32 weights");
    };
    assert_eq!(len, 4);
}

#[test]
fn weights_to_metal_bf16() {
    let Some(metal) = metal_matmul_or_skip() else {
        return;
    };
    let weights = Weights::Bf16(vec![
        (1.5_f32.to_bits() >> 16) as u16,
        ((-2.0_f32).to_bits() >> 16) as u16,
    ]);

    let metal_weights = weights.to_metal(&metal);

    assert_eq!(metal_weights.len(), 2);
    let Weights::MetalF32 { len, .. } = metal_weights else {
        panic!("expected MetalF32 weights");
    };
    assert_eq!(len, 2);
}
