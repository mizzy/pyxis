use crate::weights::Weights;
use rayon::prelude::*;
#[cfg(target_os = "macos")]
use std::sync::OnceLock;

#[cfg(target_os = "macos")]
static METAL: OnceLock<Option<crate::metal_matmul::MetalMatmul>> = OnceLock::new();

#[cfg(target_os = "macos")]
fn get_metal() -> Option<&'static crate::metal_matmul::MetalMatmul> {
    METAL
        .get_or_init(crate::metal_matmul::MetalMatmul::new)
        .as_ref()
}

#[cfg(target_os = "macos")]
pub fn get_metal_ref() -> Option<&'static crate::metal_matmul::MetalMatmul> {
    get_metal()
}

pub fn matmul(
    input: &[f32],
    weight: &Weights,
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), in_features);
    assert_eq!(weight.len(), out_features * in_features);
    if let Weights::Int8 {
        block_size,
        num_elements,
        scales,
        ..
    } = weight
    {
        assert!(*block_size > 0);
        assert_eq!(*num_elements, out_features * in_features);
        assert_eq!(scales.len(), num_elements.div_ceil(*block_size));
    }
    if let Weights::Int4 {
        data,
        scales,
        block_size,
        num_elements,
    } = weight
    {
        assert!(*block_size > 0);
        assert_eq!(data.len(), num_elements.div_ceil(2));
        assert_eq!(scales.len(), num_elements.div_ceil(*block_size));
    }

    #[cfg(target_os = "macos")]
    if let Weights::MetalF32 { buffer, .. } = weight {
        if let Some(metal) = get_metal() {
            return metal.matmul_with_buffer(input, buffer, out_features, in_features);
        }
        panic!("MetalF32 weights require Metal device");
    }

    (0..out_features)
        .into_par_iter()
        .map(|out_idx| {
            let row_start = out_idx * in_features;
            match weight {
                Weights::F32(values) => {
                    dot_product(input, &values[row_start..row_start + in_features])
                }
                Weights::Bf16(values) => {
                    dot_product_bf16(input, &values[row_start..row_start + in_features])
                }
                Weights::Int8 {
                    data,
                    scales,
                    block_size,
                    ..
                } => dot_product_int8(input, data, scales, *block_size, row_start, in_features),
                Weights::Int4 {
                    data,
                    scales,
                    block_size,
                    ..
                } => dot_product_int4(input, data, scales, *block_size, row_start, in_features),
                #[cfg(target_os = "macos")]
                Weights::MetalF32 { .. } => unreachable!("MetalF32 weights are handled above"),
            }
        })
        .collect()
}

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "aarch64")]
    {
        dot_product_neon(a, b)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_product_scalar(a, b)
    }
}

fn dot_product_bf16(a: &[f32], b_bf16: &[u16]) -> f32 {
    assert_eq!(a.len(), b_bf16.len());

    #[cfg(target_arch = "aarch64")]
    {
        dot_product_bf16_neon(a, b_bf16)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_product_bf16_scalar(a, b_bf16)
    }
}

fn dot_product_int8(
    input: &[f32],
    quantized: &[i8],
    scales: &[f32],
    block_size: usize,
    offset: usize,
    len: usize,
) -> f32 {
    assert_eq!(input.len(), len);
    assert!(block_size > 0);

    #[cfg(target_arch = "aarch64")]
    {
        dot_product_int8_neon(input, quantized, scales, block_size, offset, len)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        dot_product_int8_scalar(input, quantized, scales, block_size, offset, len)
    }
}

fn dot_product_int4(
    input: &[f32],
    packed: &[u8],
    scales: &[f32],
    block_size: usize,
    offset: usize,
    len: usize,
) -> f32 {
    assert_eq!(input.len(), len);

    let mut sum = 0.0;
    for (index, input_value) in input.iter().enumerate() {
        let global_idx = offset + index;
        let byte = packed[global_idx / 2];
        let nibble = if global_idx.is_multiple_of(2) {
            byte & 0x0f
        } else {
            (byte >> 4) & 0x0f
        };
        let scale = scales[global_idx / block_size];
        sum += *input_value * (nibble as f32 - 8.0) * scale;
    }

    sum
}

#[cfg(target_arch = "aarch64")]
fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    assert_eq!(a.len(), b.len());

    let chunks = a.len() / 4;
    let mut sum = unsafe { vdupq_n_f32(0.0) };

    for chunk in 0..chunks {
        let offset = chunk * 4;
        let va = unsafe { vld1q_f32(a.as_ptr().add(offset)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(offset)) };
        sum = unsafe { vfmaq_f32(sum, va, vb) };
    }

    let mut result = unsafe { vaddvq_f32(sum) };

    for index in chunks * 4..a.len() {
        result += a[index] * b[index];
    }

    result
}

#[cfg(target_arch = "aarch64")]
fn dot_product_bf16_neon(a: &[f32], b_bf16: &[u16]) -> f32 {
    use std::arch::aarch64::*;

    assert_eq!(a.len(), b_bf16.len());

    let chunks = a.len() / 4;
    let mut sum = unsafe { vdupq_n_f32(0.0) };

    for chunk in 0..chunks {
        let offset = chunk * 4;
        let va = unsafe { vld1q_f32(a.as_ptr().add(offset)) };
        let b_u16 = unsafe { vld1_u16(b_bf16.as_ptr().add(offset)) };
        let b_u32 = unsafe { vmovl_u16(b_u16) };
        let b_shifted = unsafe { vshlq_n_u32(b_u32, 16) };
        let vb = unsafe { vreinterpretq_f32_u32(b_shifted) };
        sum = unsafe { vfmaq_f32(sum, va, vb) };
    }

    let mut result = unsafe { vaddvq_f32(sum) };

    for index in chunks * 4..a.len() {
        let b_f32 = f32::from_bits((b_bf16[index] as u32) << 16);
        result += a[index] * b_f32;
    }

    result
}

#[cfg(target_arch = "aarch64")]
fn dot_product_int8_neon(
    input: &[f32],
    quantized: &[i8],
    scales: &[f32],
    block_size: usize,
    offset: usize,
    len: usize,
) -> f32 {
    use std::arch::aarch64::*;

    assert_eq!(input.len(), len);
    assert!(block_size > 0);

    let mut result = 0.0;
    let mut processed = 0;

    while processed < len {
        let global_start = offset + processed;
        let block_idx = global_start / block_size;
        let block_end = ((block_idx + 1) * block_size).min(offset + len);
        let block_len = block_end - global_start;
        let chunks = block_len / 4;
        let mut block_sum = unsafe { vdupq_n_f32(0.0) };

        for chunk in 0..chunks {
            let local = processed + chunk * 4;
            let global_idx = offset + local;
            let va = unsafe { vld1q_f32(input.as_ptr().add(local)) };
            let quantized_f32 = [
                quantized[global_idx] as f32,
                quantized[global_idx + 1] as f32,
                quantized[global_idx + 2] as f32,
                quantized[global_idx + 3] as f32,
            ];
            let vb = unsafe { vld1q_f32(quantized_f32.as_ptr()) };
            block_sum = unsafe { vfmaq_f32(block_sum, va, vb) };
        }

        let mut block_result = unsafe { vaddvq_f32(block_sum) };
        let tail_start = processed + chunks * 4;
        let tail_end = processed + block_len;
        for (tail_offset, input_value) in input[tail_start..tail_end].iter().enumerate() {
            let local = tail_start + tail_offset;
            let global_idx = offset + local;
            block_result += *input_value * quantized[global_idx] as f32;
        }

        result += block_result * scales[block_idx];
        processed += block_len;
    }

    result
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_product_bf16_scalar(a: &[f32], b_bf16: &[u16]) -> f32 {
    a.iter()
        .zip(b_bf16)
        .map(|(a, b)| a * f32::from_bits((*b as u32) << 16))
        .sum()
}

#[cfg(not(target_arch = "aarch64"))]
fn dot_product_int8_scalar(
    input: &[f32],
    quantized: &[i8],
    scales: &[f32],
    block_size: usize,
    offset: usize,
    len: usize,
) -> f32 {
    assert_eq!(input.len(), len);
    assert!(block_size > 0);

    input
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let global_idx = offset + index;
            input * quantized[global_idx] as f32 * scales[global_idx / block_size]
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::Weights;

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
        let weight = Weights::F32(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

        let output = matmul(&input, &weight, 3, 3);

        assert_vec_close(&output, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn matmul_known_values() {
        let input = vec![1.0, 2.0, 3.0];
        let weight = Weights::F32(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

        let output = matmul(&input, &weight, 2, 3);

        assert_vec_close(&output, &[1.0, 2.0]);
    }

    #[test]
    fn matmul_large_output() {
        let input = vec![1.0; 100];
        let weight = Weights::F32(vec![1.0; 1000 * 100]);

        let output = matmul(&input, &weight, 1000, 100);

        assert_eq!(output.len(), 1000);
        assert!(output.iter().all(|value| *value == 100.0));
    }

    #[test]
    fn dot_product_matches_scalar() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let output = dot_product(&input, &weight);

        assert_eq!(output, 70.0);
    }

    #[test]
    fn dot_product_large_aligned() {
        let input = vec![1.0; 1024];
        let weight = vec![1.0; 1024];

        let output = dot_product(&input, &weight);

        assert_eq!(output, 1024.0);
    }

    #[test]
    fn dot_product_with_remainder() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let weight = vec![7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let output = dot_product(&input, &weight);

        assert_eq!(output, 84.0);
    }

    #[test]
    fn dot_product_bf16_matches_f32() {
        let input = vec![1.0, -2.0, 0.5, 4.0, 3.0];
        let weight = vec![1.5, -0.25, 2.0, 0.125, -1.0];
        let weight_bf16: Vec<u16> = weight
            .iter()
            .map(|value: &f32| (value.to_bits() >> 16) as u16)
            .collect();

        let actual = dot_product_bf16(&input, &weight_bf16);
        let expected = dot_product(
            &input,
            &weight_bf16
                .iter()
                .map(|value| f32::from_bits((*value as u32) << 16))
                .collect::<Vec<_>>(),
        );

        assert!((actual - expected).abs() < 1e-5);
    }

    #[test]
    fn dot_product_int8_matches_f32() {
        let input = vec![0.5, -2.0, 1.5, 4.0];
        let weight = vec![1.0, -0.5, 2.0, -3.0];
        let Weights::Int8 {
            data,
            scales,
            block_size,
            ..
        } = Weights::quantize_int8(&weight, 4)
        else {
            panic!("expected int8 weights");
        };

        let actual = dot_product_int8(&input, &data, &scales, block_size, 0, input.len());
        let expected = dot_product(&input, &weight);

        assert!(
            (actual - expected).abs() < 0.05,
            "expected {actual} to be close to {expected}"
        );
    }

    #[test]
    fn matmul_with_bf16_weights() {
        let input = vec![1.0, 2.0, -1.0];
        let weight_f32 = vec![1.5, -0.5, 2.0, 0.25, 3.0, -1.0];
        let weight_bf16 = weight_f32
            .iter()
            .map(|value: &f32| (value.to_bits() >> 16) as u16)
            .collect();

        let output = matmul(&input, &Weights::Bf16(weight_bf16), 2, 3);
        let expected = matmul(&input, &Weights::F32(weight_f32), 2, 3);

        assert_vec_close(&output, &expected);
    }

    #[test]
    fn matmul_with_int8_weights() {
        let input = vec![1.0, 2.0, -1.0];
        let weight_f32 = vec![1.5, -0.5, 2.0, 0.25, 3.0, -1.0];
        let weight_int8 = Weights::quantize_int8(&weight_f32, 3);

        let output = matmul(&input, &weight_int8, 2, 3);
        let expected = matmul(&input, &Weights::F32(weight_f32), 2, 3);

        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected) {
            assert!(
                (*actual - expected).abs() < 0.05,
                "expected {actual} to be close to {expected}"
            );
        }
    }

    #[test]
    fn matmul_with_int4_weights() {
        let input = vec![1.0, 2.0, -1.0];
        let weight_f32 = vec![1.5, -0.5, 2.0, 0.25, 3.0, -1.0];
        let weight_int4 = Weights::quantize_int4(&weight_f32, 4);

        let output = matmul(&input, &weight_int4, 2, 3);
        let expected = matmul(&input, &Weights::F32(weight_f32), 2, 3);

        assert_eq!(output.len(), expected.len());
        for (actual, expected) in output.iter().zip(expected) {
            assert!(
                (*actual - expected).abs() < 0.5,
                "expected {actual} to be close to {expected}"
            );
        }
    }
}
