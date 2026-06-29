#[cfg(target_os = "macos")]
use metal::Buffer;

#[derive(Debug, Clone)]
pub enum Weights {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
    Int8 {
        data: Vec<i8>,
        scales: Vec<f32>,
        row_size: usize,
    },
    Int4 {
        data: Vec<u8>,
        scales: Vec<f32>,
        block_size: usize,
        num_elements: usize,
    },
    #[cfg(target_os = "macos")]
    MetalF32 {
        buffer: Buffer,
        len: usize,
    },
}

impl Weights {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::Bf16(values) => values.len(),
            Self::Int8 { data, .. } => data.len(),
            Self::Int4 { num_elements, .. } => *num_elements,
            #[cfg(target_os = "macos")]
            Self::MetalF32 { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32(&self) -> &[f32] {
        match self {
            Self::F32(values) => values,
            Self::Bf16(_) => panic!("cannot get f32 slice from bf16 weights"),
            Self::Int8 { .. } => panic!("cannot get f32 slice from int8 weights"),
            Self::Int4 { .. } => panic!("cannot get f32 slice from int4 weights"),
            #[cfg(target_os = "macos")]
            Self::MetalF32 { .. } => panic!("cannot get f32 slice from MetalF32 weights"),
        }
    }

    #[cfg(target_os = "macos")]
    pub fn to_metal(&self, metal: &crate::metal_matmul::MetalMatmul) -> Self {
        match self {
            Self::F32(data) => {
                let buffer = metal.create_buffer(data);
                Self::MetalF32 {
                    buffer,
                    len: data.len(),
                }
            }
            Self::Bf16(data) => {
                let f32_data: Vec<f32> = data
                    .iter()
                    .map(|&value| f32::from_bits((value as u32) << 16))
                    .collect();
                let buffer = metal.create_buffer(&f32_data);
                Self::MetalF32 {
                    buffer,
                    len: f32_data.len(),
                }
            }
            Self::Int8 { .. } | Self::Int4 { .. } | Self::MetalF32 { .. } => {
                panic!("to_metal only supports F32 and Bf16 weights")
            }
        }
    }

    pub fn quantize_int8(f32_values: &[f32], num_rows: usize, row_size: usize) -> Self {
        assert_eq!(f32_values.len(), num_rows * row_size);

        let mut data = Vec::with_capacity(f32_values.len());
        let mut scales = Vec::with_capacity(num_rows);

        for row in f32_values.chunks_exact(row_size) {
            let absmax = row.iter().map(|value| value.abs()).fold(0.0, f32::max);
            let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
            scales.push(scale);

            for &value in row {
                data.push((value / scale).round().clamp(-127.0, 127.0) as i8);
            }
        }

        Self::Int8 {
            data,
            scales,
            row_size,
        }
    }

    pub fn quantize_int4(f32_values: &[f32], block_size: usize) -> Self {
        assert!(block_size > 0);

        let num_elements = f32_values.len();
        let num_blocks = num_elements.div_ceil(block_size);
        let mut data = vec![0x88; num_elements.div_ceil(2)];
        let mut scales = Vec::with_capacity(num_blocks);

        for (block_idx, block) in f32_values.chunks(block_size).enumerate() {
            let absmax = block.iter().map(|value| value.abs()).fold(0.0, f32::max);
            let scale = if absmax == 0.0 { 1.0 } else { absmax / 7.0 };
            scales.push(scale);

            for (offset, &value) in block.iter().enumerate() {
                let global_idx = block_idx * block_size + offset;
                let quantized = ((value / scale).round().clamp(-8.0, 7.0) as i8 + 8) as u8;
                let byte = &mut data[global_idx / 2];
                if global_idx.is_multiple_of(2) {
                    *byte = (*byte & 0xf0) | quantized;
                } else {
                    *byte = (*byte & 0x0f) | (quantized << 4);
                }
            }
        }

        Self::Int4 {
            data,
            scales,
            block_size,
            num_elements,
        }
    }
}

impl From<Vec<f32>> for Weights {
    fn from(values: Vec<f32>) -> Self {
        Self::F32(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_len_f32() {
        let weights = Weights::F32(vec![0.0; 10]);

        assert_eq!(weights.len(), 10);
    }

    #[test]
    fn weights_len_bf16() {
        let weights = Weights::Bf16(vec![0; 10]);

        assert_eq!(weights.len(), 10);
    }

    #[test]
    fn weights_len_int8() {
        let weights = Weights::Int8 {
            data: vec![0; 10],
            scales: vec![1.0; 2],
            row_size: 5,
        };

        assert_eq!(weights.len(), 10);
    }

    #[test]
    fn weights_len_int4() {
        let weights = Weights::Int4 {
            data: vec![0; 5],
            scales: vec![1.0; 1],
            block_size: 32,
            num_elements: 10,
        };

        assert_eq!(weights.len(), 10);
    }

    #[test]
    fn quantize_int8_preserves_values_approximately() {
        let values = vec![1.0, 2.0, -3.0, 4.0];
        let weights = Weights::quantize_int8(&values, 1, 4);
        let Weights::Int8 {
            data,
            scales,
            row_size,
        } = weights
        else {
            panic!("expected int8 weights");
        };

        assert_eq!(row_size, 4);
        assert_eq!(scales.len(), 1);

        let dequantized: Vec<f32> = data.iter().map(|value| *value as f32 * scales[0]).collect();

        for (actual, expected) in dequantized.iter().zip(values) {
            assert!(
                (*actual - expected).abs() < 0.02,
                "expected {actual} to be close to {expected}"
            );
        }
    }

    #[test]
    fn quantize_int8_handles_zeros() {
        let weights = Weights::quantize_int8(&[0.0, 0.0, 0.0, 0.0], 2, 2);
        let Weights::Int8 {
            data,
            scales,
            row_size,
        } = weights
        else {
            panic!("expected int8 weights");
        };

        assert_eq!(data, vec![0, 0, 0, 0]);
        assert_eq!(scales, vec![1.0, 1.0]);
        assert_eq!(row_size, 2);
    }

    #[test]
    fn quantize_int8_scale_is_absmax_over_127() {
        let weights = Weights::quantize_int8(&[1.0, -2.0, 4.0, 0.0, -10.0, 5.0, 1.0, 2.0], 2, 4);
        let Weights::Int8 { scales, .. } = weights else {
            panic!("expected int8 weights");
        };

        assert!((scales[0] - 4.0 / 127.0).abs() < f32::EPSILON);
        assert!((scales[1] - 10.0 / 127.0).abs() < f32::EPSILON);
    }

    #[test]
    fn quantize_int4_preserves_values_approximately() {
        let values = vec![1.0, 2.0, -3.0, 4.0];
        let weights = Weights::quantize_int4(&values, 32);
        let Weights::Int4 {
            data,
            scales,
            block_size,
            num_elements,
        } = weights
        else {
            panic!("expected int4 weights");
        };

        assert_eq!(block_size, 32);
        assert_eq!(num_elements, values.len());
        assert_eq!(scales.len(), 1);

        let dequantized: Vec<f32> = (0..num_elements)
            .map(|index| {
                let byte = data[index / 2];
                let nibble = if index.is_multiple_of(2) {
                    byte & 0x0f
                } else {
                    (byte >> 4) & 0x0f
                };
                (nibble as f32 - 8.0) * scales[index / block_size]
            })
            .collect();

        for (actual, expected) in dequantized.iter().zip(values) {
            assert!(
                (*actual - expected).abs() < 0.5,
                "expected {actual} to be close to {expected}"
            );
        }
    }

    #[test]
    fn quantize_int4_handles_zeros() {
        let weights = Weights::quantize_int4(&[0.0, 0.0, 0.0, 0.0], 32);
        let Weights::Int4 {
            data,
            scales,
            block_size,
            num_elements,
        } = weights
        else {
            panic!("expected int4 weights");
        };

        assert_eq!(data, vec![0x88, 0x88]);
        assert_eq!(scales, vec![1.0]);
        assert_eq!(block_size, 32);
        assert_eq!(num_elements, 4);
    }

    #[test]
    fn quantize_int4_packing() {
        let weights = Weights::quantize_int4(&[7.0, -7.0], 32);
        let Weights::Int4 { data, scales, .. } = weights else {
            panic!("expected int4 weights");
        };

        assert_eq!(data, vec![0x1f]);
        assert_eq!(scales, vec![1.0]);
    }

    #[test]
    fn quantize_int4_block_size() {
        let weights = Weights::quantize_int4(&[1.0, 2.0, 3.0, 4.0, -5.0, 6.0, -7.0, 8.0], 4);
        let Weights::Int4 {
            scales,
            block_size,
            num_elements,
            ..
        } = weights
        else {
            panic!("expected int4 weights");
        };

        assert_eq!(block_size, 4);
        assert_eq!(num_elements, 8);
        assert_eq!(scales.len(), 2);
        assert!((scales[0] - 4.0 / 7.0).abs() < f32::EPSILON);
        assert!((scales[1] - 8.0 / 7.0).abs() < f32::EPSILON);
    }
}
