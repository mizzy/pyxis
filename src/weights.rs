#[derive(Debug, Clone, PartialEq)]
pub enum Weights {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
    Int8 {
        data: Vec<i8>,
        scales: Vec<f32>,
        row_size: usize,
    },
}

impl Weights {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::Bf16(values) => values.len(),
            Self::Int8 { data, .. } => data.len(),
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
}
