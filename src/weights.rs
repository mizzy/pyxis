#[derive(Debug, Clone, PartialEq)]
pub enum Weights {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

impl Weights {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::Bf16(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32(&self) -> &[f32] {
        match self {
            Self::F32(values) => values,
            Self::Bf16(_) => panic!("cannot get f32 slice from bf16 weights"),
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
}
