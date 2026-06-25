use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Dtype {
    F32,
    F16,
    BF16,
}

impl Dtype {
    pub fn element_size(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

#[derive(Debug, Deserialize)]
struct RawTensorInfo {
    dtype: serde_json::Value,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Debug)]
pub struct SafeTensors {
    mmap: memmap2::Mmap,
    data_offset: usize,
    tensors: HashMap<String, TensorInfo>,
}

impl SafeTensors {
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "safetensors file is too short to contain a header size",
            ));
        }

        let mut size_buf = [0u8; 8];
        size_buf.copy_from_slice(&mmap[..8]);
        let header_size = u64::from_le_bytes(size_buf) as usize;
        let data_offset = 8usize
            .checked_add(header_size)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "header size overflow"))?;

        if mmap.len() < data_offset {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "safetensors file is too short to contain the declared header",
            ));
        }

        let header_json: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&mmap[8..data_offset])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut tensors = HashMap::new();
        for (key, value) in header_json {
            if key == "__metadata__" {
                continue;
            }
            let raw: RawTensorInfo = serde_json::from_value(value)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            let dtype = match serde_json::from_value(raw.dtype) {
                Ok(dtype) => dtype,
                Err(_) => continue,
            };
            let info = TensorInfo {
                dtype,
                shape: raw.shape,
                data_offsets: raw.data_offsets,
            };
            tensors.insert(key, info);
        }

        Ok(Self {
            mmap,
            data_offset,
            tensors,
        })
    }

    pub fn tensor_f32(&self, name: &str) -> io::Result<Vec<f32>> {
        let info = self.tensors.get(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("tensor not found: {name}"))
        })?;

        let [start, end] = info.data_offsets;
        let start = self.data_offset.checked_add(start).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "tensor start offset overflow")
        })?;
        let end = self.data_offset.checked_add(end).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "tensor end offset overflow")
        })?;

        if start > end || end > self.mmap.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "tensor data offsets are outside the file",
            ));
        }

        let bytes = &self.mmap[start..end];
        if !bytes.len().is_multiple_of(info.dtype.element_size()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "tensor byte length is not aligned to dtype element size",
            ));
        }

        let values = match info.dtype {
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            Dtype::BF16 => bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
        };

        Ok(values)
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    pub fn tensors(&self) -> &HashMap<String, TensorInfo> {
        &self.tensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_safetensors(tensors: serde_json::Value, data: &[u8]) -> NamedTempFile {
        let header_bytes = serde_json::to_vec(&tensors).unwrap();
        let header_size = (header_bytes.len() as u64).to_le_bytes();

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&header_size).unwrap();
        file.write_all(&header_bytes).unwrap();
        file.write_all(data).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn dtype_element_size_returns_byte_widths() {
        assert_eq!(Dtype::F32.element_size(), 4);
        assert_eq!(Dtype::F16.element_size(), 2);
        assert_eq!(Dtype::BF16.element_size(), 2);
    }

    #[test]
    fn dtype_deserializes_from_json_strings() {
        assert_eq!(
            serde_json::from_str::<Dtype>("\"F32\"").unwrap(),
            Dtype::F32
        );
        assert_eq!(
            serde_json::from_str::<Dtype>("\"F16\"").unwrap(),
            Dtype::F16
        );
        assert_eq!(
            serde_json::from_str::<Dtype>("\"BF16\"").unwrap(),
            Dtype::BF16
        );
    }

    #[test]
    fn load_returns_tensor_metadata() {
        let file = create_test_safetensors(
            serde_json::json!({
                "__metadata__": {"format": "pt"},
                "weight": {
                    "dtype": "F32",
                    "shape": [3, 4],
                    "data_offsets": [0, 48]
                },
                "bias": {
                    "dtype": "BF16",
                    "shape": [4],
                    "data_offsets": [48, 56]
                }
            }),
            &[0u8; 56],
        );

        let tensors = SafeTensors::load(file.path()).unwrap();

        assert_eq!(tensors.tensor_names().len(), 2);
        assert!(tensors.tensor_info("__metadata__").is_none());

        let weight = tensors.tensor_info("weight").unwrap();
        assert_eq!(weight.dtype, Dtype::F32);
        assert_eq!(weight.shape, vec![3, 4]);
        assert_eq!(weight.data_offsets, [0, 48]);

        let bias = tensors.tensor_info("bias").unwrap();
        assert_eq!(bias.dtype, Dtype::BF16);
        assert_eq!(bias.shape, vec![4]);
        assert_eq!(bias.data_offsets, [48, 56]);
    }

    #[test]
    fn load_skips_tensor_with_unsupported_dtype() {
        let data: Vec<u8> = [1.0f32, 2.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .chain([1i64, 2].iter().flat_map(|value| value.to_le_bytes()))
            .collect();
        let file = create_test_safetensors(
            serde_json::json!({
                "float_values": {
                    "dtype": "F32",
                    "shape": [2],
                    "data_offsets": [0, 8]
                },
                "int_values": {
                    "dtype": "I64",
                    "shape": [2],
                    "data_offsets": [8, 24]
                }
            }),
            &data,
        );

        let tensors = SafeTensors::load(file.path()).unwrap();

        assert_eq!(tensors.tensor_f32("float_values").unwrap(), vec![1.0, 2.0]);
        assert!(tensors.tensor_info("int_values").is_none());
    }

    #[test]
    fn load_returns_error_for_nonexistent_file() {
        let err = SafeTensors::load(Path::new("/tmp/pyxis-missing-file.safetensors")).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn tensor_f32_reads_f32_tensor_data() {
        let values = [1.0f32, 2.0, 3.0];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let file = create_test_safetensors(
            serde_json::json!({
                "values": {
                    "dtype": "F32",
                    "shape": [3],
                    "data_offsets": [0, 12]
                }
            }),
            &data,
        );

        let tensors = SafeTensors::load(file.path()).unwrap();

        assert_eq!(tensors.tensor_f32("values").unwrap(), values);
    }

    #[test]
    fn tensor_f32_reads_bf16_tensor_data() {
        let values = [1.5f32, -2.25, 3.5];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
            .collect();
        let file = create_test_safetensors(
            serde_json::json!({
                "values": {
                    "dtype": "BF16",
                    "shape": [3],
                    "data_offsets": [0, 6]
                }
            }),
            &data,
        );

        let tensors = SafeTensors::load(file.path()).unwrap();
        let loaded = tensors.tensor_f32("values").unwrap();

        assert_eq!(loaded.len(), values.len());
        for (actual, expected) in loaded.iter().zip(values) {
            assert!((actual - expected).abs() < 0.01);
        }
    }

    #[test]
    fn tensor_f32_returns_error_for_unknown_tensor() {
        let file = create_test_safetensors(
            serde_json::json!({
                "values": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4]
                }
            }),
            &1.0f32.to_le_bytes(),
        );

        let tensors = SafeTensors::load(file.path()).unwrap();
        let err = tensors.tensor_f32("missing").unwrap_err();

        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn tensors_returns_all_metadata() {
        let file = create_test_safetensors(
            serde_json::json!({
                "weight": {
                    "dtype": "F32",
                    "shape": [3, 4],
                    "data_offsets": [0, 48]
                }
            }),
            &[0u8; 48],
        );

        let tensors = SafeTensors::load(file.path()).unwrap();

        assert_eq!(tensors.tensors().len(), 1);
        assert!(tensors.tensors().contains_key("weight"));
    }

    #[test]
    fn tensor_f32_reads_f16_tensor_data() {
        let values = [1.5f32, -2.25, 3.5];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|value| half::f16::from_f32(*value).to_le_bytes())
            .collect();
        let file = create_test_safetensors(
            serde_json::json!({
                "values": {
                    "dtype": "F16",
                    "shape": [3],
                    "data_offsets": [0, 6]
                }
            }),
            &data,
        );

        let tensors = SafeTensors::load(file.path()).unwrap();
        let loaded = tensors.tensor_f32("values").unwrap();

        assert_eq!(loaded.len(), values.len());
        for (actual, expected) in loaded.iter().zip(values) {
            assert!((actual - expected).abs() < 0.001);
        }
    }

    #[test]
    fn load_ignores_metadata_entry() {
        let file = create_test_safetensors(
            serde_json::json!({
            "__metadata__": {"format": "pt"},
            "weight": {
                "dtype": "F32",
                "shape": [3, 4],
                "data_offsets": [0, 48]
            }
            }),
            &[0u8; 48],
        );
        let tensors = SafeTensors::load(file.path()).unwrap();
        assert!(!tensors.tensors().contains_key("__metadata__"));
    }
}
